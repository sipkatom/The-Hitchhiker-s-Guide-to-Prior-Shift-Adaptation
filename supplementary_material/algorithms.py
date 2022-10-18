import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix

def hard_confusion_matrix(predictions, targets):
    ''' Compute conditional confusion matrix from classifier's predictions
        and ground truth labels.
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        targets: torch.Tensor (num_data,) with ground truth labels coresponding to the predictions
    Returns:
        mat: torch.Tensor (num_classes, num_classes) with hard conditional confusion matrix
    '''
    num_classes = predictions.shape[1]
    
    y = torch.argmax(predictions, dim=1)
    
    mat = confusion_matrix(targets.numpy(), y.numpy(), normalize='true', labels=np.arange(num_classes)).T
    mat = torch.from_numpy(mat)
    
    return mat.float()

def soft_confusion_matrix(predictions, targets):
    ''' Compute soft confusion matrix from classifier's predictions
        and ground truth labels.
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        targets: torch.Tensor (num_data,) with ground truth labels coresponding to the predictions
    Returns:
        mat: torch.Tensor (num_classes, num_classes) with soft confusion matrix
    '''
    num_classes = predictions.shape[1]
    
    mat = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for i in range(num_classes):
        mask = targets == i
        m = torch.mean(predictions[mask,:], dim=0)
        mat[:,i] = m
    
    return mat

def joint_confusion_matrix(predictions, targets, weights):
    ''' Compute joint confusion matrix (for BBSE) from classifier's predictions
        and ground truth labels.
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        targets: torch.Tensor (num_data,) with ground truth labels coresponding to the predictions
    Returns:
        mat: torch.Tensor (num_classes, num_classes) with hard conditional confusion matrix
    '''
    num_classes = predictions.shape[1]

    y = torch.argmax(predictions, dim=1)
    
    mat = confusion_matrix(targets.numpy(), y.numpy(), normalize='all', labels=np.arange(num_classes)).T
    mat = torch.from_numpy(mat)

    mat = mat*weights
    mat = mat / torch.sum(mat)
    
    return mat.float()

def compute_joint_soft_confusion_matrix(predictions, targets, weights):
    num_classes = predictions.shape[1]
    
    mat = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for i in range(num_classes):
        mask = targets == i
        m = torch.mean(predictions[mask,:], dim=0)
        mat[:,i] = m
    
    mat = mat*weights
    
    return mat / torch.sum(mat)

def count_classes(targets, num_classes):
    """ Count number of samples per class in labeled dataset.

    Args:
        targets: torch.Tensor (num_data,) with ground truth labels in the dataset
        num_classes: int representing number of classes in the dataset
    Returns:
        counts: torch.Tensor (num_classes, ) with number of samples per class
    """
    counts = torch.zeros(num_classes)
    for i in range(num_classes):
        counts[i] = (targets == i).sum().float()
    return counts

def accuracy(predictions, gt):
    """ Compute accuracy given predictions and ground truth labels.

    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with output predictions
        gt: torch.Tensor (num_data) with ground truth labels.
    Returns:
        accuracy: float with classifier accuracy
    """
    size = gt.shape[0]
    predictions = torch.argmax(predictions, dim=1)
    acc = torch.sum(predictions == gt).double()/size
    return (acc*100).item()

def adjust_predictions(predictions, trainset_priors, test_set_distribution=None):
    """ Adjust classifier's predictions to prior shift,
        knowing the training set distribution and a different test set distribution.
        I.e. predictions are multiplied by the ratio of class priors.
        
        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,)
        test_set_distribution: torch.Tensor (num_classes,); if None - use uniform distribution
    Returns:
        adjust_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
    """
    if test_set_distribution is None:
        test_set_distribution = torch.ones(trainset_priors.shape)
    adjusted_predictions = predictions * test_set_distribution / trainset_priors
    adjusted_predictions = adjusted_predictions / torch.sum(adjusted_predictions, dim=1).unsqueeze(1) # normalize to sum to 1
    return adjusted_predictions

def simplex_projection(y):
    """
    Projection onto the probability simplex, based on https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf

    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    u = -np.sort(-y.numpy()) # sort y in descending order
    j = np.arange(1, len(y)+1)
    phi_obj = u + 1/j * (1-np.cumsum(u))
    positive = np.argwhere(phi_obj > 0)
    if positive.size == 0: raise ValueError("Numerical issues - extremely large values after update.. DECREASE LEARNING RATE")
    phi = positive.max() + 1
    lam = 1/phi * (1-np.sum(u[:phi]))
    x = np.maximum(y+lam,0)

    return torch.Tensor(x)

###############
# Calibration #
###############

def learn_calibration(model_outputs, targets, lr, iters, weights):
    ''' Implements Bias-Corrected Temperature Scaling (BCTS) from https://arxiv.org/pdf/1901.06852.pdf.
        
        Code modified from:
        https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Args:
        model_outputs: torch.Tensor (num_data, num_classes) with outputs of the model before softmax (logits)
        targets: torch.Tensor (num_data,) with ground truth labels coresponding to the predictions
        lr: float representing learning rate
        iters: int specifying number of iterartions
    Returns:
        T: float with learned temperarture
        b: torch.Tensor (num_classes,) with learned biases
    '''
    T = torch.tensor([1.], requires_grad=True)
    b = torch.ones(model_outputs.shape[1], requires_grad=True)
    
    nll_criterion = nn.CrossEntropyLoss(weight=weights)

    before_temperature_nll = nll_criterion(model_outputs, targets).item()
    
    print('Before calibration - NLL: %.3f ' % (before_temperature_nll))
    
    optimizer = optim.LBFGS([T, b], lr=lr, max_iter=iters)
    
    def eval():
        loss = nll_criterion(model_outputs/T + b, targets)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(model_outputs/T + b, targets).item()
    print('After calibration - NLL: %.3f ' % (after_temperature_nll))

    return T.item(), b.detach()

################
# EM algorithm #
################

def estimate_priors_from_predictions(predictions):
    """ Estimate class priors from predictions.
        
        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
    Returns:
        priors: torch.Tensor (num_classes) with estimated class priors
    """
    
    priors = torch.mean(predictions, dim=0)
    return priors


def EM_priors_estimation(predictions, trainset_priors, test_init_distribution=None, termination_difference=0.0001, verbose=False):
    """ EM algorithm for test set prior estimation and adjust classifier's predictions 
        to prior shift.

        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        test_init_distribution: torch.Tensor (num_classes,) to initialize test set distribution.
                                If None, use trainset_priors.
        termination_error: float defining the distance of posterior predictions for termination.
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    """
    if test_init_distribution is None:
        test_init_distribution = trainset_priors.detach().clone()
        
    testset_priors = test_init_distribution / torch.sum(test_init_distribution)
    step = 0

    while True:
        step += 1
        new_predictions = adjust_predictions(predictions, trainset_priors, testset_priors)
        new_testset_priors = estimate_priors_from_predictions(new_predictions)

        difference = torch.sum((new_testset_priors - testset_priors)**2)
        if verbose: print("EM step ", step, "; diff. %.8f" % (difference))
        if difference < termination_difference*termination_difference:
            if verbose: print("Finished. Difference", difference, "< termination value", termination_difference)
            break
        testset_priors = new_testset_priors
        
    return new_predictions, new_testset_priors

#########################
#  MLE and MAP Estimate #
#########################

# Projected Gradient Ascent for ML  estimate is applied by iteratively running next_step_projectedGA()
# Projected Gradient Ascent for MAP estimate is applied by iteratively running next_step_projectedGA_with_prior()
  
def compute_gradient(x,a):
    """
    Compute gradient from Eq. 12 from:
    http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Sulc_Improving_CNN_Classifiers_by_Estimating_Test-Time_Priors_ICCVW_2019_paper.pdf
    
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    d = torch.sum(a*x, dim=1)
    g = torch.sum(a*(1/d.unsqueeze(1)), dim=0)
    return g

def log_dirichlet_gradient(x, alpha, numerical_min_prior=1e-8):
    """
    Compute gradient from Eq. 15 from:
    http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Sulc_Improving_CNN_Classifiers_by_Estimating_Test-Time_Priors_ICCVW_2019_paper.pdf
    
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    g = (alpha - 1) / torch.max(input=x, other=torch.Tensor([numerical_min_prior]))
    return g

def next_step_projectedGA(x, a, learning_rate):
    """
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    g = compute_gradient(x,a)
    nx = x + learning_rate * g
    nx = simplex_projection(nx)
    nx = nx / nx.sum()
    return nx

def next_step_projectedGA_with_prior(x, a, learning_rate, alpha, prior_relative_weight = 1.0):
    """
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    g = compute_gradient(x,a)
    g_prior = log_dirichlet_gradient(x, alpha)
    nx = x + learning_rate * (g + prior_relative_weight * g_prior)
    nx = simplex_projection(nx)
    nx = nx / nx.sum()
    return nx

def MLE_estimate(predictions, trainset_priors, num_iter, test_init_distribution=None, lr=1e-7, termination_difference=0.0001):
    ''' 
    Maximum likelihood estimate according to 
    http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Sulc_Improving_CNN_Classifiers_by_Estimating_Test-Time_Priors_ICCVW_2019_paper.pdf
    
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        num_iter: int max. number of iterations
        test_init_distribution: torch.Tensor (num_classes,) to initialize test set distribution.
                                If None, use trainset_priors.
        termination_difference: float defining the distance of posterior predictions for termination.
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    '''
    mask = (trainset_priors == 0)
    a = predictions/torch.where(mask, torch.ones_like(trainset_priors), trainset_priors)
    a[:,mask] = 0

    if test_init_distribution is None:
        testset_priors = trainset_priors.detach().clone()
    else:
        testset_priors = test_init_distribution
    testset_priors = testset_priors / torch.sum(testset_priors)

    for iteration in range(int(num_iter)):
        new_testset_priors = next_step_projectedGA(testset_priors, a, learning_rate=lr)

        difference = torch.sum((new_testset_priors - testset_priors)**2)
        if difference < termination_difference*termination_difference:
            break
        testset_priors = new_testset_priors
    
    new_predictions = adjust_predictions(predictions, trainset_priors, testset_priors)
    return new_predictions, new_testset_priors

def MAP_estimate(predictions, trainset_priors, num_iter, test_init_distribution=None, lr=1e-8, termination_difference=0.0001, alpha=3):
    ''' 
    Maximum aposteriori estimate according to 
    http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Sulc_Improving_CNN_Classifiers_by_Estimating_Test-Time_Priors_ICCVW_2019_paper.pdf
    
    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        test_init_distribution: torch.Tensor (num_classes,) to initialize test set distribution.
                                If None, use trainset_priors.
        termination_difference: float defining the distance of posterior predictions for termination.
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    '''
    mask = (trainset_priors == 0)
    a = predictions/torch.where(mask, torch.ones_like(trainset_priors), trainset_priors)
    a[:,mask] = 0

    if test_init_distribution is None:
        map_priors = trainset_priors.detach().clone()
    else:
        map_priors = test_init_distribution
    map_priors = map_priors / torch.sum(map_priors)
    
    for iteration in range(int(num_iter)):
        testset_priors = next_step_projectedGA_with_prior(map_priors, a, alpha=alpha, learning_rate=lr)

        difference = torch.sum((testset_priors - map_priors)**2)
        if difference < termination_difference*termination_difference:
            break
        map_priors = testset_priors
    
    new_predictions = adjust_predictions(predictions, trainset_priors, map_priors)
    return new_predictions, testset_priors


###########################
#  CONFUSION MATRIX BASED #
###########################

def CM_estimate(predictions, confusion_matrix, soft=False):
    ''' 
    Test set prior estimation using confusion matrix inversion (Equation 4 in the paper).
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        confusion_matrix: torch.Tensor (num_classes, num_classes) 
        soft: bool indicator for soft confusion matrix
    Returns:
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    '''
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = torch.zeros(num_classes)
        class_pred = torch.argmax(predictions, dim=1)
        for i in range(num_classes):
            decision_distribution[i] = torch.sum(class_pred == i)
    else:
        decision_distribution = torch.sum(predictions, dim=0)

    decision_distribution = decision_distribution / torch.sum(decision_distribution)
    
    if torch.matrix_rank(confusion_matrix) == num_classes:
        conf_inv = torch.inverse(confusion_matrix)
    else:
        conf_inv = torch.pinverse(confusion_matrix)
        
    priors_estimate = (conf_inv @ decision_distribution.unsqueeze(1)).squeeze()
    priors_estimate[priors_estimate < 0] = 0
    priors_estimate = priors_estimate / torch.sum(priors_estimate)
    
    return priors_estimate

def BBSE_estimate(predictions, confusion_matrix, soft=False):
    ''' 
    Test set prior estimation using confusion matrix inversion (Equation 4 in the paper).
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        confusion_matrix: torch.Tensor (num_classes, num_classes) 
        soft: bool indicator for soft confusion matrix
    Returns:
        w: torch.Tensor (num_classes,) estimated priors ratio
    '''
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = torch.zeros(num_classes)
        class_pred = torch.argmax(predictions, dim=1)
        for i in range(num_classes):
            decision_distribution[i] = torch.sum(class_pred == i)
    else:
        decision_distribution = torch.sum(predictions, dim=0)

    decision_distribution = decision_distribution / torch.sum(decision_distribution)
    
    if torch.matrix_rank(confusion_matrix) == num_classes:
        conf_inv = torch.inverse(confusion_matrix)
    else:
        conf_inv = torch.pinverse(confusion_matrix)
        
    w = (conf_inv @ decision_distribution.unsqueeze(1)).squeeze()
    
    return w


def matrix_correction_MLE(predictions, trainset_priors, confusion_matrix, soft=False, max_iter=1000, lr=1e-3):
    ''' 
        Maximum likelihood estimation of test set priors using confusion matrix, proposed in Section 3.1.
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        confusion_matrix: torch.Tensor (num_classes, num_classes)
        soft: bool indicator for soft condusion matrix
        max_iter: int max. number of iterations
        lr: float learning rate
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    '''
    d = 2 # number to divide lr
    num = 50 # lr will be devided each num iterations by d
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = torch.zeros(num_classes)
        class_pred = torch.argmax(predictions, dim=1)
        for i in range(num_classes):
            decision_distribution[i] = torch.sum(class_pred == i)
    else:
        decision_distribution = torch.sum(predictions, dim=0)
    
    decision_distribution = decision_distribution / torch.sum(decision_distribution)

    new_testset_priors = trainset_priors.detach().clone()
    new_testset_priors = new_testset_priors/torch.sum(new_testset_priors)
    for i in range(int(max_iter)):
        o = confusion_matrix @ new_testset_priors
        mask = (o == 0)
        grad = decision_distribution / torch.where(mask, torch.ones_like(o), o)
        grad[mask] = 0
        
        grad = confusion_matrix*grad.unsqueeze(1)
        grad = grad.sum(0)
        
        lr_cur = lr/d**(i//num)
        p_updated = new_testset_priors + lr_cur*grad
        new_testset_priors = simplex_projection(p_updated)

    
    new_predictions = adjust_predictions(predictions, trainset_priors, new_testset_priors)
        
    return new_predictions, new_testset_priors

def matrix_correction_MAP(predictions, trainset_priors, confusion_matrix, alpha=3, soft=False, max_iter=1000, lr=0.01):
    ''' 
    Maximum a-posteriori estimation of test set priors with a Dirichlet hyperprior, using confusion matrix, 
    proposed in Section 3.2.
    Args:
        predictions: torch.Tensor (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        confusion_matrix: torch.Tensor (num_classes, num_classes) to initialize test set distribution.
                                If None, use trainset_priors.
        alpha: float hyperprior of dirichlet distribution
        soft: bool indicator for soft condusion matrix
        max_iter: int max. number of iterations
        lr: float learning rate
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    '''
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = torch.zeros(num_classes)
        class_pred = torch.argmax(predictions, dim=1)
        for i in range(num_classes):
            decision_distribution[i] = torch.sum(class_pred == i)
    else:
        decision_distribution = torch.sum(predictions, dim=0)
    
    N = torch.sum(decision_distribution)
    decision_distribution = decision_distribution / N
    
    new_testset_priors = trainset_priors.detach().clone()
    
    new_testset_priors = new_testset_priors/torch.sum(new_testset_priors)
    for i in range(int(max_iter)):
        o = confusion_matrix @ new_testset_priors
        mask = (o == 0)
        grad_l = decision_distribution/torch.where(mask, torch.ones_like(o), o)
        grad_l[mask] = 0
        
        grad_l = confusion_matrix*grad_l.unsqueeze(1)
        grad_l = grad_l.sum(0)
        
        grad_a = ((alpha-1)/new_testset_priors)
        
        grad = grad_a/N + grad_l # divide N because decision_distribution are normalized
        
        p_updated = new_testset_priors + lr*grad
        new_testset_priors = simplex_projection(p_updated)

    new_predictions = adjust_predictions(predictions, trainset_priors, new_testset_priors)

    return new_predictions, new_testset_priors
