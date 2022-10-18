import numpy as np
from .tools import adjust_predictions, simplex_projection, soft_confusion_matrix, hard_confusion_matrix

def estimate_priors(predictions, val_pred, val_targets, init_prior, method="SCM-L", args=dict()):
    """ Estimates class distribution test set.

    Args:
        predictions: np.array (num_data, num_classes) with predictions on test set
        val_pred: np.array (num_data, num_classes) with predictions on validation set
        val_targets: np.array (num_data,) with ground truth labels coresponding 
                    to the predictions on validation dataset
        init_prior: np.array (num_classes) vector of priors used to initialize the algorithm,
                    for EM algorithm, it is has to be prior on training set.
        method: str, the algorithm used for priors estimation
        args: Dict[str, Any], arguments for the algorithm
    
    Returns:
        priors: np.array (num_classes) with estimated class priors    
    """
    if method == "EM":
        termination_difference = args.get("termination_difference", 0.001)
        _, priors = EM_priors_estimation(predictions, init_prior, termination_difference)
    elif method == "CM":
        cm = hard_confusion_matrix(val_pred, val_targets)
        priors = CM_estimate(predictions, cm, False)
    elif method == "CM-L":
        max_iter = args.get("max_iter", 1000)
        lr = args.get("lr", 1e-4)
        cm = hard_confusion_matrix(val_pred, val_targets)
        priors = matrix_correction_MLE(predictions, init_prior, cm, False, max_iter, lr)
    elif method == "CM-M":
        max_iter = args.get("max_iter", 1000)
        lr = args.get("lr", 1e-3)
        alpha = args.get("alpha", 3)
        cm = hard_confusion_matrix(val_pred, val_targets)
        priors = matrix_correction_MAP(predictions, init_prior, cm, alpha, False, max_iter, lr)
    elif method == "SCM":
        cm = soft_confusion_matrix(val_pred, val_targets)
        priors = CM_estimate(predictions, cm, True)
    elif method == "SCM-L":
        max_iter = args.get("max_iter", 1000)
        lr = args.get("lr", 1e-3)
        cm = soft_confusion_matrix(val_pred, val_targets)
        priors = matrix_correction_MLE(predictions, init_prior, cm, True, max_iter, lr)
    elif method == "SCM-M":
        max_iter = args.get("max_iter", 1000)
        lr = args.get("lr", 1e-3)
        alpha = args.get("alpha", 3)
        cm = soft_confusion_matrix(val_pred, val_targets)
        priors = matrix_correction_MAP(predictions, init_prior, cm, alpha, True, max_iter, lr)
    else:
        raise ValueError("Unknown prior estimation method.")

    return priors

################
# EM algorithm #
################

def estimate_priors_from_predictions(predictions):
    """ Estimate class priors from predictions.
        
        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: np.array (num_data, num_classes) with predictions
    Returns:
        priors: np.array (num_classes) with estimated class priors
    """
    
    priors = np.mean(predictions, axis=0)
    return priors

def EM_priors_estimation(predictions, trainset_priors, test_init_distribution=None, termination_difference=0.0001, verbose=False):
    """ EM algorithm for test set prior estimation and adjust classifier's predictions 
        to prior shift.

        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        trainset_priors: torch.Tensor (num_classes,) with the train set distribution
        test_init_distribution: torch.Tensor (num_classes,) to initialize test set distribution.
                                If None, use trainset_priors.
        termination_error: float defining the distance of posterior predictions for termination.
    Returns:
        new_predictions: torch.Tensor (num_data, num_classes) with adjusted predictions
        new_testset_priors: torch.Tensor (num_classes,) with the estimated test set distribution
    """
    if test_init_distribution is None:
        test_init_distribution = trainset_priors.copy()
        
    testset_priors = test_init_distribution / np.sum(test_init_distribution)
    step = 0

    while True:
        step += 1
        new_predictions = adjust_predictions(predictions, trainset_priors, testset_priors)
        new_testset_priors = estimate_priors_from_predictions(new_predictions)

        difference = np.sum((new_testset_priors - testset_priors)**2)
        if verbose: print("EM step ", step, "; diff. %.8f" % (difference))
        if difference < termination_difference*termination_difference:
            if verbose: print("Finished. Difference", difference, "< termination value", termination_difference)
            break
        testset_priors = new_testset_priors
        
    return new_predictions, new_testset_priors

###########################
#  CONFUSION MATRIX BASED #
###########################

def CM_estimate(predictions, confusion_matrix, soft=False):
    ''' 
    Test set prior estimation using confusion matrix inversion (Equation 4 in the paper).
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        confusion_matrix: np.array (num_classes, num_classes) 
        soft: bool indicator for soft confusion matrix
    Returns:
        new_testset_priors: np.array (num_classes,) with the estimated test set distribution
    '''
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = np.zeros(num_classes)
        class_pred = np.argmax(predictions, axis=1)
        for i in range(num_classes):
            decision_distribution[i] = np.sum(class_pred == i)
    else:
        decision_distribution = np.sum(predictions, axis=0)

    decision_distribution = decision_distribution / np.sum(decision_distribution)
    
    if np.linalg.matrix_rank(confusion_matrix) == num_classes:
        conf_inv = np.linalg.inv(confusion_matrix)
    else:
        conf_inv = np.linalg.pinv(confusion_matrix)
        
    priors_estimate = (conf_inv @ np.expand_dims(decision_distribution, 1)).squeeze()
    priors_estimate[priors_estimate < 0] = 0
    priors_estimate = priors_estimate / np.sum(priors_estimate)
    
    return priors_estimate

def matrix_correction_MLE(predictions, trainset_priors, confusion_matrix, soft=False, max_iter=1000, lr=1e-3):
    ''' 
        Maximum likelihood estimation of test set priors using confusion matrix, proposed in Section 3.1.
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        trainset_priors: np.array (num_classes,) with the train set distribution
        confusion_matrix: np.array (num_classes, num_classes)
        soft: bool indicator for soft condusion matrix
        max_iter: int max. number of iterations
        lr: float learning rate
    Returns:
        new_predictions: np.array (num_data, num_classes) with adjusted predictions
        new_testset_priors: np.array (num_classes,) with the estimated test set distribution
    '''
    d = 2 # number to divide lr
    num = 50 # lr will be devided each num iterations by d
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = np.zeros(num_classes)
        class_pred = np.argmax(predictions, axis=1)
        for i in range(num_classes):
            decision_distribution[i] = np.sum(class_pred == i)
    else:
        decision_distribution = np.sum(predictions, axis=0)
    
    decision_distribution = decision_distribution / np.sum(decision_distribution)

    new_testset_priors = trainset_priors.copy()
    new_testset_priors = new_testset_priors/np.sum(new_testset_priors)
    for i in range(int(max_iter)):
        o = confusion_matrix @ new_testset_priors
        mask = (o == 0)
        grad = decision_distribution / np.where(mask, np.ones_like(o), o)
        grad[mask] = 0
        
        grad = confusion_matrix*np.expand_dims(grad, 1)
        grad = grad.sum(0)
        
        lr_cur = lr/d**(i//num)
        p_updated = new_testset_priors + lr_cur*grad
        new_testset_priors = simplex_projection(p_updated)
        
    return new_testset_priors

def matrix_correction_MAP(predictions, trainset_priors, confusion_matrix, alpha=3, soft=False, max_iter=1000, lr=0.01):
    ''' 
    Maximum a-posteriori estimation of test set priors with a Dirichlet hyperprior, using confusion matrix, 
    proposed in Section 3.2.
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        trainset_priors: np.array (num_classes,) with the train set distribution
        confusion_matrix: np.array (num_classes, num_classes) to initialize test set distribution.
                                If None, use trainset_priors.
        alpha: float hyperprior of dirichlet distribution
        soft: bool indicator for soft condusion matrix
        max_iter: int max. number of iterations
        lr: float learning rate
    Returns:
        new_predictions: np.array (num_data, num_classes) with adjusted predictions
        new_testset_priors: np.array (num_classes,) with the estimated test set distribution
    '''
    num_classes = predictions.shape[1]
    
    if not soft:
        decision_distribution = np.zeros(num_classes)
        class_pred = np.argmax(predictions, axis=1)
        for i in range(num_classes):
            decision_distribution[i] = np.sum(class_pred == i)
    else:
        decision_distribution = np.sum(predictions, axis=0)
    
    N = np.sum(decision_distribution)
    decision_distribution = decision_distribution / N
    
    new_testset_priors = trainset_priors.copy()
    
    new_testset_priors = new_testset_priors/np.sum(new_testset_priors)
    for i in range(int(max_iter)):
        o = confusion_matrix @ new_testset_priors
        mask = (o == 0)
        grad_l = decision_distribution/np.where(mask, np.ones_like(o), o)
        grad_l[mask] = 0
        
        grad_l = confusion_matrix*np.expand_dims(grad_l, 1)
        grad_l = grad_l.sum(0)
        
        grad_a = ((alpha-1)/new_testset_priors)
        
        grad = grad_a/N + grad_l # divide N because decision_distribution are normalized
        
        p_updated = new_testset_priors + lr*grad
        new_testset_priors = simplex_projection(p_updated)

    return new_testset_priors
