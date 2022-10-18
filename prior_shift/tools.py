import numpy as np
from sklearn.metrics import confusion_matrix

def adjust_predictions(predictions, trainset_priors, test_set_distribution=None):
    """ Adjust classifier's predictions to prior shift,
        knowing the training set distribution and a different test set distribution.
        I.e. predictions are multiplied by the ratio of class priors.
        
        Code modified from:
        https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        trainset_priors: np.array (num_classes,)
        test_set_distribution: np.array (num_classes,); if None - use uniform distribution
    Returns:
        adjusted_predictions: np.array (num_data, num_classes) with adjusted predictions
    """
    if test_set_distribution is None:
        test_set_distribution = np.ones(trainset_priors.shape)
    adjusted_predictions = predictions * test_set_distribution / trainset_priors
    adjusted_predictions = adjusted_predictions / np.expand_dims(np.sum(adjusted_predictions, axis=1), 1) # normalize to sum to 1
    return adjusted_predictions

def simplex_projection(y):
    """
    Projection onto the probability simplex, based on https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf

    Code modified from:
    https://github.com/sulc/priors-example/blob/master/cifar-priors-example.ipynb
    """
    u = -np.sort(-y) # sort y in descending order
    j = np.arange(1, len(y)+1)
    phi_obj = u + 1/j * (1-np.cumsum(u))
    positive = np.argwhere(phi_obj > 0)
    if positive.size == 0: raise ValueError("Numerical issues - extremely large values after update.. DECREASE LEARNING RATE")
    phi = positive.max() + 1
    lam = 1/phi * (1-np.sum(u[:phi]))
    x = np.maximum(y+lam,0)

    return x

def hard_confusion_matrix(predictions, targets):
    ''' Compute conditional confusion matrix from classifier's predictions
        and ground truth labels.
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        targets: np.array (num_data,) with ground truth labels coresponding to the predictions
    Returns:
        mat: np.array (num_classes, num_classes) with hard conditional confusion matrix
    '''
    num_classes = predictions.shape[1]
    
    y = np.argmax(predictions, axis=1)
    
    mat = confusion_matrix(targets, y, normalize='true', labels=np.arange(num_classes)).T
    
    return mat.astype(float)

def soft_confusion_matrix(predictions, targets):
    ''' Compute soft confusion matrix from classifier's predictions
        and ground truth labels.
    Args:
        predictions: np.array (num_data, num_classes) with predictions
        targets: np.array (num_data,) with ground truth labels coresponding to the predictions
    Returns:
        mat: np.array (num_classes, num_classes) with soft confusion matrix
    '''
    num_classes = predictions.shape[1]
    
    mat = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        mask = targets == i
        m = np.mean(predictions[mask,:], axis=0)
        mat[:,i] = m
    
    return mat