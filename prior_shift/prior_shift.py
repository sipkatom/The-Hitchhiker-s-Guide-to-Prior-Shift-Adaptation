import numpy as np
from .prior_estimation import estimate_priors
from .tools import adjust_predictions

def adapt_to_prior_shift(test_pred, val_pred, val_targets, train_pred, train_targets=None, method="SCM-L", args=dict()):
    """ Adapts given prediction on test set to label shift.

    Args:
        test_pred: np.array (n_test, num_classes) with predictions on test set, which should be adapted
        val_pred: np.array (n_val, num_classes) with predictions on validation set
        val_targets: np.array (n_val,) with ground truth labels coresponding 
                    to the predictions on validation dataset
        train_pred: np.array (n_train, num_classes) with predictions on trainning set
        train_targets: np.array (n_train,) with ground truth labels coresponding
                    to the predictions on trainning dataset
        method: str, the algorithm used for priors estimation
        args: Dict[str, Any], arguments for the algorithm

    Returns:
        np.array (n_test, num_classes) with adjusted predictions
    """
    if method != "EM" and (val_pred is None or val_targets is None):
        raise ValueError(f"For method {method} validation set is required!")
    
    tp = args.pop("trainset_prior", "classif")
    if tp == "classif":
        if train_pred is None:
            raise ValueError(f"When `trainset_prior==classif`, `train_pred` cannot be None.")
        train_prior = get_classifier_priors(train_pred)
    elif tp == "count":
        if train_targets is None:
            raise ValueError(f"When `trainset_prior==count`, `train_targets` cannot be None.")
        train_prior = count_classes(train_targets, test_pred.shape[1])
    else:
        raise ValueError("Unknown method for computing trainnig priors.")

    test_prior = estimate_priors(test_pred, val_pred, val_targets, train_prior, method, args)
    adjusted_preds = adjust_predictions(test_pred, train_prior, test_prior)

    return adjusted_preds


def get_classifier_priors(training_predictions):
    """ Estimate training priors by averaging training predictions

    Args:
        training_predictions: np.array (num_data, num_classes)
    Returns:
        np.array (num_classes, ) of estimated training priors
    """
    return np.mean(training_predictions, axis=0)


def count_classes(targets, num_classes):
    """ Count number of samples per class in set of labels.

    Args:
        targets: np.array (num_data,) with ground truth labels in the dataset
        num_classes: int representing number of classes in the dataset
    Returns:
        counts: np.array (num_classes, ) with number of samples per class
    """
    counts = np.zeros(num_classes)
    for i in range(num_classes):
        counts[i] = (targets == i).sum().astype(float)
    return counts