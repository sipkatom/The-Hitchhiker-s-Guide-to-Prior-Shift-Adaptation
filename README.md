# The Hitchhiker's Guide to Prior-Shift Adaptation

## Suplementary material

This sample code for prior shift adaptation in `supplementary_material` directory is a supplementary material for the paper: 

Tomas Sipka, Milan Sulc and Jiri Matas. [The Hitchhiker's Guide to Prior-Shift Adaptation](https://openaccess.thecvf.com/content/WACV2022/html/Sipka_The_Hitchhikers_Guide_to_Prior-Shift_Adaptation_WACV_2022_paper.html). WACV, 2022.

See [supplementary_material/example.ipynb](supplementary_material/example.ipynb) for the usage of the methods and their comparison.

## Prior-shift package

The code in `prior_shift` directory can be used as python package, providing a convenient `adapt_to_prior_shift` function for adapting your predictions to prior shift. See the sample usage in [sample.ipynb](sample.ipynb).

### Installation
Requirements:
```
numpy
sklearn
```

Copy `prior_shift` directory into your project and import it.

### Usage
You can adapt your predictions with default settings:
```
import prior_shift as ps
test_pred_adapted = ps.adapt_to_prior_shift(test_pred, val_pred, val_targets, train_pred)
```
or you can call the function with diffenrent arguments to change the algorithm or its hyperparameters:
```
adapt_to_prior_shift(test_pred, val_pred, val_targets, train_pred, train_targets, method, args)

Arguments:
  test_pred: np.array (n_test, num_classes), predictions on test set, which should be adapted
  val_pred: np.array (n_val, num_classes), predictions on validation set
  val_targets: np.array (n_val,), ground truth labels coresponding to the predictions on 
      validation dataset
  train_pred: np.array (n_train, num_classes), predictions on trainning set
  train_targets: np.array (n_train,), ground truth labels coresponding to the predictions 
      on trainning dataset
  method: str, the algorithm used for priors estimation, allowed values are "EM", "CM", "SCM", 
      "CM-L", "SCM-L", "CM-M", "SCM-M", as default SCM-L is used
  args: Dict[str, Any], parameters used for the algorithm (see below)
```

Method's parameters:

**EM** 
- "termination_difference" (float), default=0.001
- "trainset_prior" ({"classif", "count"}), default="classif"

**CM**
- "trainset_prior" ({"classif", "count"}), default="classif"

**SCM**
- "trainset_prior" ({"classif", "count"}), default="classif"

**CM-L** 
- "lr" (float), default=1e-4<br/>
- "max_iter" (integer), default=1000
- "trainset_prior" ({"classif", "count"}), default="classif"
       
**SCM-L**
- "lr" (float), default=1e-3<br/>
- "max_iter" (integer), default=1000
- "alpha" (float), default=3
- "trainset_prior" ({"classif", "count"}), default="classif"
        
**CM-M**
- "lr" (float), default=1e-3<br/>
- "max_iter" (integer), default=1000
- "trainset_prior" ({"classif", "count"}), default="classif"
        
**SCM-M**
- "lr" (float), default=1e-3<br/>
- "max_iter" (integer), default=1000
- "alpha" (float), default=3
- "trainset_prior" ({"classif", "count"}), default="classif"

**Notes**
- EM algorithm does not require validation set (`val_pred` and `val_targets` can be set to `None`).
- Validation set can have arbitrary distribution.
- Using training set instead of validation set can lead to decreased performance due to overfitting.
- If `trainset_prior="classif"` then `train_targets` can be set to `None`
- If `trainset_prior="count"` then `train_pred` can be set to `None`

## Citation

If you use the code please cite the paper:

```
@InProceedings{Sipka_2022_WACV,
    author    = {\v{S}ipka, Tom\'a\v{s} and \v{S}ulc, Milan and Matas, Ji\v{r}{\'\i}}},
    title     = {The Hitchhiker's Guide to Prior-Shift Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1516-1524}
}
```
