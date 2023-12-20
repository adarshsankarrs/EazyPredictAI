# EazyPredict 



EazyPredict serves as a quick way to try out multiple prediction algorithms on data while writing as few lines as possible. It also provides the possibility to create an ensemble of the top models

The 'EazyPredict' module was heavily inspired by [LazyPredict](https://github.com/shankarpandala/lazypredict). This module varies in terms of its functionality and intended use, as outlined in the following ways:

- The 'EazyPredict' module utilizes a limited number of prediction algorithms (around 9) in order to minimize memory usage and prevent potential issues on platforms such as Kaggle.

- Users have the option to input a custom list of prediction algorithms (as demonstrated in the example provided) in order to perform personalized comparisons.

- The models can be saved to an output folder at the user's discretion and are returned as a dictionary, allowing for easy addition of custom hyperparameters.

- The top 5 models are selected to create an ensemble using a voting classifier.

# Installation

```python
pip install eazypredict
```

# Usage

### For classification

```python
from eazypredict.EazyClassifier import EazyClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=123)

clf = EazyClassifier()

model_list, prediction_list, model_results = clf.fit(X_train, X_test, y_train, y_test)

print(model_results)
```
#### OUTPUT
```
                        Accuracy  f1 score  ROC AUC score
XGBClassifier           0.978947  0.978990       0.979302
LGBMClassifier          0.971930  0.971930       0.969594
RandomForestClassifier  0.968421  0.968516       0.968953
RidgeClassifier         0.964912  0.964670       0.955671
MLPClassifier           0.961404  0.961185       0.952923
GaussianNB              0.957895  0.957707       0.950176
DecisionTreeClassifier  0.936842  0.937093       0.935800
KNeighborsClassifier    0.936842  0.936407       0.925264
SVC                     0.919298  0.917726       0.896778
SGDClassifier           0.831579  0.834856       0.861811
```

### For regression

```python
from eazypredict.EazyRegressor import EazyRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

california_housing = fetch_california_housing(as_frame=True)
X, y = california_housing.data, california_housing.target

scaler = RobustScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size= 0.2)

reg = EazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
```
#### OUTPUT
```
                           RMSE       R Squared
LGBMRegressor              0.468025	  0.838435
XGBRegressor               0.470601	  0.836651
RandomForestRegressor      0.490834	  0.822303
MLPRegressor               0.590093	  0.743167
KNeighborsRegressor        0.646152	  0.692051
NuSVR                      0.656935	  0.681687
DecisionTreeRegressor      0.704143	  0.634294
LinearRegression           0.736127	  0.600318
Ridge                      0.736145	  0.600297
GaussianProcessRegressor   10.893585	-86.528957
```

### Creating an ensemble model

```python
reg = EazyRegressor()

model_dict, prediction_list, model_results = reg.fit(X_train, y_train, X_test, y_test)

ensemble_reg, ensemble_results = reg.fitVotingEnsemble(model_dict, model_results)
print(ensemble_results)
```
#### OUTPUT
```
                                                            RMSE        R Squared
LGBMRegressor XGBRegressor RandomForestRegress...           0.48638   0.825514
```

### Custom Estimators

Get more estimators from [sklearn](https://scikit-learn.org/1.0/modules/generated/sklearn.utils.all_estimators.html).

```python
custom_list = [
  "LinearSVC",
  "NearestCentroid",
  "ExtraTreeClassifier",
  "LinearDiscriminantAnalysis",
  "AdaBoostClassifier"
]

clf = EazyClassifier(classififers=custom_list)
model_list, prediction_list, model_results = clf.fit(X_train, y_train, X_test, y_test)

print(model_results)
```
#### OUTPUT
```
                            Accuracy  f1 score  ROC AUC score
AdaBoostClassifier          0.961404  0.961444       0.959245
LinearDiscriminantAnalysis  0.961404  0.961089       0.950816
ExtraTreeClassifier         0.908772  0.909134       0.905393
NearestCentroid             0.898246  0.894875       0.865545
LinearSVC                   0.838596  0.841756       0.867305
```

# Future Plans subject to improvisaton

- Hyperparameter Tuning Feature
- Parallel computation of training