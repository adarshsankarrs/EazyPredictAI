import sklearn
import xgboost
import lightgbm
from tqdm import tqdm
from sklearn.utils import all_estimators
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)
import os
from sklearn.ensemble import VotingRegressor


REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "KNeighborsRegressor",
    "NuSVR",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GaussianProcessRegressor",
    "MLPRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    
    
]



class EazyRegressor:
    """
    For regression
        from eazypredict.EazyRegressor import EazyRegressor

        from sklearn import datasets
        from sklearn.utils import shuffle
        import numpy as np

        boston = datasets.load_boston(as_frame=True)
        X, y = shuffle(boston.data, boston.target, random_state=13)
        X = X.astype(np.float32)

        offset = int(X.shape[0] * 0.9)

        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        reg = EazyRegressor()
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)

        print(models)
        
    OUTPUT
                                    RMSE  R Squared
    LinearRegression           54.964651   0.506806
    LGBMRegressor              55.941752   0.489115
    RandomForestRegressor      56.544922   0.478039
    KNeighborsRegressor        57.351191   0.463048
    XGBRegressor               58.316092   0.444828
    Ridge                      60.245277   0.407488
    NuSVR                      71.055247   0.175780
    DecisionTreeRegressor      85.416106  -0.191051
    MLPRegressor              156.578937  -3.002373
    GaussianProcessRegressor  332.711971 -17.071231
    """
    def __init__(self, regressors="all", save_dir=False, sort_by="rmse"):
        """Initializes the classifier class

        Args:
            regressors (str/list, optional): Takes in a custom list of sklearn regressors. Defaults to "all".
            save_dir (str, optional): Path to output folder to save models in a pickle format. Defaults to False.
            sort_by (str, optional): One of rmse, r_squared. Sorts the output dataframe according to this metric. Defaults to "rmse".
        """
        
        self.regressors = regressors
        self.save_dir = save_dir
        self.sort_by = sort_by

    def __getRegressorList(self):
        """
        Helper function to get all the regressor names and functions from given arguments
        """
        if self.regressors == "all":
            self.regressors = REGRESSORS

        regressor_list = self.regressors
        self.regressors = [e for e in all_estimators() if e[0] in regressor_list]

        if "XGBRegressor" in regressor_list:
            self.regressors.append(("XGBRegressor", xgboost.XGBRegressor))

        if "LGBMRegressor" in regressor_list:
            self.regressors.append(("LGBMRegressor", lightgbm.LGBMRegressor))

    def fit(self, X_train, X_test, y_train, y_test):
        """Function to train the model on training data and evaluate it on the testing data

        Args:
            X_train (pandas.DataFrame/numpy.ndarray)): Training subset of feature data
            X_test (pandas.DataFrame/numpy.ndarray): Testing subset of feature data
            y_train (pandas.DataFrame/numpy.ndarray): Training subset of label data
            y_test (pandas.DataFrame/numpy.ndarray): Testing subset of label data

        Returns:
            dictionary, dictionary, pandas.DataFrame: A dictionary of model_name:function_name, dictionary of model_name:results, sorted dataframe containing the results 
        """
        if isinstance(X_train, np.ndarray) or isinstance(X_test, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.__getRegressorList()

        prediction_list = {}
        model_list = {}
        model_results = {}

        for name, model in tqdm(self.regressors):
            model = model()
            if isinstance(y_train, np.ndarray): 
                model.fit(X_train, y_train.ravel())
            else:
                model.fit(X_train, y_train.values.ravel())
                
            y_pred = model.predict(X_test)

            model_list[name] = model
            prediction_list[name] = y_pred
            if self.save_dir:
                folder_path = os.path.join(self.save_dir, "regressor_model")

                os.makedirs(folder_path, exist_ok=True)
                pickle.dump(
                    model,
                    open(os.path.join(folder_path, f"{name}_model.sav"), "wb"),
                )

            results = []

            try:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            except Exception as exception:
                rmse = None
                print("Ran into an error while calculating rmse score for " + name)
                print(exception)

            try:
                r_squared = r2_score(y_test, y_pred)
            except Exception as exception:
                r_squared = None
                print("Ran into an error while calculating r_squared for " + name)
                print(exception)

            results.append(rmse)
            results.append(r_squared)

            model_results[name] = results

        if self.sort_by == "rmse":
            model_results = dict(sorted(model_results.items(), key=lambda x: x[1]))
        elif self.sort_by == "r_squared":
            model_results = dict(
                sorted(model_results.items(), key=lambda x: x[2], reverse=True)
            )
        else:
            raise Exception("Invalid evaluation metric " + str(self.sort_by))

        result_df = pd.DataFrame(model_results).transpose()
        result_df.columns = ["RMSE", "R Squared"]

        return model_list, prediction_list, result_df
    
    def fitVotingEnsemble(self, model_dict, model_results, num_models=5):
        """Creates an ensemble of models and returns the model and the performance report

        Args:
            model_dict (dictionary): A dictionary containing the different sklearn model names and the function names
            model_results (DataFrame): A DataFrame containing the results of running eazypredict fit methods
            num_models (int, optional): Number of models to be included in the embeddding. Defaults to 5.

        Returns:
            regressor, dataframe: Returns an ensemble sklearn classifier and the results validated on the dataset
        """
        estimators = []
        ensemble_name = ""
        model_results = model_results.iloc[:, 0]
        count = 0
        for model, acc in model_results.items():
            estimators.append((model, model_dict[model]))
            ensemble_name += f"{model} "
            count += 1
            if count == num_models:
                break
        ensemble_reg = VotingRegressor(estimators)
        ensemble_reg.fit(self.X_train, self.y_train.values.ravel())

        y_pred = ensemble_reg.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r_squared = r2_score(self.y_test, y_pred)

        result_dict = {}
        result_dict["Models"] = ensemble_name
        result_dict["RMSE"] = rmse
        result_dict["R Squared"] = r_squared

        result_df = pd.DataFrame(result_dict, index=[0])
        return ensemble_reg, result_df
