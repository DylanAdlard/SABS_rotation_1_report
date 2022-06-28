import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.utils import resample


class SearchModels(object):

    """A class to rapidly iterate over input datarames to find the most
    effective combination of features, as well as to perfrom grid searches
    for hyperparameter tuning for the specified model. The methods are
    modular in design, and new model types can therefore be inserted with ease.

    The class should be instantiated with input parameters, followed by the
    desired model function

    Parameters
    ----------

    1st - Dictionary of dataframes to be searched over {'df_name': df}
        - Each df must only contain numeric feature columns, and a final target column
    2nd - Metric that is being tuned for (accuracy or precision)
    3rd - The column name of the df that contains the target set
    4th - The seed value used for random states
    5th - Whether upsampling should be run (True/False)
    6th - if upsample=True, this is the number of times the smaller class is duplicated


    E.g. Result = SearchModels(ML_df_dict, 'precision', 'target_col', 1234, upsample=True, upsample_coef=3).RandomForest()

    Returns
    ----------
    SearchModels().Model() returns a dictionary with the following key, value pairs:

    {
        "best_df_name": the key of the best performing df from the input,
        "best_df": the best performing df as determined by the grid search with cross validation,
        "best_score": the cross validation score of the best performing model and df from the grid search,
        "average_" + self.metric: average accuracy/precision post cross-validation and grid search,
        "estimator": the best performing model as determined by CV and the grid search,
        "built_model_performance": the performance metrics of the best model,
                                    after retraining on the whole training set,
        "built_model_PR_curve": the precision recall curve of the best model,
                                    after retraining on the whole training set,
        "built_model_ROC_curve": the ROC curve of the best model,
                                    after retraining on the whole training set,
        "built_feature_importance": feature importance plot of the best model
    }

    """

    def __init__(
        self,
        ML_df_dict,
        metric,
        target_column,
        random_seed,
        upsample=False,
        upsample_coef=1,
    ):

        self.model_dict = ML_df_dict
        self.metric = metric
        self.target_column = target_column
        self.random_seed = random_seed
        self.upsample = upsample
        self.upsample_coef = upsample_coef

        self.run_all = False

    def AllModels(self):

        self.run_all = True

        LR = self.LogisticRegression()
        SVC = self.LinearSVC()
        Tree = self.DecisionTree()
        Forest = self.RandomForest()

        score_dict = {
            LR[0]: ["LogisticRegression", LR[1]],
            SVC[0]: ["LinearSVC", SVC[1]],
            Tree[0]: ["DecisionTree", Tree[1]],
            Forest[0]: ["RandomForest", Forest[1]],
        }

        best_score = np.max([i for i in score_dict.keys()])

        best_model = score_dict[best_score]

        self.estimator = best_model[1]

        built_model = self.build_model(best_model[0], self.estimator["df"])

        if best_model[0] in ["LogisticRegression", "LinearSVC"]:
            feature_importance = SearchModels._plot_feature_coefficients(
                built_model["built_model"],
                built_model["data_array"],
                self.estimator["df"],
            )
        elif best_model[0] in ["DecisionTree", "RandomForest"]:
            feature_importance = SearchModels._plot_feature_importance(
                built_model["built_model"],
                built_model["data_array"],
                self.estimator["df"],
            )

        return {
            "best_df_name": self.estimator["df_name"],
            "best_df": self.estimator["df"],
            "best_score": self.estimator["cross_validation_score"],
            "average_" + self.metric: self.estimator["average_" + self.metric],
            "estimator": self.estimator["estimator"],
            "built_model_performance": built_model["performance"],
            "built_model_PR_curve": built_model["precision_recall_curve"],
            "built_model_ROC_curve": built_model["ROC_curve"],
            "built_feature_importance": feature_importance,
        }

    def LogisticRegression(self):
        """Constructs a Logistic Regression pipeline with a grid search
        containing preprocessing strategeies and different values for C.

        Post construction, LinearModelSearch() is called to run the pipeline,
        and the resulting best estimator is used to retrain the model and test.

        Returns
        ----------
        Dictionary as described in the primary class docstring

        """

        scores = []
        estimators = {}

        for key, df in self.model_dict.items():
            # assumes the last column is the target column
            data_array = df[df.columns[:-1]].to_numpy()

            pipe = Pipeline(
                [
                    ("preprocessing", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=10000)),
                ]
            )
            param_grid = {
                "preprocessing": [
                    StandardScaler(),
                    MinMaxScaler(),
                    RobustScaler(),
                    None,
                ],
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            }

            score_obj = self.LinearModelSearch(
                pipe, param_grid, data_array, scores, estimators, key, df
            )
            scores = score_obj["scores"]
            estimators = score_obj["estimators"]

        best_av_score = np.max(scores)
        self.estimator = estimators[best_av_score]

        if self.run_all == True:
            return [best_av_score, estimators[best_av_score]]

        built_model = self.build_model("LogisticRegression", self.estimator["df"])

        feature_importance = SearchModels._plot_feature_coefficients(
            built_model["built_model"],
            built_model["data_array"],
            self.estimator["df"],
        )

        return {
            "best_df_name": self.estimator["df_name"],
            "best_df": self.estimator["df"],
            "best_score": self.estimator["cross_validation_score"],
            "average_" + self.metric: self.estimator["average_" + self.metric],
            "estimator": self.estimator["estimator"],
            "built_model_performance": built_model["performance"],
            "built_model_PR_curve": built_model["precision_recall_curve"],
            "built_model_ROC_curve": built_model["ROC_curve"],
            "built_feature_importance": feature_importance,
        }

    def LinearSVC(self):
        """Constructs a LinearSVC pipeline with a grid search
        containing preprocessing strategeies and different values for C.

        Post construction, LinearModelSearch() is called to run the pipeline,
        and the resulting best estimator is used to retrain the model and test.

        Returns
        ----------
        Dictionary as described in the primary class docstring

        """

        scores = []
        estimators = {}

        for key, df in self.model_dict.items():
            # assuems the last column is the target column
            data_array = df[df.columns[:-1]].to_numpy()

            pipe = Pipeline(
                [
                    ("preprocessing", StandardScaler()),
                    ("classifier", sklearn.svm.LinearSVC(max_iter=100000, dual=False)),
                ]
            )

            param_grid = {
                "preprocessing": [
                    StandardScaler(),
                    MinMaxScaler(),
                    RobustScaler(),
                    None,
                ],
                "classifier__C": [0.01, 0.1, 1, 10, 100],
            }

            score_obj = self.LinearModelSearch(
                pipe, param_grid, data_array, scores, estimators, key, df
            )
            scores = score_obj["scores"]
            estimators = score_obj["estimators"]

        best_av_score = np.max(scores)
        self.estimator = estimators[best_av_score]

        if self.run_all == True:
            return [best_av_score, estimators[best_av_score]]

        built_model = self.build_model("LinearSVC", self.estimator["df"])

        feature_importance = SearchModels._plot_feature_coefficients(
            built_model["built_model"],
            built_model["data_array"],
            self.estimator["df"],
        )

        return {
            "best_df_name": self.estimator["df_name"],
            "best_df": self.estimator["df"],
            "best_score": self.estimator["cross_validation_score"],
            "average_" + self.metric: self.estimator["average_" + self.metric],
            "estimator": self.estimator["estimator"],
            "built_model_performance": built_model["performance"],
            "built_model_PR_curve": built_model["precision_recall_curve"],
            "built_model_ROC_curve": built_model["ROC_curve"],
            "built_feature_importance": feature_importance,
        }

    def LinearModelSearch(
        self, pipe, param_grid, data_array, scores, estimators, key, df
    ):
        """Runs the grid search when called by the linear model methods.
        Uses either average accuracy or average precision depending on
        instantiation parameters.

        Returns
        ----------
        Dictionary of the form:
        {
            'scores': [array of internal custon model scores],
            'estimator': {
                'df_name': name of dataframe,
                'df': the df,
                'estimator': the estimator as defined by the grid search,
                'cross_validation_score: the best cross validation score from the grid search
                'average accuracy/precision': the average cross-validated accuracy/precision
                                                from the grid search
            }
        }
        """

        target = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            data_array, target, random_state=self.random_seed
        )

        if self.upsample == True:
            upsampled_arrays = SearchModels._upsample(
                X_train, y_train, self.upsample_coef
            )
            X_train, y_train = (
                upsampled_arrays["data_array"],
                upsampled_arrays["target"],
            )

        if self.metric == "accuracy":
            StratifiedKFold = sklearn.model_selection.StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(pipe, param_grid, cv=StratifiedKFold)
            grid_strat.fit(X_train, y_train)

            best_score = grid_strat.best_score_
            accuracy = accuracy_score(y_test, grid_strat.predict(X_test))
            mean_score = np.mean([best_score, accuracy])
            scores.append(mean_score)
            estimators[mean_score] = {
                "df_name": key,
                "df": df,
                "estimator": grid_strat.best_estimator_,
                "cross_validation_score": best_score,
                "average_accuracy": accuracy,
            }

        elif self.metric == "precision":
            StratifiedKFold = sklearn.model_selection.StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(
                pipe, param_grid, scoring="average_precision", cv=StratifiedKFold
            )
            grid_strat.fit(X_train, y_train)

            best_score = grid_strat.best_score_
            precision = average_precision_score(
                y_test, grid_strat.decision_function(X_test)
            )
            mean_score = np.mean([best_score, precision])
            scores.append(mean_score)
            estimators[mean_score] = {
                "df_name": key,
                "df": df,
                "estimator": grid_strat.best_estimator_,
                "cross_validation_score": best_score,
                "average_precision": precision,
            }

        return {"scores": scores, "estimators": estimators}

    def DecisionTree(self):
        """Constructs a single tree with a grid search containing different
        values for max_depth, min_samples_split, min_samples_leaf,
        and max_features

        Post construction, TreeModelSearch() is called to run the classifier,
        and the resulting best estimator is used to retrain the model and test.

        Returns
        ----------
        Dictionary as described in the primary class docstring

        """

        scores = []
        estimators = {}

        for key, df in self.model_dict.items():
            # assuems the last column is the target column
            data_array = df[df.columns[:-1]].to_numpy()

            tree = sklearn.tree.DecisionTreeClassifier(random_state=self.random_seed)

            param_grid = {
                "max_depth": [2, 4, 6, 8, 10, 12, 14, None],
                "min_samples_split": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                "max_features": ["sqrt", "log2", None],
            }

            score_obj = self.TreeModelSearch(
                tree, param_grid, data_array, scores, estimators, key, df
            )

            scores = score_obj["scores"]
            estimators = score_obj["estimators"]

        best_av_score = np.max(scores)
        self.estimator = estimators[best_av_score]

        if self.run_all == True:
            return [best_av_score, estimators[best_av_score]]

        built_model = self.build_model("DecisionTree", self.estimator["df"])

        feature_importance = SearchModels._plot_feature_importance(
            built_model["built_model"],
            built_model["data_array"],
            self.estimator["df"],
        )

        return {
            "best_df_name": self.estimator["df_name"],
            "best_df": self.estimator["df"],
            "best_score": self.estimator["cross_validation_score"],
            "average_" + self.metric: self.estimator["average_" + self.metric],
            "estimator": self.estimator["estimator"],
            "built_model_performance": built_model["performance"],
            "built_model_PR_curve": built_model["precision_recall_curve"],
            "built_model_ROC_curve": built_model["ROC_curve"],
            "built_feature_importance": feature_importance,
        }

    def RandomForest(self):
        """Constructs a random forest with a grid search containing different
        values for n_estimators and max_depth.

        Post construction, TreeModelSearch() is called to run the classifier,
        and the resulting best estimator is used to retrain the model and test.

        Returns
        ----------
        Dictionary as described in the primary class docstring

        """

        scores = []
        estimators = {}

        for key, df in self.model_dict.items():
            # assumes the last column is the target column
            data_array = df[df.columns[:-1]].to_numpy()

            forest = RandomForestClassifier(n_jobs=-1, random_state=self.random_seed)

            param_grid = {
                "n_estimators": [10, 20, 30, 40, 50, 60, 100],
                "max_depth": [2, 4, 6, 8, 10, 12, 14, 16],
            }

            score_obj = self.TreeModelSearch(
                forest, param_grid, data_array, scores, estimators, key, df
            )

            scores = score_obj["scores"]
            estimators = score_obj["estimators"]

        best_av_score = np.max(scores)
        self.estimator = estimators[best_av_score]

        if self.run_all == True:
            return [best_av_score, estimators[best_av_score]]

        built_model = self.build_model("RandomForest", self.estimator["df"])

        feature_importance = SearchModels._plot_feature_importance(
            built_model["built_model"],
            built_model["data_array"],
            self.estimator["df"],
        )

        return {
            "best_df_name": self.estimator["df_name"],
            "best_df": self.estimator["df"],
            "best_score": self.estimator["cross_validation_score"],
            "average_" + self.metric: self.estimator["average_" + self.metric],
            "estimator": self.estimator["estimator"],
            "built_model_performance": built_model["performance"],
            "built_model_PR_curve": built_model["precision_recall_curve"],
            "built_model_ROC_curve": built_model["ROC_curve"],
            "built_feature_importance": feature_importance,
        }

    def TreeModelSearch(
        self, pipe, param_grid, data_array, scores, estimators, key, df
    ):
        """Runs the grid search when called by the tree model methods.
        Uses either average accuracy or average precision depending on
        instantiation parameters.

        Returns
        ----------
        Dictionary of the form:
        {
            'scores': [array of internal custon model scores],
            'estimator': {
                'df_name': name of dataframe,
                'df': the df,
                'estimator': the estimator as defined by the grid search,
                'cross_validation_score: the best cross validation score from the grid search
                'average accuracy/precision': the average cross-validated accuracy/precision
                                                from the grid search
            }
        }
        """

        target = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            data_array, target, random_state=self.random_seed
        )

        if self.upsample == True:
            upsampled_arrays = SearchModels._upsample(
                X_train, y_train, self.upsample_coef
            )
            X_train, y_train = (
                upsampled_arrays["data_array"],
                upsampled_arrays["target"],
            )

        if self.metric == "accuracy":
            StratifiedKFold = sklearn.model_selection.StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(pipe, param_grid, cv=StratifiedKFold)
            grid_strat.fit(X_train, y_train)

            best_score = grid_strat.best_score_
            accuracy = accuracy_score(y_test, grid_strat.predict(X_test))
            mean_score = np.mean([best_score, accuracy])
            scores.append(mean_score)
            estimators[mean_score] = {
                "df_name": key,
                "df": df,
                "estimator": grid_strat.best_estimator_,
                "cross_validation_score": best_score,
                "average_accuracy": accuracy,
            }

        elif self.metric == "precision":
            StratifiedKFold = sklearn.model_selection.StratifiedKFold(n_splits=5)
            grid_strat = GridSearchCV(
                pipe, param_grid, cv=StratifiedKFold, scoring="average_precision"
            )
            grid_strat.fit(X_train, y_train)

            best_score = grid_strat.best_score_
            precision = average_precision_score(
                y_test, grid_strat.predict_proba(X_test)[:, 1]
            )
            mean_score = np.mean([best_score, precision])
            scores.append(mean_score)
            estimators[mean_score] = {
                "df_name": key,
                "df": df,
                "estimator": grid_strat.best_estimator_,
                "cross_validation_score": best_score,
                "average_precision": precision,
            }

        return {"scores": scores, "estimators": estimators}

    def build_model(self, model, df):
        """Builds and tests the specified model using the entire original
        training and test sets.

        Returns
        ----------
        Returns dictionary of the form:
        {
            "built_model": the pipeline or model object,
            "data_array": the training data,
            "performance": the calculated perforamance metrics (including matrix),
            "precision_recall_curve": the model's precision_recall_curve,
            "ROC_curve": the model's ROC_curve,
        }

        """

        data_array = df[df.columns[:-1]].to_numpy()

        target = df[self.target_column]

        if (model == "LogisticRegression") or (model == "LinearSVC"):
            pipe = Pipeline(
                [
                    ("preprocessing", self.estimator["estimator"]["preprocessing"]),
                    ("classifier", self.estimator["estimator"]["classifier"]),
                ]
            )

        elif (model == "DecisionTree") or (model == "RandomForest"):
            pipe = self.estimator["estimator"]

        X_train, X_test, y_train, y_test = train_test_split(
            data_array, target, random_state=self.random_seed,
        )

        if self.upsample == True:
            upsampled_arrays = SearchModels._upsample(
                X_train, y_train, self.upsample_coef
            )
            X_train, y_train = (
                upsampled_arrays["data_array"],
                upsampled_arrays["target"],
            )

        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test).astype(int)

        performance = SearchModels._calc_performance(y_test, prediction)

        precision_recall_curve = SearchModels._plot_precision_recall(
            model, pipe, X_test, y_test
        )

        ROC_curve = SearchModels._plot_ROC_curve(model, pipe, X_test, y_test)

        return {
            "built_model": pipe,
            "data_array": data_array,
            "performance": performance,
            "precision_recall_curve": precision_recall_curve,
            "ROC_curve": ROC_curve,
        }

    @staticmethod
    def _calc_performance(y_test, prediction):
        """Calculates the performance metrics from y_test and the predicted results

        Returns
        ----------
        Dict of the form:
        {
            'Precision': True positives/(true positives + false positives)
            'Sensitivity': (Recall): (True positives/true positives + false negatives)
            'Specificity': True negative rate: True negatives/(true negatives + false positives)
            'FPR': False positives/(false positives + true negatives): 1-TNR
            'VME': False positives/ test set positives
            'ME': False negatives/ test set negatives
            'confusion_matrix': confusion matrix
        }

        """

        # generate confusion matrix
        confusion = confusion_matrix(y_test, prediction)

        # calculate precision, sensitivity, specifcity, FPR, errors
        Precision = (confusion[1][1]) / (confusion[1][1] + confusion[0][1])
        Sensitivity = (confusion[1][1]) / (confusion[1][1] + confusion[1][0])
        Specificity = (confusion[0][0]) / (confusion[0][0] + confusion[0][1])
        FPR = 1 - Specificity
        very_major_error = (confusion[0][1] / y_test[y_test == 0].count()) * 100
        major_error = (confusion[1][0] / y_test[y_test == 1].count()) * 100

        performance = {
            "Precision": Precision,
            "Sensitivity": Sensitivity,
            "Specificity": Specificity,
            "FPR": FPR,
            "VME": very_major_error,
            "ME": major_error,
            "confusion_matrix": confusion,
        }

        return performance

    @staticmethod
    def _plot_precision_recall(model, pipe, X_test, y_test):
        """Plots the precision recall curve

        Returns
        ----------
        Preicision Recall Curve
        """

        if (model == "LogisticRegression") or (model == "LinearSVC"):
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
                y_test, pipe.decision_function(X_test)
            )
            close_threshold = np.argmin(np.abs(thresholds))
        elif (model == "DecisionTree") or (model == "RandomForest"):
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
                y_test, pipe.predict_proba(X_test)[:, 1]
            )
            close_threshold = np.argmin(np.abs(thresholds - 0.5))

        fig = plt.figure()
        plt.plot(precision, recall, figure=fig)
        plt.plot(
            precision[close_threshold],
            recall[close_threshold],
            "^",
            c="k",
            markersize=10,
            label="threshold",
            fillstyle="none",
            mew=2,
            figure=fig,
        )
        plt.xlabel("precision", figure=fig)
        plt.ylabel("recall", figure=fig)
        plt.legend(loc="best")
        plt.title("Precision-recall curve", figure=fig)
        precision_recall_curve = fig
        plt.close()

        return precision_recall_curve

    @staticmethod
    def _plot_ROC_curve(model, pipe, X_test, y_test):
        """Plots the ROC curve

        Returns
        ----------
        ROC curve
        """

        if (model == "LogisticRegression") or (model == "LinearSVC"):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_test, pipe.decision_function(X_test)
            )
            close_threshold = np.argmin(np.abs(thresholds))
        elif (model == "DecisionTree") or (model == "RandomForest"):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_test, pipe.predict_proba(X_test)[:, 1]
            )
            close_threshold = np.argmin(np.abs(thresholds - 0.5))

        fig = plt.figure()
        plt.plot(fpr, tpr, figure=fig)
        plt.plot(
            fpr[close_threshold],
            tpr[close_threshold],
            "^",
            markersize=10,
            label="threshold",
            fillstyle="none",
            c="k",
            mew=2,
            figure=fig,
        )
        plt.xlabel("FPR", figure=fig)
        plt.ylabel("TPR", figure=fig)
        plt.legend(loc=4)
        plt.title("ROC curve", figure=fig)
        ROC_curve = fig
        plt.close()

        return ROC_curve

    @staticmethod
    def _plot_feature_importance(model, data_array, ML_df):
        """Plots the feature importance plot for tree models

        Returns
        ----------
        Feature importance plot
        """

        n_features = data_array.shape[1]
        fig = plt.figure()
        plt.barh(
            np.arange(n_features),
            model.feature_importances_,
            align="center",
            figure=fig,
        )
        plt.yticks(np.arange(n_features), ML_df.columns[:-1], figure=fig)
        plt.xlabel("feature importance", figure=fig)
        plt.ylabel("feature", figure=fig)
        plt.ylim(-1, n_features)
        plt.title("feature importance plot", figure=fig)
        feature_importance_plot = fig
        plt.close()

        return feature_importance_plot

    @staticmethod
    def _plot_feature_coefficients(model, data_array, ML_df):
        """Plots the feature coefficint plot for linear models.
        This aims to provide a similar idea of importance as feature importance plots

        Returns
        ----------
        Feature coefficient plot
        """

        n_features = data_array.shape[1]
        coef_df = pd.DataFrame(
            {
                "coefficients": (model.named_steps["classifier"].coef_)[0],
                "n_features": np.arange(n_features),
            }
        )

        coef_df = coef_df.melt("n_features", var_name="Params", value_name="importance")

        fig = plt.figure()
        plt.barh(
            coef_df["n_features"], coef_df["importance"], align="center", figure=fig
        )
        plt.yticks(coef_df["n_features"], ML_df.columns[:-1], figure=fig)
        plt.xlabel("coefficients", figure=fig)
        plt.ylabel("feature", figure=fig)
        plt.title("feature coefficient plot", figure=fig)
        coef_plot = fig
        plt.close()

        return coef_plot

    @staticmethod
    def _upsample(data_array, target, upsample_coef):
        """Upsamples the minority class by a defined number of multiples
        _upsample should only be called after the data is split, and should
        only be called on the training sets to avoid data leakage (the model
        must be tested on non-upsampled data)

        Returns
        ----------
        Dictionary of the form:
        {
            'data_array': the upsampled feature set, 89
            'target': the upsampled target set
        }
        """

        minority_class = 1 - target.mode()[0]
        data_array_up, target_up = resample(
            data_array[target == minority_class],
            target[target == minority_class],
            stratify=data_array[target == minority_class],
            n_samples=(upsample_coef * data_array[target == minority_class].shape[0]),
            replace=True,
            random_state=0,
        )

        data_array = np.vstack((data_array, data_array_up))
        target = np.hstack((target, target_up))

        return {"data_array": data_array, "target": target}
