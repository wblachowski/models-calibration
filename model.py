import pandas as pd
import h2o
from sklearn.base import ClassifierMixin
from h2o.estimators import H2OEstimator
from calibration import IsotonicCalibrator, PlattCalibrator
from sklearn.calibration import calibration_curve


class CalibratableModelFactory:
    def get_model(self, base_model):
        if self._is_h2o(base_model):
            return H2OModel(base_model)
        elif self._is_scikit_probability(base_model):
            return ScikitProbabilityModel(base_model)
        elif self._is_scikit_distance(base_model):
            return ScikitDistanceModel(base_model)

    def _is_h2o(self, base_model):
        return isinstance(base_model, H2OEstimator)

    def _is_scikit_probability(self, base_model):
        return isinstance(base_model, ClassifierMixin) and hasattr(
            base_model, "predict_proba"
        )

    def _is_scikit_distance(self, base_model):
        return isinstance(base_model, ClassifierMixin) and hasattr(
            base_model, "decision_function"
        )


class CalibratableModelMixin:
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__
        self.platt_calibrator = None
        self.isotonic_calibrator = None
        self.calibrators = {
            "platt": None,
            "isotonic": None,
        }

    def calibrate(self, X, y):
        predictions = self.predict(X)
        prob_true, prob_pred = calibration_curve(y, predictions, n_bins=10)
        self.calibrators["platt"] = PlattCalibrator(prob_pred, prob_true)
        self.calibrators["isotonic"] = IsotonicCalibrator(prob_pred, prob_true)

    def calibrate_probabilities(self, probabilities, method="isotonic"):
        if method not in self.calibrators:
            raise ValueError("Method has to be either 'platt' or 'isotonic'")
        if self.calibrators[method] is None:
            raise ValueError("Fit the calibrators first")
        return self.calibrators[method].calibrate(probabilities)

    def predict_calibrated(self, X, method="isotonic"):
        return self.calibrate_probabilities(self.predict(X), method)


class H2OModel(CalibratableModelMixin):
    def train(self, X, y):
        self.features = list(range(len(X[0])))
        self.target = "target"
        train_frame = self._to_h2o_frame(X, y)
        self.model.train(x=self.features, y=self.target, training_frame=train_frame)

    def score(self, X, y):
        test_frame = self._to_h2o_frame(X, y)
        return self.model.model_performance(test_frame).accuracy()[0][1]

    def predict(self, X):
        predict_frame = self._to_h2o_frame(X)
        return self.model.predict(predict_frame).as_data_frame()["p1"].tolist()

    def _to_h2o_frame(self, X, y=None):
        df = pd.DataFrame(data=X, columns=self.features)
        if y is not None:
            df[self.target] = y
        h2o_frame = h2o.H2OFrame(df)
        if y is not None:
            h2o_frame[self.target] = h2o_frame[self.target].asfactor()
        return h2o_frame


class ScikitModelMixin:
    def train(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)


class ScikitProbabilityModel(ScikitModelMixin, CalibratableModelMixin):
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class ScikitDistanceModel(ScikitModelMixin, CalibratableModelMixin):
    def predict(self, X):
        probs = self.model.decision_function(X)
        return (probs - probs.min()) / (probs.max() - probs.min())