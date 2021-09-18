import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OEstimator as H2OClassifier
from sklearn.base import ClassifierMixin as ScikitClassifier
from tensorflow.keras import Model as KerasBaseModel
from sklearn.calibration import calibration_curve

from calibration import IsotonicCalibrator, SigmoidCalibrator


class CalibratableModelFactory:
    def get_model(self, base_model):
        if isinstance(base_model, H2OClassifier):
            return H2OModel(base_model)
        elif isinstance(base_model, ScikitClassifier):
            return ScikitModel(base_model)
        elif isinstance(base_model, KerasBaseModel):
            return KerasModel(base_model)
        raise ValueError("Unsupported model passed as an argument")


class CalibratableModelMixin:
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__
        self.sigmoid_calibrator = None
        self.isotonic_calibrator = None
        self.calibrators = {
            "sigmoid": None,
            "isotonic": None,
        }

    def calibrate(self, X, y):
        predictions = self.predict(X)
        prob_true, prob_pred = calibration_curve(y, predictions, n_bins=10)
        self.calibrators["sigmoid"] = SigmoidCalibrator(prob_pred, prob_true)
        self.calibrators["isotonic"] = IsotonicCalibrator(prob_pred, prob_true)

    def calibrate_probabilities(self, probabilities, method="isotonic"):
        if method not in self.calibrators:
            raise ValueError("Method has to be either 'sigmoid' or 'isotonic'")
        if self.calibrators[method] is None:
            raise ValueError("Fit the calibrators first")
        return self.calibrators[method].calibrate(probabilities)

    def predict_calibrated(self, X, method="isotonic"):
        return self.calibrate_probabilities(self.predict(X), method)

    def score(self, X, y):
        return self._get_accuracy(y, self.predict(X))

    def score_calibrated(self, X, y, method="isotonic"):
        return self._get_accuracy(y, self.predict_calibrated(X, method))

    def _get_accuracy(self, y, preds):
        return np.mean(np.equal(y.astype(np.bool), preds >= 0.5))


class H2OModel(CalibratableModelMixin):
    def train(self, X, y):
        self.features = list(range(len(X[0])))
        self.target = "target"
        train_frame = self._to_h2o_frame(X, y)
        self.model.train(x=self.features, y=self.target, training_frame=train_frame)

    def predict(self, X):
        predict_frame = self._to_h2o_frame(X)
        return self.model.predict(predict_frame).as_data_frame()["p1"].to_numpy()

    def _to_h2o_frame(self, X, y=None):
        df = pd.DataFrame(data=X, columns=self.features)
        if y is not None:
            df[self.target] = y
        h2o_frame = h2o.H2OFrame(df)
        if y is not None:
            h2o_frame[self.target] = h2o_frame[self.target].asfactor()
        return h2o_frame


class ScikitModel(CalibratableModelMixin):
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class KerasModel(CalibratableModelMixin):
    def train(self, X, y):
        self.model.fit(X, y, batch_size=128, epochs=10, verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()
