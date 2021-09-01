import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log((1 / np.array(prob_true)) - 1).reshape(-1, 1)
        prob_pred = np.array(prob_pred).reshape(-1, 1)
        model = LinearRegression().fit(prob_pred, prob_true)
        self.alpha = model.coef_[0, 0]
        self.beta = model.predict([[0]])[0, 0]

    def calibrate(self, probabilities):
        return 1 / (1 + np.exp(self.alpha * np.array(probabilities) + self.beta))

    def _filter_out_of_domain(self, prob_pred, prob_true):
        return list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))


class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)