import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        self.regressor = LinearRegression().fit(
            prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1)
        )

    def calibrate(self, probabilities):
        return 1 / (1 + np.exp(-self.regressor.predict(probabilities.reshape(-1, 1))))

    def _filter_out_of_domain(self, prob_pred, prob_true):
        filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
        return np.array(filtered)


class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)