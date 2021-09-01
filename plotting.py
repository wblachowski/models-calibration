import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_sample(X, title=None, size=6):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, im in enumerate(np.random.permutation(X)[:size]):
        fig.add_subplot(1, size, i + 1)
        plt.imshow(im.reshape(28, 28), cmap="gray")
    fig.tight_layout()
    st.set_y(0.6)
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_calibration_curve(y, probs, title):
    brier_score = brier_score_loss(y, probs)
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(
        prob_pred,
        prob_true,
        marker=".",
        color="orange",
    )
    plt.title(f"{title}\nBrier score: {round(brier_score, 4)}")
    plt.show()
    return prob_true, prob_pred


def plot_fitted_calibrator(prob_true, prob_pred, prob_calibrated):
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker=".", color="orange")
    plt.plot(prob_pred, prob_calibrated, color="red")
    plt.title(f"Fitted calibrator")
    plt.show()