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


def plot_fitted_calibrator(prob_true, prob_pred, prob_calibrated, title=None):
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker=".", color="orange")
    plt.plot(prob_pred, prob_calibrated, color="red")
    plt.title(title)
    plt.show()


def plot_calibration_info_for_models(models, X, y):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model in models:
        name = model.name
        prob_pos = model.predict(X)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, prob_pos, n_bins=10
        )
        brier_score = brier_score_loss(fraction_of_positives, mean_predicted_value)

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker=".",
            label=f"{name} (BS={round(brier_score, 2)})",
        )

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.show()