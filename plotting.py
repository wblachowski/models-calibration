import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_sample(X, title=None, size=8):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, im in enumerate(np.random.permutation(X)[:size]):
        fig.add_subplot(1, size, i + 1)
        plt.axis("off")
        plt.imshow(im.reshape(28, 28), cmap="gray")
    fig.tight_layout()
    st.set_y(0.59)


def plot_sample_predictions(models, X, X_unscaled, size=8):
    indexes = np.random.permutation(len(X))[:size]
    fig = plt.figure(figsize=(6, len(models) * 1.6), constrained_layout=True)
    subfigs = fig.subfigures(len(models))
    for j, model in enumerate(models):
        subfig = subfigs.flat[j]
        subfig.suptitle(model.name)
        for i, idx in enumerate(indexes):
            prediction = model.predict(np.array([X[idx]]))[0]
            subfig.add_subplot(1, size, i + 1)
            plt.axis("off")
            plt.title(f"{round(100*prediction,2)}%", fontsize=10)
            plt.imshow(X_unscaled[idx].reshape(28, 28), cmap="gray")


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
    plt.title(f"{title}\nBrier score: {round(brier_score, 3)}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    return prob_true, prob_pred


def plot_fitted_calibrator(prob_true, prob_pred, prob_calibrated, title=None):
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker=".", color="orange")
    plt.plot(prob_pred, prob_calibrated, color="red")
    plt.title(title)
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")


def plot_calibration_details_for_models(
    models, X, y, calibrated=False, method="isotonic"
):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model in models:
        name = model.name
        probabilities = (
            model.predict(X)
            if not calibrated
            else model.predict_calibrated(X, method=method)
        )
        prob_true, prob_pred = calibration_curve(y, probabilities, n_bins=10)
        brier_score = brier_score_loss(y, probabilities)

        ax1.plot(
            prob_pred,
            prob_true,
            marker=".",
            label=f"{name} (BS={round(brier_score, 3)})",
        )

        ax2.hist(
            probabilities, range=(0, 1), bins=10, label=name, histtype="step", lw=2
        )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title("Calibration plots")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)