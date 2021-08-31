import matplotlib.pyplot as plt
import numpy as np


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