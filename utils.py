import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from h2o.estimators import H2OEstimator
import h2o
import pandas as pd

def plot_sample(X, title=None, size=6):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, im in enumerate(np.random.permutation(X)[:size]):
        ax1 = fig.add_subplot(1, size, i+1)
        plt.imshow(im.reshape(28,28), cmap='gray')
    fig.tight_layout()
    st.set_y(0.6)
    fig.subplots_adjust(top=0.85)
    plt.show()
    
    
def predict_positive_probs(clf, X):
    if isinstance(clf, H2OEstimator):
        df = pd.DataFrame(data = X, columns=[*list(range(len(X[0])))])
        return clf.predict(h2o.H2OFrame(df)).as_data_frame()['p1'].tolist()
    elif isinstance(clf, ClassifierMixin) and hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    elif isinstance(clf, ClassifierMixin) and hasattr(clf, "decision_function"):
        probs = clf.decision_function(X)
        return (probs - probs.min()) / (probs.max() - probs.min())
    else:
        raise TypeError("Unrecognized classifier")
        
def train_classifier(clf, X_train, y_train, X_test, y_test):
    if isinstance(clf, H2OEstimator):
        train = _to_h2o_frame(X_train, y_train)
        test = _to_h2o_frame(X_test, y_test)
        clf.train(x=[str(x) for x in range(len(X_train[0]))],
                 y="target",
                 training_frame=train,
                 validation_frame=test)
        accuracy = clf.model_performance(test).accuracy()
        accuracy = round(100*accuracy[0][1], 2)
    elif isinstance(clf, ClassifierMixin):
        clf.fit(X_train, y_train)
        accuracy = round(100*clf.score(X_test, y_test), 2)
    else:
        raise TypeError("Unrecognized classifier")
    print(f"Trained {clf.__class__.__name__}, accuracy: {accuracy}%")
    
def _to_h2o_frame(X, y):
    df = pd.DataFrame(data = [[*data, target] for data, target in zip(X,y) ], columns=[*list(range(len(X[0]))), 'target'])
    h2o_frame =  h2o.H2OFrame(df)
    h2o_frame['target'] = h2o_frame['target'].asfactor()
    return h2o_frame

    