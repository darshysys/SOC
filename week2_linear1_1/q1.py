import json
import numpy as np

def ols_with_intercept(X, y):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    w_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    w0 = w_full[0]
    w = w_full[1:]
    return w, w0

def ols_no_intercept(X, y):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

def predict_with_intercept(X, w, w0):
    y_hat = np.dot(X, w) + w0
    return y_hat


def predict_no_intercept(X, w):
    y_hat = np.dot(X, w)
    return y_hat

def compute_metrics(y, y_hat):
    N = len(y)

    mse = np.sum((y-y_hat)**2)/N

    corr_matrix = np.corrcoef(y, y_hat)
    corr = corr_matrix[0, 1]

    corr_sq = corr**2

    y_mean = np.mean(y)
    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_mean)**2)
    r2 = 1 - ss_res/ss_tot

    return {
        "mse": float(mse),
        "corr": float(corr),
        "corr_sq": float(corr_sq),
        "r2": float(r2),
    }


def load_data():
    train_data = np.genfromtxt('q1_train.csv', delimiter=',')
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    test_data = np.genfromtxt('q1_test.csv', delimiter=',')
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    outlier_data = np.genfromtxt('q1_outliers.csv', delimiter=',')
    X_outliers = outlier_data[:, :-1]
    y_outliers = outlier_data[:, -1]

    X_train_outlier = np.vstack([X_train, X_outliers])
    y_train_outlier = np.hstack([y_train, y_outliers])

    return X_train, y_train, X_test, y_test, X_train_outlier, y_train_outlier


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_train_outlier, y_train_outlier = load_data()

    w, w0 = ols_with_intercept(X_train, y_train)

    yhat_train = predict_with_intercept(X_train, w, w0)
    yhat_test = predict_with_intercept(X_test, w, w0)

    standard_train_metrics = compute_metrics(y_train, yhat_train)
    standard_test_metrics = compute_metrics(y_test, yhat_test)

    w_o, w0_o = ols_with_intercept(X_train_outlier, y_train_outlier)

    yhat_train_outlier = predict_with_intercept(X_train_outlier, w_o, w0_o)
    yhat_test_outlier = predict_with_intercept(X_test, w_o, w0_o)

    outlier_train_metrics = compute_metrics(y_train_outlier, yhat_train_outlier)
    outlier_test_metrics = compute_metrics(y_test, yhat_test_outlier)

    w_no = ols_no_intercept(X_train, y_train)

    yhat_train_no = predict_no_intercept(X_train, w_no)
    yhat_test_no = predict_no_intercept(X_test, w_no)

    no_intercept_train_metrics = compute_metrics(y_train, yhat_train_no)
    no_intercept_test_metrics = compute_metrics(y_test, yhat_test_no)

    metrics = {
        "standard_ols": {
            "train": standard_train_metrics,
            "test": standard_test_metrics,
        },
        "outlier_ols": {
            "train": outlier_train_metrics,
            "test": outlier_test_metrics,
        },
        "no_intercept_ols": {
            "train": no_intercept_train_metrics,
            "test": no_intercept_test_metrics,
        },
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)