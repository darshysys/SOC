"""
q1_solve.py
============
Students must implement the following functions and classes:
- load_data
- train_test_split_custom
- OLSClassifier
- LogisticRegressionGD
"""
import numpy as np

# ============================================
# TASK 0: DATA LOADING
# ============================================

def load_data(file_path: str):
    """
    Load dataset from a given CSV file.

    Parameters
    ----------
    file_path : str
        Path to CSV file. Last column should be the label (y in {-1, +1})

    Returns
    -------
    X : np.ndarray
        Features of shape (N, d)
    y : np.ndarray
        Labels of shape (N,), values in {-1, +1}
    """
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    all_rows = []

    for i in range(len(lines)):
        line = lines[i]
        line = line.strip()  
        if line == '':
            continue

        values = line.split(',')
        row = []
        for j in range(len(values)):
            row.append(float(values[j]))
        all_rows.append(row)

    data = np.array(all_rows)
    num_rows = data.shape[0]
    num_cols = data.shape[1]

    X = data[:, 0:num_cols-1]
    y = data[:, num_cols-1]

    return X, y


# ============================================
# TASK 0.5: TRAIN TEST SPLIT
# ============================================

def test_train_split(X: np.ndarray, y: np.ndarray, test_size=0.2):
    """
    Split dataset into train and test sets.
    DO NOT alter the order of samples in X and y
    """
    N = X.shape[0]

    N_train = int(N*(1-test_size))
    N_test = N-N_train

    X_train = X[0:N_train]
    X_test = X[N_train:N_train + N_test]

    y_train = y[0:N_train]
    y_test = y[N_train:N_train + N_test]

    return X_train, X_test, y_train, y_test


def normalized_test_train_split(X: np.ndarray, y: np.ndarray, test_size=0.2, test_train_split_func=test_train_split):
    """
    Split dataset into train and test sets and normalize features.
    """
    X_train, X_test, y_train, y_test = test_train_split_func(X, y, test_size)

    mean_vals = np.zeros(X_train.shape[1])
    std_vals = np.zeros(X_train.shape[1])

    for col in range(X_train.shape[1]):
        mean_vals[col] = np.mean(X_train[:, col])
        std_vals[col] = np.std(X_train[:, col])

    X_train_normalized = np.zeros(X_train.shape)
    for col in range(X_train.shape[1]):
        if std_vals[col] == 0:
            X_train_normalized[:, col] = 0
        else:
            X_train_normalized[:, col] = (X_train[:, col]-mean_vals[col])/std_vals[col]

    X_test_normalized = np.zeros(X_test.shape)
    for col in range(X_test.shape[1]):
        if std_vals[col] == 0:
            X_test_normalized[:, col] = 0
        else:
            X_test_normalized[:, col] = (X_test[:, col]-mean_vals[col])/std_vals[col]

    return X_train_normalized, X_test_normalized, y_train, y_test


# ============================================
# TASK 1: LEAST SQUARES CLASSIFICATION
# ============================================

class OLSClassifier:
    """
    Ordinary Least Squares classifier using gradient descent.
    Predicts labels {-1, +1}.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.mse_loss_history = []

    def linear_gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss.
        MSE = (1/N) * sum((Xw - y)^2)
        gradient = (2/N) * X^T * (Xw - y)
        """
        N = X.shape[0]
        predictions = np.dot(X, w)
        errors = predictions-y
        grad = (2/N)*np.dot(X.T, errors)

        return grad

    def compute_mse_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MSE loss = (1/N) * sum((Xw - y)^2)
        """
        N = X.shape[0]
        predictions = np.dot(X, w)
        errors = predictions-y
        squared_errors = errors**2
        loss = np.sum(squared_errors)/N

        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear model using gradient descent.
        """
        num_features = X.shape[1]
        iteration = 0
        converged = False

        self.w = np.zeros(num_features)
        self.mse_loss_history = []
        while iteration < self.max_iter and converged == False:
            grad = self.linear_gradient(self.w, X, y)
            w_new = self.w-self.lr*grad
            weight_change = np.linalg.norm(w_new-self.w)

            self.w = w_new
            current_loss = self.compute_mse_loss(self.w, X, y)
            self.mse_loss_history.append(current_loss)
            if weight_change < self.tol:
                converged = True
            iteration = iteration+1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels {-1, +1}.
        """
        raw_predictions = np.dot(X, self.w)
        num_samples = X.shape[0]
        y_pred = np.zeros(num_samples)
        for i in range(num_samples):
            if raw_predictions[i] >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1

        return y_pred


# ============================================
# TASK 2: LOGISTIC REGRESSION
# ============================================

class LogisticRegressionGD:
    """
    Logistic Regression classifier using batch gradient descent.
    Uses labels {-1, +1} with logistic loss = (1/N) * sum(log(1 + exp(-y * Xw)))
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.logistic_loss_history = []

    def logistic_gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of logistic loss.
        Loss = (1/N) * sum(log(1 + exp(-y * w^T x)))
        Gradient = -(1/N) * X^T * (y * sigmoid(-y * Xw))
        """
        N = X.shape[0]
        margin = y*np.dot(X, w)
        sigmoid_vals = 1/(1+np.exp(margin))
        grad = -(1/N)*np.dot(X.T, y*sigmoid_vals)
        return grad

    def compute_logistic_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic loss = (1/N) * sum(log(1 + exp(-y * Xw)))
        """
        N = X.shape[0]
        margin = y*np.dot(X, w)
        loss_per_sample = np.zeros(N)
        for i in range(N):
            loss_per_sample[i] = np.log(1+np.exp(-margin[i]))
        total_loss = np.sum(loss_per_sample)/N
        return total_loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit logistic regression using batch gradient descent.
        """
        num_features = X.shape[1]
        iteration = 0
        converged = False

        self.w = np.zeros(num_features)
        self.logistic_loss_history = []
        while iteration < self.max_iter and converged == False:
            grad = self.logistic_gradient(self.w, X, y)
            w_new = self.w - self.lr * grad
            weight_change = np.linalg.norm(w_new - self.w)
            self.w = w_new
            current_loss = self.compute_logistic_loss(self.w, X, y)
            self.logistic_loss_history.append(current_loss)
            if weight_change < self.tol:
                converged = True
            iteration = iteration + 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels {-1, +1}.
        If w^T x >= 0, predict +1, else predict -1.
        """
        scores = np.dot(X, self.w)
        num_samples = X.shape[0]
        y_pred = np.zeros(num_samples)
        for i in range(num_samples):
            if scores[i] >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        return y_pred