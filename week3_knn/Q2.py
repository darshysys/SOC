import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data and labels.
        Parameters:
            X: Training features (numpy array)
            y: Training labels (numpy array)
        """
        self.X_train = X
        self.y_train = y

    def predict_L2(self, X_test, k):
        """
        Predict labels for the test set using L2 (Euclidean) distance.
        Parameters:
            X_test: Test features (numpy array)
            k: Number of neighbors
        Returns:
            y_pred: Predicted labels for X_test (numpy array of +1 or -1)
        """
        # TODO: Implement vectorized L2 distance and majority vote
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test)
        train_sq_sum = np.sum(self.X_train**2, axis=1)
        test_sq_sum = np.sum(X_test**2, axis=1)
        cross_term = np.dot(X_test, self.X_train.T)
        dist_matrix = test_sq_sum.reshape(-1, 1) + train_sq_sum.reshape(1, -1) - 2*cross_term
        for i in range(num_test):
            dist_row = dist_matrix[i]
            sorted_indices = np.argsort(dist_row)
            top_k_indices = sorted_indices[0:k]
            top_k_labels = self.y_train[top_k_indices]
            vote_sum = np.sum(top_k_labels)
            if vote_sum >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        return y_pred

    def predict_L1(self, X_test, k):
        """
        Predict labels for the test set using L1 (Manhattan) distance.
        Parameters:
            X_test: Test features (numpy array)
            k: Number of neighbors
        Returns:
            y_pred: Predicted labels for X_test (numpy array of +1 or -1)
        """
        # TODO: Implement vectorized L1 distance and majority vote
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test)
        diff_matrix = np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
        dist_matrix = np.sum(diff_matrix, axis=2)
        for i in range(num_test):
            dist_row = dist_matrix[i]
            sorted_indices = np.argsort(dist_row)
            top_k_indices = sorted_indices[0:k]
            top_k_labels = self.y_train[top_k_indices]
            vote_sum = np.sum(top_k_labels)
            if vote_sum >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        return y_pred

def compute_accuracy(y_true, y_pred):
    """
    Calculate the percentage of correct predictions.
    Parameters:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        accuracy: Float representing accuracy
    """
    # TODO: Implement accuracy calculation
    num_correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            num_correct = num_correct + 1
    accuracy = num_correct / total
    return accuracy

def standardize(X_train, X_test):
    """
    Standardize features to mean 0 and variance 1.
    Parameters:
        X_train: Raw training features
        X_test: Raw test features
    Returns:
        X_train_std, X_test_std: Standardized feature arrays
    """
    # TODO: Standardize X_test using statistics derived ONLY from X_train
    num_cols = X_train.shape[1]
    mean_vals = np.zeros(num_cols)
    std_vals = np.zeros(num_cols)
    for j in range(num_cols):
        col_sum = 0
        for i in range(X_train.shape[0]):
            col_sum = col_sum + X_train[i][j]
        mean_vals[j] = col_sum / X_train.shape[0]
    for j in range(num_cols):
        sq_sum = 0
        for i in range(X_train.shape[0]):
            diff = X_train[i][j] - mean_vals[j]
            sq_sum = sq_sum + (diff*diff)
        std_vals[j] = (sq_sum / X_train.shape[0])**0.5
    X_train_std = np.zeros(X_train.shape)
    X_test_std = np.zeros(X_test.shape)
    for i in range(X_train.shape[0]):
        for j in range(num_cols):
            if std_vals[j] == 0:
                X_train_std[i][j] = 0
            else:
                X_train_std[i][j] = (X_train[i][j] - mean_vals[j]) / std_vals[j]
    for i in range(X_test.shape[0]):
        for j in range(num_cols):
            if std_vals[j] == 0:
                X_test_std[i][j] = 0
            else:
                X_test_std[i][j] = (X_test[i][j] - mean_vals[j]) / std_vals[j]
    return X_train_std, X_test_std

def get_pearson_indices(X, y, m):
    """
    Select top m features based on absolute Pearson correlation with label y.
    Parameters:
        X: Feature array
        y: Label array
        m: Number of features to select
    Returns:
        indices: Array of indices for the top m features
    """
    # TODO: Implement vectorized Pearson correlation and return top m indices
    num_cols = X.shape[0]
    num_features = X.shape[1]
    corr_vals = np.zeros(num_features)
    y_mean = np.mean(y)
    y_diff = y - y_mean
    y_std = np.sqrt(np.sum(y_diff**2))
    for j in range(num_features):
        x_col = X[:, j]
        x_mean = np.mean(x_col)
        x_diff = x_col - x_mean
        x_std = np.sqrt(np.sum(x_diff**2))
        if x_std == 0 or y_std == 0:
            corr_vals[j] = 0
        else:
            corr_vals[j] = np.sum(x_diff*y_diff) / (x_std*y_std)
    abs_corr_vals = np.abs(corr_vals)
    sorted_indices = np.argsort(abs_corr_vals)
    sorted_indices = sorted_indices[::-1]
    indices = sorted_indices[0:m]
    return indices

if __name__ == "__main__":
    # you are allowed to use loops here

    # TODO: Load data using pandas
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # TODO: Execute Task A (Vary k, use L2)
    knn_model = KNN()
    knn_model.fit(X_train, y_train)
    k_values = [1, 5, 10, 20, 50]
    for k_val in k_values:
        y_pred = knn_model.predict_L2(X_test, k_val)
        acc = compute_accuracy(y_test, y_pred)
        print("Task A - k =", k_val, "Accuracy =", acc)

    # TODO: Execute Task B (Standardize, then vary m for Pearson selection, use k=20, L2)
    X_train_std, X_test_std = standardize(X_train, X_test)
    knn_model_std = KNN()
    m_values = [1, 5, 10, 20, 50]
    for m_val in m_values:
        top_m_indices = get_pearson_indices(X_train_std, y_train, m_val)
        X_train_m = X_train_std[:, top_m_indices]
        X_test_m = X_test_std[:, top_m_indices]
        knn_model_std.fit(X_train_m, y_train)
        y_pred = knn_model_std.predict_L2(X_test_m, 20)
        acc = compute_accuracy(y_test, y_pred)
        print("Task B - m =", m_val, "Accuracy =", acc)

    # TODO: Execute Task C (Standardize, use all features, use k=20, L1)
    knn_model_l1 = KNN()
    knn_model_l1.fit(X_train_std, y_train)
    y_pred_l1 = knn_model_l1.predict_L1(X_test_std, 20)
    acc_l1 = compute_accuracy(y_test, y_pred_l1)
    print("Task C - L1 k=20 Accuracy =", acc_l1)