#!/usr/bin/env python3
"""
q1_solve.py

Implement ALL functions below.
Do NOT import sklearn.
"""

import numpy as np
from typing import List, Tuple


# -------------------------
# Utilities
# -------------------------
def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add bias (column of ones) as first column.
    See main.py for usage.
    """
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    new_X = np.zeros((num_rows, num_cols+1))
    for i in range(num_rows):
        new_X[i][0] = 1
        for j in range(num_cols):
            new_X[i][j+1] = X[i][j]
    return new_X


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    total_error = 0
    n = len(y)
    for i in range(n):
        diff = y[i] - y_pred[i]
        total_error = total_error + (diff*diff)
    mean_error = total_error / n
    return mean_error


def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training statistics.
    Returns standardized X, mean, and stddev.
    See main.py for usage.
    """
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    mean_vals = np.zeros(num_cols)
    std_vals = np.zeros(num_cols)
    for j in range(num_cols):
        col_sum = 0
        for i in range(num_rows):
            col_sum = col_sum + X[i][j]
        mean_vals[j] = col_sum / num_rows
    for j in range(num_cols):
        sq_sum = 0
        for i in range(num_rows):
            diff = X[i][j] - mean_vals[j]
            sq_sum = sq_sum + (diff*diff)
        std_vals[j] = (sq_sum / num_rows)**0.5
    X_standardized = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if std_vals[j] == 0:
                X_standardized[i][j] = 0
            else:
                X_standardized[i][j] = (X[i][j] - mean_vals[j]) / std_vals[j]
    return X_standardized, mean_vals, std_vals


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply training standardization.
    See main.py for usage.
    """
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    X_standardized = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if std[j] == 0:
                X_standardized[i][j] = 0
            else:
                X_standardized[i][j] = (X[i][j] - mean[j]) / std[j]
    return X_standardized


# -------------------------
# Ridge Regression
# -------------------------
def ridge_regression_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge regression:
        (X^T X + λD) w = X^T y
    where D[0,0] = 0 (bias not regularized).
    """
    num_cols = X.shape[1]
    D = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        if i == 0:
            D[i][i] = 0
        else:
            D[i][i] = 1
    XtX = np.dot(X.T, X)
    lam_D = lam * D
    A = XtX + lam_D
    Xty = np.dot(X.T, y)
    w = np.linalg.solve(A, Xty)
    return w


# -------------------------
# Cross-validation
# -------------------------
def k_fold_split(N: int, k: int) -> List[np.ndarray]:
    """
        k-fold split after shuffling
        Returns list of k arrays of indices.
    """
    all_indices = np.arange(N)
    np.random.shuffle(all_indices)
    fold_list = []
    fold_size = N // k
    start_index = 0
    for i in range(k):
        if i == k-1:
            one_fold = all_indices[start_index:]
        else:
            one_fold = all_indices[start_index:start_index+fold_size]
        fold_list.append(one_fold)
        start_index = start_index + fold_size
    return fold_list


def ridge_cv(X: np.ndarray, y: np.ndarray, lam: float, k: int) -> float:
    """
    k-fold CV MSE for ridge.
    Use the k_fold_split function above to get the folds
    Use ridge_regression_closed_form to fit the model.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lam: regularization parameter
        k: number of folds
    Returns average MSE across folds.
    """
    fold_list = k_fold_split(X.shape[0], k)
    total_mse = 0
    for fold_index in range(k):
        val_indices = fold_list[fold_index]
        train_indices_list = []
        for j in range(k):
            if j != fold_index:
                for idx in fold_list[j]:
                    train_indices_list.append(idx)
        train_indices = np.array(train_indices_list)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        w = ridge_regression_closed_form(X_train, y_train, lam)
        y_pred = np.dot(X_val, w)
        fold_mse = mse(y_val, y_pred)
        total_mse = total_mse + fold_mse
    avg_mse = total_mse / k
    return avg_mse


# -------------------------
# Hyperparameter search
# -------------------------
def grid_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    lambdas: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate each λ using CV.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lambdas: (M,) array of λ values to evaluate
        k: number of folds
    Returns:
        lambdas: (M,) same as input
        mses: (M,) average CV MSE for each λ
    """
    num_lambdas = len(lambdas)
    mses = np.zeros(num_lambdas)
    for i in range(num_lambdas):
        current_lambda = lambdas[i]
        current_mse = ridge_cv(X, y, current_lambda, k)
        mses[i] = current_mse
    return lambdas, mses


def random_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    n_iter: int,
    low_exp: float,
    high_exp: float,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample λ = 10^u, where u ~ Uniform(low_exp, high_exp).
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        n_iter: number of λ values to sample
        low_exp: lower bound of exponent
        high_exp: upper bound of exponent
        k: number of folds
    Returns:
        lambdas: (n_iter,) sampled λ values
        mses: (n_iter,) average CV MSE for each λ
    """
    sampled_lambdas = np.zeros(n_iter)
    mses = np.zeros(n_iter)
    for i in range(n_iter):
        u = np.random.uniform(low_exp, high_exp)
        current_lambda = 10**u
        sampled_lambdas[i] = current_lambda
        current_mse = ridge_cv(X, y, current_lambda, k)
        mses[i] = current_mse
    return sampled_lambdas, mses