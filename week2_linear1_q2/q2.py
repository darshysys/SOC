import numpy as np
import matplotlib.pyplot as plt

def poly_features(x, degree):
    N = len(x)
    Phi = np.zeros((N, degree+1))
    for d in range(degree+1):
        Phi[:, d] = x**d
    return Phi

def fit_ols(Phi, y):
    w = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return w

def predict(x, degree, w):
    Phi = poly_features(x, degree)
    y_hat = np.dot(Phi, w)
    return y_hat

def mse(y, y_hat):
    loss = np.sum((y-y_hat)**2)/len(y)
    return loss

def k_fold_cv(x, y, degree, k=5):
    N = len(x)
    indices = np.arange(N)
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    fold_size = N//k
    val_mse_list = []

    for fold in range(k):
        val_start = fold*fold_size
        val_end = val_start+fold_size

        x_val = x_shuffled[val_start:val_end]
        y_val = y_shuffled[val_start:val_end]

        x_train_fold = np.concatenate([x_shuffled[:val_start], x_shuffled[val_end:]])
        y_train_fold = np.concatenate([y_shuffled[:val_start], y_shuffled[val_end:]])

        Phi_train = poly_features(x_train_fold, degree)
        w = fit_ols(Phi_train, y_train_fold)

        y_hat_val = predict(x_val, degree, w)
        val_mse_list.append(mse(y_val, y_hat_val))

    avg_val_mse = np.mean(val_mse_list)
    return avg_val_mse


def evaluate_degrees(x, y, D_max, k=5):
    train_mse = []
    cv_mse = []

    for d in range(1, D_max+1):
        Phi = poly_features(x, d)
        w = fit_ols(Phi, y)

        y_hat_train = predict(x, d, w)
        train_mse.append(mse(y, y_hat_train))

        cv_mse.append(k_fold_cv(x, y, d, k))

    return train_mse, cv_mse

def select_degree(cv_mse):
    best_degree = 1
    best_mse = cv_mse[0]
    for i in range(1, len(cv_mse)):
        if cv_mse[i] < best_mse:
            best_mse = cv_mse[i]
            best_degree = i+1
    return best_degree

def fit_final_model(x, y, degree):
    Phi = poly_features(x, degree)
    w = fit_ols(Phi, y)
    return w

def plot_errors(train_mse, cv_mse):
    D = len(train_mse)
    plt.plot(range(1, D+1), train_mse, label="Training MSE")
    plt.plot(range(1, D+1), cv_mse, label="CV MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def load_data():
    data = np.genfromtxt('q2_train.csv', delimiter=',')
    x_train = data[:, 0]
    y_train = data[:, 1]
    return x_train, y_train

if __name__ == "__main__":
    np.random.seed(1234)

    x_train, y_train = load_data()
    D_max = 10
    k = 5

    train_mse, cv_mse = evaluate_degrees(x_train, y_train, D_max, k)

    best_d = select_degree(cv_mse)
    w = fit_final_model(x_train, y_train, best_d)

    print("Optimal degree:", best_d)
    print("Weights:", w)

    plot_errors(train_mse, cv_mse)