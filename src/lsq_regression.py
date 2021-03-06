""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def task12():
    """ Subtask 1: Least Squares and Double Descent Phenomenon

        Requirements for the plot:
        - make one subplot for each lambda
        - each subplot should contain mean and std. of train/test errors
        - labels for mean train/test errors are 'mean train error', 'mean test error' and must be included in the plots

        Subtask 2: Dual Representation with Kernel

        Requirements for the plots:
        - make one subplot for each M
        - each subplot should contain the n=10th row of both the kernel matrix and the feature product \Phi\Phi^T
        - labels should be "features" and "kernel" and must be included in a legend
        - each subplot must contain a title with the number of random features and the mean absolute difference between kernel and feature product.
    """

    fig1, ax1 = plt.subplots(1, 3, figsize=(17, 5))
    plt.suptitle('Task 1 - Regularized Least Squares and Double Descent Phenomenon (without ylim)', fontsize=16)
    for a in ax1.reshape(-1):
        a.set_ylabel('error')
        a.set_xlabel('number of random features')

    fig1b, ax1b = plt.subplots(1, 3, figsize=(17, 5))
    plt.suptitle('Task 1 - Regularized Least Squares and Double Descent Phenomenon (using ylim)', fontsize=16)
    for a in ax1b.reshape(-1):
        a.set_ylim([0, 40])
        a.set_ylabel('error')
        a.set_xlabel('number of random features')

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle('Task 2 - Dual Representation with Kernel', fontsize=16)

    lams = [1e-8, 1e-5, 1e-3]  # use this for subtask 1
    m_array = [10, 200, 800]  # use this for subtask 2
    mae_array = 1e3 * np.ones((3))  # use this for subtask 2 (MAE = mean absolute error)

    """ Start of your code 
    """

    # parameters
    d = 5
    sigma = 2

    # 1.4: Create training data
    N = 200
    X, y = __create_data(N, d, sigma)

    # 1.4: Create test data
    X_t, y_t = __create_data(50, d, sigma)

    # test_losses, train_losses = __task_1(lams, N, d, X, y, X_t, y_t)
    dual_kernels_train, primal_kernels_train, MAEs = __task_2(lams[0], N, d, X, y, X_t, y_t)

    """ End of your code
    """

    # __plot_task_1(ax1, ax1b, lams, train_losses, test_losses)
    __plot_task_2(ax2, m_array, dual_kernels_train, primal_kernels_train, MAEs)

    return fig1, fig1b, fig2


def __plot_task_2(ax2, m_array, dual_kernels_train, primal_kernels_train, MAEs):
    for m_idx, a in enumerate(ax2.reshape(-1)):
        a.legend()
        a.set_title('#Features M=%i, MAE=%f' % (m_array[m_idx], (MAEs[m_idx])))
        x_axis = range(len(dual_kernels_train[m_idx][9, :]))
        a.plot(x_axis, dual_kernels_train[m_idx][9, :])
        a.plot(x_axis, primal_kernels_train[m_idx][9, :])


def __plot_task_1(ax1, ax1b, lams, train_losses, test_losses):
    features = []
    [features.append(10 * k + 1) for k in range(61)]

    train_mean = np.mean(train_losses, axis=2)
    test_mean = np.mean(test_losses, axis=2)
    train_std = np.std(train_losses, axis=2)
    test_std = np.std(test_losses, axis=2)

    # task 1: without ylim
    for lam_idx, a in enumerate(ax1.reshape(-1)):
        a.plot(features, train_mean[lam_idx, :], label='train')
        a.fill_between(features, train_mean[lam_idx, :] - train_std[lam_idx, :], train_mean[lam_idx, :] + train_std[lam_idx, :], alpha=0.5)

        a.plot(features, test_mean[lam_idx, :], label='test')
        a.fill_between(features, test_mean[lam_idx, :] - test_std[lam_idx, :], test_mean[lam_idx, :] + test_std[lam_idx, :], alpha=0.5)

        a.legend()
        a.set_title(r'$\lambda=$' + str(lams[lam_idx]))

    # task 1: with ylim
    for lam_idx, a in enumerate(ax1b.reshape(-1)):
        a.plot(features, train_mean[lam_idx, :], label='train')
        a.fill_between(features, train_mean[lam_idx, :] - train_std[lam_idx, :], train_mean[lam_idx, :] + train_std[lam_idx, :], alpha=0.5)

        a.plot(features, test_mean[lam_idx, :], label='test')
        a.fill_between(features, test_mean[lam_idx, :] - test_std[lam_idx, :], test_mean[lam_idx, :] + test_std[lam_idx, :], alpha=0.5)

        a.legend()
        a.set_title(r'$\lambda=$' + str(lams[lam_idx]))


def __task_1(lams, N, d, X, y, X_t, y_t):
    train_losses = np.zeros((3, 61, 5))
    test_losses = np.zeros((3, 61, 5))

    for l in range(3):  # loop over lambdas
        print(str.format("Calculating lambda {0}", lams[l]))
        lambda_ = lams[l]  # lambda value to reproduce double desecent phenomenon
        for v in range(0, 61):  # loop over feature vectors
            for r in range(5):  # experiments
                V = __create_feature_vectors(v, d)
                I = np.diag(np.ones(V.shape[0]))

                # 1.6: Implement w* and calculate MSEy
                theta = __calculate_theta(V, X)
                theta_t = __calculate_theta(V, X_t)
                A = lambda_ * I + theta.T @ theta
                Q, R = np.linalg.qr(A)
                z = Q.T @ theta.T @ y

                # w_analytical_1 = np.linalg.inv(A) @ theta.T @ y
                # w_analytical_2 = np.linalg.pinv(A) @ theta.T @ y

                # 1.7: Reproduce double descent phenomenon
                w_ml = np.linalg.solve(R, z)  # computation via QR decomposition
                # w_ml = w_analytical_1
                mse_train, mse_test = __perform_linear_regression(N, theta, theta_t, w_ml, y, y_t)
                train_losses[l, v, r] = mse_train  # average mse
                test_losses[l, v, r] = mse_test

    return test_losses, train_losses


def __task_2(lam, N, d, X, y, X_t, y_t):
    # 2.4
    base_kernel_train = __calculate_kernel(X=X, X_prime=X, d=d)
    base_kernel_test = __calculate_kernel(X=X_t, X_prime=X, d=d)
    alpha = __calculate_alpha_dual_problem(base_kernel_train, lam, y)
    mse_train, mse_test = __perform_linear_regression_kernel(N, lam, alpha, base_kernel_train, base_kernel_test, y, y_t)

    print(f"MSE for training: {np.around(mse_train, 2)}")
    print(f"MSE for test: {np.around(mse_test, 2)}")

    # 2.5
    primal_kernels_train = []
    dual_kernels_train = []
    MAEs = []

    # Calculate train and test errors for dual solution
    mse_train_dual, mse_test_dual = __perform_linear_regression_kernel(N, lam, alpha, base_kernel_train, base_kernel_test, y, y_t)

    for m in [10, 200, 800]:
        print(f'Number of features = {m}:')

        # Create both kernels
        V = __create_feature_vectors_k(m, d)
        dual_kernel_train = __calculate_kernel(X=V, X_prime=X, d=d)
        dual_kernels_train.append(dual_kernel_train)
        theta_train = __calculate_theta(V, X)
        primal_kernel_train = theta_train @ theta_train.T
        primal_kernels_train.append(primal_kernel_train)
        mae = np.mean(np.abs(dual_kernel_train[9] - primal_kernel_train[9]))
        MAEs.append(mae)
        print(f'Comparison of MAE between K and theta * theta.T = {np.around(mae, 2)}')

        theta_test = __calculate_theta(V, X_t)

        # # Calculate train and test error for primal solution
        # I = np.diag(np.ones(V.shape[0]))
        # mse_train_primal = __calculate_primal_error(I, lam, theta_train, y)
        # mse_test_primal = __calculate_primal_error(I, lam, theta_test, y_t)
        #
        # print(f'train primal = {np.around(mse_train_primal, 2)}; train dual = {np.around(mse_train_dual, 2)}; train diff = {np.around(np.abs(mse_train_dual - mse_train_primal), 2)}')
        # print(f'test primal = {np.around(mse_test_primal, 2)}; test dual = {np.around(mse_test_dual, 2)}; test diff = {np.around(np.abs(mse_test_dual - mse_test_primal), 2)}')
        # print('----------------------------------------------')

    return dual_kernels_train, primal_kernels_train, MAEs


def __calculate_primal_error(I, lam, theta, y):
    A = lam * I + theta.T @ theta
    Q, R = np.linalg.qr(A)
    z = Q.T @ theta.T @ y
    w_ml = np.linalg.solve(R, z)
    y_hat = theta @ w_ml
    mse = 1 / y_hat.shape[0] * np.sum((y - y_hat) ** 2)  # equation (4)

    return mse


def __calculate_kernel(X, X_prime, d):
    # theta = 1 / cos(X @ X_prime.T / np.linalg.norm(X, axis=1) * np.linalg.norm(X_prime, axis=1))
    theta = 1 / cos(X @ X_prime.T)  # data is already on unit sphere, hence norm can be omitted in code
    kernel = 1 / (2 * np.pi * d) * (sin(theta) + (np.pi - theta) * cos(theta))
    #kernel = 1/(2*np.pi) * np.linalg.norm(X_prime, ord=2) * sin(theta) + (np.pi - theta) * cos(theta)
    return kernel

def __calculate_alpha_dual_problem(kernel, lambda_, y_train):
    I = np.diag(np.ones(y_train.shape[0]))
    alpha = -np.linalg.inv((kernel + I * lambda_)) @ (lambda_ * y_train)
    return alpha


def __calculate_alpha_representer(kernel, lambda_, y_train):
    I = np.diag(np.ones(y_train.shape[0]))
    alpha = np.linalg.inv(kernel + I * lambda_) @ y_train
    return alpha


def __calculate_theta(V, X):
    return 1 / np.sqrt(V.shape[0]) * (np.maximum(X @ V.T, 0))


def __perform_linear_regression_kernel(N, lambda_, alpha, kernel1, kernel2, y, y_t):
    y_hat_train = -1/lambda_ * kernel1 @ alpha
    y_hat_test = -1/lambda_ * kernel2 @ alpha
    mse_train = 1 / y_hat_train.shape[0] * np.sum((y - y_hat_train) ** 2)  # equation (4)
    mse_test = 1 / y_hat_test.shape[0] * np.sum((y_t - y_hat_test) ** 2)  # equation (4)

    return mse_train, mse_test


def __perform_linear_regression(N, theta, theta_t, w_ml, y, y_t):
    y_hat_train = theta @ w_ml
    y_hat_test = theta_t @ w_ml
    mse_train = 1 / y_hat_train.shape[0] * np.sum((y - y_hat_train) ** 2)  # equation (4)
    mse_test = 1 / y_hat_test.shape[0] * np.sum((y_t - y_hat_test) ** 2)  # equation (4)

    return mse_train, mse_test


def __create_feature_vectors_k(N, M):
    data_points = np.random.normal(size=(N, M))
    return data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1))


def __create_feature_vectors(k, d):
    N = 10 * k + 1
    data_points = np.random.normal(size=(N, d))
    return data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1))


def __create_data(N, d, sigma):
    X = np.random.normal(size=(N, d))
    X = X / np.linalg.norm(X, axis=1).reshape((N, 1))
    y = 1 / (1 / 4 + (X @ np.ones(d)) ** 2) + np.random.normal(np.zeros(N), sigma ** 2)

    return X, y


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    f1, f1b, f2 = task12()

    pdf.savefig(f1)
    pdf.savefig(f1b)
    pdf.savefig(f2)
    pdf.close()
