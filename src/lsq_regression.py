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
    plt.suptitle('Task 1 - Regularized Least Squares and Double Descent Phenomenon', fontsize=16)
    for a in ax1.reshape(-1):
        a.set_ylim([0,40])
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

    # 1.5: Create random features
    # num_random_features = 50
    # V, _ = __create_data(num_random_features, d, sigma)

    train_losses = np.zeros((3, 61, 5))
    test_losses = np.zeros((3, 61, 5))
    # train_losses_k = np.zeros((3, 61, 5))
    # test_losses_k = np.zeros((3, 61, 5))
    #kernel_train = __calculate_kernel(X=X, X_prime=X)

    for l in range(3):  # loop over lambdas
        print(str.format("Calculating lambda {0}", lams[l]))
        lambda_ = lams[l]  # lambda value to reproduce double desecent phenomenon
        for v in range(0, 61):  # loop over feature vectors
            for r in range(5):  # experiments
                V = __create_feature_vectors(v, d)
                # V = feature_vectors[v]
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


    # task 2: Dual representation
    # M = [10, 200, 800]
    # kernels = list()
    # for m in M:
    #     V = __create_feature_vectors(m, d)
    #     kernels.append(__calculate_kernel(X=X, X_prime=V))


    """ End of your code
    """

    features = range(61)
    train_mean = np.mean(train_losses, axis=2)
    test_mean = np.mean(test_losses, axis=2)
    train_std = np.std(train_losses, axis=2)
    test_std = np.std(test_losses, axis=2)
    # train_mean_k = np.mean(train_losses_k, axis=2)
    # test_mean_k = np.mean(test_losses_k, axis=2)
    for lam_idx, a in enumerate(ax1.reshape(-1)):
        a.plot(features, train_mean[lam_idx, :], label='train')
        a.fill_between(features, train_mean[lam_idx, :] - train_std[lam_idx, :], train_mean[lam_idx, :] + train_std[lam_idx, :], alpha=0.5)

        a.plot(features, test_mean[lam_idx, :], label='test')
        a.fill_between(features, test_mean[lam_idx, :] - test_std[lam_idx, :], test_mean[lam_idx, :] + test_std[lam_idx, :], alpha=0.5)

        # a.plot(features, train_mean_k[lam_idx, :], label='train_k')
        # a.plot(features, test_mean_k[lam_idx, :], label='test_k')
        # a.plot(features, train_losses_k[lam_idx], label='train_k')
        # a.plot(features, test_losses_k[lam_idx], label='test_k')
        a.legend()
        a.set_title(r'$\lambda=$' + str(lams[lam_idx]))

    for m_idx, a in enumerate(ax2.reshape(-1)):
        a.legend()
        a.set_title('#Features M=%i, MAE=%f' % (m_array[m_idx], (mae_array[m_idx])))
        # a.plot(range(len(kernels[m_idx][9, :])), kernels[m_idx][9, :])

    return fig1, fig2

def __calculate_kernel(X, X_prime):
    theta =  1 / cos( X_prime.T @ X / np.linalg.norm(X, ord=2)  * np.linalg.norm(X_prime, ord=2) )
    kernel = 1/(2*np.pi) * np.linalg.norm(X_prime, ord=2) * sin(theta) + (np.pi - theta) * cos(theta)
    return kernel

def __calculate_alpha(kernel, lambda_, y_train):
    I = np.diag(np.ones(y_train.shape[0]))
    alpha = -(kernel + I * lambda_) @ (lambda_ * y_train)
    return alpha

def __calculate_theta(V, X):
    # theta = np.zeros((X.shape[0], V.shape[0], X.shape[1])) #  N x M x d -> example: 200 x 11 x 5  -> 200 x 11 x 1
    # for r in range(X.shape[1]):
    #     v = V[:, r].reshape(V.shape[0], 1)
    #     x = X[:, r].reshape(X.shape[0], 1)
    #     theta[:, :, r] = (x @ v.T) 

    # return  1 / np.sqrt(V.shape[0]) * (np.maximum(np.amax(theta, axis=2), 0)) # loog for max in each dimension d = 5
    return 1 / np.sqrt(V.shape[0]) * (np.maximum(X @ V.T, 0)) # loog for max in each dimension d = 5


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

def __create_feature_vectors(k, d):
    N = 10 * k + 1
    data_points = np.random.normal(size=(N, d)) #np.random.rand <- uniform
    return data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1))


def __create_data(N, d, sigma):
    X = np.random.normal(size=(N, d)) 
    X = X / np.linalg.norm(X, axis=1).reshape((N, 1))
    y = 1 / ((1 / 4 + (X @ np.ones(d)) ** 2)) + np.random.normal(np.zeros((N)), sigma ** 2)

    return X, y


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    f1, f2 = task12()

    pdf.savefig(f1)
    pdf.savefig(f2)
    pdf.close()
