""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
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
        # a.set_ylim([0,40])
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

    for l in range(3):  # loop over lambdas
        for v in range(0, 61):  # loop over feature vectors
            for r in range(5):  # experiments
                V = __create_feature_vectors(v, d)
                # V = feature_vectors[v]
                I = np.diag(np.ones(V.shape[0]))
                lambda_ = lams[l]  # lambda value to reproduce double desecent phenomenon

                # 1.6: Implement w* and calculate MSE
                theta = __calculate_theta(V, X)
                theta_t = __calculate_theta(V, X_t)
                # R = lambda_ * I + theta.T @ theta
                # z = theta.T @ y
                A = lambda_ * I + theta.T @ theta
                Q, R = np.linalg.qr(A)
                z = Q.T @ theta.T @ y
                # w_analytical = np.linalg.pinv(theta) @ y

                w_analytical_1 = np.linalg.inv(A) @ theta.T @ y
                w_analytical_2 = np.linalg.pinv(A) @ theta.T @ y

                # 1.7: Reproduce double descent phenomenon
                w_ml = np.linalg.solve(R, z)  # computation via QR decomposition
                # w_ml = w_analytical_2
                mse_train, mse_test = __perform_linear_regression(N, theta, theta_t, w_ml, y, y_t)

                # maybe still some scaling problems
                train_losses[l, v, r] = mse_train  # average mse
                test_losses[l, v, r] = mse_test

    # task 2: Dual representation



    """ End of your code
    """

    features = range(61)
    train_mean = np.mean(train_losses, axis=2)
    test_mean = np.mean(test_losses, axis=2)
    for lam_idx, a in enumerate(ax1.reshape(-1)):
        a.plot(features, train_mean[lam_idx, :], label='train')
        a.plot(features, test_mean[lam_idx, :], label='test')
        a.legend()
        a.set_title(r'$\lambda=$' + str(lams[lam_idx]))

    for m_idx, a in enumerate(ax2.reshape(-1)):
        a.legend()
        a.set_title('#Features M=%i, MAE=%f' % (m_array[m_idx], (mae_array[m_idx])))

    return fig1, fig2


def __calculate_theta(V, X):
    return 1 / np.sqrt(V.shape[0]) * (np.maximum(X @ V.T, 0))


def __perform_linear_regression(N, theta, theta_t, w_ml, y, y_t):
    y_hat_train = theta @ w_ml
    y_hat_test = theta_t @ w_ml
    mse_train = 1 / N * np.sum((y - y_hat_train) ** 2)  # equation (4)
    mse_test = 1 / N * np.sum((y_t - y_hat_test) ** 2)  # equation (4)

    return mse_train, mse_test


def __create_feature_vectors(k, d):
    feature_vectors = []
    N = 10 * k + 1
    data_points = np.random.rand(N, d)
    # feature_vectors.append(data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1)))

    return data_points


def __create_data(N, d, sigma):
    # data_points = np.random.uniform(size=(N, d))
    X = np.random.uniform(size=(N, d))  # data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1))
    y = 1 / ((1 / 4 + (X @ np.ones(d)) ** 2)) + np.random.normal(0, sigma ** 2)

    return X, y


if __name__ == '__main__':
    try:
        pdf = PdfPages('figures.pdf')
        f1, f2 = task12()

        pdf.savefig(f1)
        pdf.savefig(f2)
        pdf.close()
    except:
        pass
