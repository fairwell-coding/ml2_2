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
        a.set_ylim([0,40])
        a.set_ylabel('error')
        a.set_xlabel('number of random features')

    fig2, ax2 = plt.subplots(1,3,figsize=(15, 5))
    plt.suptitle('Task 2 - Dual Representation with Kernel', fontsize=16)

    lams = [1e-8,1e-5,1e-3] # use this for subtask 1
    m_array = [10,200,800] # use this for subtask 2
    mae_array = 1e3*np.ones((3)) # use this for subtask 2 (MAE = mean absolute error)

    """ Start of your code 
    """

    # parameters
    d = 5
    sigma = 2

    # 4. Create training data
    num_training_samples = 200
    X, y = __create_data(num_training_samples, d, sigma)

    # 4. Create test data
    X_t, y_t = __create_data(50, d, sigma)

    # 5. Create random features
    num_random_features = 50
    V, _ = __create_data(num_random_features, d, sigma)

    # 6. Implement w*
    theta = 1 / np.sqrt(num_random_features) * (np.maximum(X @ V.T, 0))
    I = np.diag(np.ones(num_random_features))
    lambda_ = 1e-8  # lambda value to reproduce double desecent phenomenon

    R = lambda_ * I + theta.T @ theta
    z = theta.T @ y
    w = np.linalg.inv(R) @ z  # analytical computation

    w_ml = np.linalg.solve(R, z)  # computation via QR decomposition

    """ End of your code
    """
    
    for lam_idx, a in enumerate(ax1.reshape(-1)):
        a.legend()
        a.set_title(r'$\lambda=$'+str(lams[lam_idx]))

    for m_idx, a in enumerate(ax2.reshape(-1)):
        a.legend()
        a.set_title('#Features M=%i, MAE=%f' %(m_array[m_idx],(mae_array[m_idx])))

    return fig1, fig2


def __create_data(N, d, sigma):
    data_points = np.random.rand(N, d)
    X = data_points / np.linalg.norm(data_points, axis=1).reshape((N, 1))
    y = np.reciprocal((1 / 4 + (X @ np.ones(d)) ** 2)) + np.random.normal(0, sigma ** 2)

    return X, y


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    f1, f2 = task12()

    pdf.savefig(f1)
    pdf.savefig(f2)
    pdf.close()
