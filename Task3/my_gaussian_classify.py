from __future__ import division
import numpy as np

def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   epsilon   : A scalar parameter for regularisation (type=float)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=int_)
    #  Ms    : D-by-K ndarray of mean vectors (dtype=np.float_)
    #  Covs  : D-by-D-by-K ndarray of covariance matrices (dtype=np.float_)

    #YourCode - Bayes classification with multivariate Gaussian distributions.

    classes = len(np.bincount(Ctrn[:, 0])) # 26 (alphabet)
    features = Xtrn.shape[1] # 784
    testLen = len(Xtst)

    Ms = np.zeros((features, classes))
    Covs = np.zeros((features, features, classes))
    Cprobs = np.zeros((testLen, classes))
    expectedPreds = (testLen, 1)
    Ms_inverse = Ms.T
    for k in range(classes):
        # Remember Ctrn is a M-by-1 vector
        # row_indices is the list of row indices in Ctrn where we're dealing with this char
        row_indices, _ = np.where(Ctrn == k)

        Ms[:,k] = Xtrn[row_indices].sum(axis=0) / len(row_indices)

        Mn = np.tile(Ms[:, k][np.newaxis,:], (len(Xtrn[row_indices]), 1))

        minused = Xtrn[row_indices]-Mn
        cov = minused.T.dot(minused) / len(row_indices)

        np.fill_diagonal(cov, np.diagonal(cov) + epsilon)

        Covs[:,:,k] = cov

        Covs_inv = np.linalg.inv(cov)
        Covs_logdet = np.linalg.slogdet(cov)[1]/2

        mean = Xtst - Ms_inverse[k,:]
        for i in range(testLen):
            Cprobs[i,k] = -.5*mean[i].dot(Covs_inv).dot(mean[i].T)-Covs_logdet
    
    Cpreds = np.argmax(Cprobs, axis=1).reshape(expectedPreds)

    return (Cpreds, Ms, Covs)
