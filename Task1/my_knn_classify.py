import math
import numpy as np
from scipy import stats

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # type: (np.float_, np.int_, np.float_, float[]) -> np.int_

    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)

    np.random.seed(2)
    print("Running dot product")

    DI = MySqDist(Xtrn, Xtst)

    maxK = max(Ks) # -1 because if we wanted 1 nearest neighbours, that would be index 0

    idx = DI.argpartition(maxK, axis=1)[:, :maxK]

    rows, _ = idx.shape
    columns = len(Ks)

    # modes = np.zeros((rows, columns))
    # for i in range(len(Ks)):
    #     k = Ks[i]
    #
    #     # set column i to the mode of each row (mode of the least k columns in idx)
    #     modes[:, i] = stats.mode(idx.argpartition(k, axis=1)[:,:k], axis=1)[0][:, 0]

    # print(modes)

    
    # np.dot(Xtrn, Xtrn.T)
    print "Done!2"
    Cpreds = None
    
    return Cpreds

# consumes memory, kept for comedic value
# def MySqDist(X, Y):
#     print "MySqDist"
#     XX = np.dot(X, X.T)
#     YY = np.dot(Y, Y.T)
#     print(len(Y), len(YY ))
#     return XX - 2 * np.dot(X, Y.T) + YY

def MySqDist(Y, X):
    XX = (X ** 2).sum(axis=1)[:, np.newaxis]
    YY = (Y ** 2).sum(axis=1)[np.newaxis, :]

    return XX - 2 * X.dot(Y.T) + YY


def MySqDist_first(U, v):
    """
    U = MxN
    v = 1xN vector
    Return: 1xM row vector of square distances
    """
    s = (U - v) ** 2
    return np.sqrt(s[:, 0] + s[:, 1])

