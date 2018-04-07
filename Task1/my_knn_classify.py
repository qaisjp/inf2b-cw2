import math
import numpy as np

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

def MySqDist(X, Y):
    print "MySqDist1"
    XX = (X ** 2).sum(axis=1)[:, np.newaxis]
    YY = (Y ** 2).sum(axis=1)[:, np.newaxis]

    print ""
    print(X)
    print ""
    print(Y.T)

    return XX - 2 * (X*Y.T) + YY


def MySqDist_first(U, v):
    """
    U = MxN
    v = 1xN vector
    Return: 1xM row vector of square distances
    """
    s = (U - v) ** 2
    return np.sqrt(s[:, 0] + s[:, 1])

