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

    # print("Running MySqDist")
    DI = MySqDist(Xtst, Xtrn)
    # print("Done with MySqDist")

    maxK = max(Ks) # -1 because if we wanted 1 nearest neighbours, that would be index 0

    idx = DI.argpartition(maxK, axis=1)[:, :maxK]

    # I initially subpartitioned instead of sorting the resultant
    # matrix. I guess max 20 columns isn't that much work anyway.
    idx.sort(axis=1)

    rows, _ = idx.shape
    columns = len(Ks)

    modes = np.zeros((rows, columns), dtype=int)
    for i in range(len(Ks)):
        k = Ks[i]

        # This subpartitioning code doesn't work.
        # # We subpartition because it's still cheaper than sorting
        # # TODO: is subpartitioning lenK times better than sorting once?
        # subpartition = idx
        # print (k, idx.shape)
        # if k < idx.shape[1]: # we can't subpartition on the final size
        #     subpartition = idx.argpartition(k, axis=1)[:,:k]

        # get the first k items
        partition = idx[:, :k]

        # label the partition
        labelled = Ctrn[:, 0][partition]

        # get mode of each labelled row
        columnMode = stats.mode(labelled, axis=1)[0][:, 0]

        # set column i to the column vector "columnMode"
        modes[:, i] = columnMode

    # print(modes)

    Cpreds = modes
    return Cpreds

# consumes memory, kept for comedic value
# def MySqDist(X, Y):
#     print "MySqDist"
#     XX = np.dot(X, X.T)
#     YY = np.dot(Y, Y.T)
#     print(len(Y), len(YY ))
#     return XX - 2 * np.dot(X, Y.T) + YY

def MySqDist(X, Y):
    XX = (X ** 2).sum(axis=1)[:, np.newaxis]
    YY = (Y ** 2).sum(axis=1)[np.newaxis, :]

    return XX - 2 * X.dot(Y.T) + YY


# def MySqDist_first(U, v):
#     """
#     U = MxN
#     v = 1xN vector
#     Return: 1xM row vector of square distances
#     """
#     s = (U - v) ** 2
#     return np.sqrt(s[:, 0] + s[:, 1])

