from __future__ import division
import numpy as np

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   threshold   : A scalar threshold (type=float)
    # Output:
    #  Cpreds : N-by-1 ndarray of predicted labels for Xtst (dtype=np.int_)

    np.random.seed(2)

    #YourCode - binarisation of Xtrn and Xtst.
    Xtrn_binarised = np.where(Xtrn < threshold, 0, 1)
    Xtst_binarised = np.where(Xtst < threshold, 0, 1)
    Xtst_binarised_inverse = 1 - Xtst_binarised

    # number of classes and number of features
    classes = len(np.bincount(Ctrn[:, 0])) # 26 (alphabet)
    features = Xtrn.shape[1] # 784

    #YourCode - naive Bayes classification with multivariate Bernoulli distributions
    probs = np.zeros((2, classes, features))
    epsilon = np.nextafter(0, 1)
    for k in range(classes):
        # Remember Ctrn is a M-by-1 vector
        # row_indices is the list of row indices in Ctrn where we're dealing with this char
        row_indices, _ = np.where(Ctrn == k)
        
        probs[1][k] = Xtrn_binarised[row_indices].sum(axis=0) / len(row_indices)

    # add smallest possible number to prevent issues with using 0
    # this used to be in the above for loop too
    probs[1][probs[1] == 0] = epsilon
        
    # this used to be in the above for loop
    probs[0] = 1 - probs[1]

    testLen = len(Xtst)
    Cprobs = np.zeros((testLen, classes))
    expectedPreds = (testLen, 1)

    for i in range(testLen):
        Cprobs[i] = np.sum(
            np.log(
                (probs[1] ** Xtst_binarised[i])
                *
                (probs[0] ** Xtst_binarised_inverse[i])
            ),
            axis=1
        )

    return np.argmax(Cprobs, axis=1).reshape(expectedPreds)
