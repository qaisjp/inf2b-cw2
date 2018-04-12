#! /usr/bin/env python
# A sample template for my_knn_system.py

import numpy as np
import scipy.io as sio
from my_knn_classify import *
# import my_knn_classify as knnc

from my_confusion import *
import time

Xtrn = None
Xtst = None
Ctrn = None
Ctst = None
Cpreds = None

def load_dataset():
    global Xtrn, Xtst, Ctrn, Ctst

    # Load the data set
    #   NB: replace <UUN> with your actual UUN.
    # filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1620208/data.mat";
    filename = "../data.mat";
    data = sio.loadmat(filename);

    # Feature vectors: Convert uint8 to double, and divide by 255.
    Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float32) /255.0
    Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float32) /255.0
    # Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
    Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
    Ctrn = Ctrn.reshape((Ctrn.size, 1))
    Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
    Ctst = Ctst.reshape((Ctst.size, 1))

load_dataset()

def run():
    global Cpreds

    #YourCode - Prepare measuring time
    startTime = time.clock()

    # Run K-NN classification
    kb = [1,3,5,10,20];
    Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)

    #YourCode - Measure the user time taken, and display it.
    timeTaken = time.clock() - startTime
    print "Time taken: %d seconds" % timeTaken

    print "k\t\tN\t\tNerrs\t\tacc"
    for i in range(len(kb)):
        k = kb[i]

        #YourCode - Get confusion matrix and accuracy for each k in kb.
        CM, acc = my_confusion(Ctst[:,0], Cpreds[:,i])

        #YourCode - Save each confusion matrix.
        filename = 'cm%d' % k
        sio.savemat(filename, {filename: CM}, oned_as="row")

        #YourCode - Display the required information - k, N, Nerrs, acc for each element of kb
        print "%d\t\t%d\t\t%d\t\t%f" % (
            k,
            len(Ctst),
            len(Ctst) * (1-acc),
            acc
        )
run()