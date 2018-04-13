#! /usr/bin/env python
# A sample template for my_bnb_system.py

import numpy as np
import scipy.io as sio
import time
from my_bnb_classify import *
from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1620208/data.mat";
data = sio.loadmat(filename);

# Feature vectors: Convert uint8 to double   (but do not divide by 255)
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_)
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_)
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time
startTime = time.clock()

# Run classification
threshold = 1.0
print "Threshold:\t%f" % threshold
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)

#YourCode - Measure the user time taken, and display it.
timeTaken = time.clock() - startTime
print "Time taken:\t%f seconds" % timeTaken

#YourCode - Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

#YourCode - Save the confusion matrix as "Task2/cm.mat".
sio.savemat('cm.mat', {'cm': CM}, oned_as='row')

#YourCode - Display the required information - N, Nerrs, acc.
print "\nN\t\tNerrs\t\tacc"
print "%d\t\t%d\t\t%f" % (
    len(Ctst),
    len(Ctst) * (1-acc),
    acc
)

# t = np.zeros((250, 1))
# for threshold in range(2, 250, 20):
#     print "Threshold:\t%f" % threshold
#     Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)

#     #YourCode - Get a confusion matrix and accuracy
#     _, acc = my_confusion(Ctst, Cpreds)

#     #YourCode - Save the confusion matrix as "Task2/cm.mat".
#     t[threshold] = acc

# sio.savemat('plot.mat', {'thresholds': t}, oned_as='row')