#! /usr/bin/env python
# A sample template for my_gaussian_system.py

import numpy as np
import scipy.io as sio
import time
from my_gaussian_classify import *
from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "../data.mat";
data = sio.loadmat(filename);

# Feature vectors: Convert uint8 to double, and divide by 255
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time
startTime = time.clock()

# Run classification
epsilon = 0.01
(Cpreds, Ms, Covs) = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)

#YourCode - Measure the user time taken, and display it.
timeTaken = time.clock() - startTime
print "Time taken:\t%f seconds" % timeTaken

#YourCode - Get a confusion matrix and accuracy
CM, acc = my_confusion(Ctst, Cpreds)

#YourCode - Save the confusion matrix as "Task3/cm.mat".
sio.savemat('cm.mat', {'cm': CM}, oned_as='row')

#YourCode - Save the mean vector and covariance matrix for class 26,
#           i.e. save Mu(:,25) and Cov(:,:,25) as "Task3/m26.mat" and
#           "Task3/cov26.mat", respectively.

#YourCode - Display the required information - N, Nerrs, acc.
print "\nN\t\tNerrs\t\tacc"
print "%d\t\t%d\t\t%f" % (
    len(Ctst),
    len(Ctst) * (1-acc),
    acc
)