%
% Template for my_bnb_system.m
%
% load the data set
%   NB: replace <UUN> with your actual UUN.
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/<UUN>/data.mat');

% Feature vectors: Convert uint8 data to double (but do not divide by 255)
Xtrn = double(dataset.train.images);
Xtst = double(dataset.test.images);
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare to measure time

% Run classification
threshold = 1;
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);

%YourCode - Measure the time taken, and display it.

%YourCode - Get a confusion matrix and accuracy

%YourCode - Save the confusion matrix as "Task2/cm.mat".

%YourCode - Display the required information - N, Nerrs, acc.





  
