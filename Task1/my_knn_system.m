%
% Template for my_knn_system.m
%
% load the data set
%   NB: replace <UUN> with your actual UUN.
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/<UUN>/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare measuring time

% Run K-NN classification
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb);

%YourCode - Measure the time taken, and display it.

%YourCode - Get confusion matrix and accuracy for each k in kb.

%YourCode - Save each confusion matrix.

%YourCode - Display the required information - k, N, Nerrs, acc for
%           each element of kb.





  
