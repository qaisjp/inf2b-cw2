function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
% Input:
%   Xtrn : M-by-D training data matrix (of floating-point numbers
%          in doubleprecision format, which is the default in Matlab)
%          of training data, where M is the number of training samples,
%          and D is the the number of elements in a sample.
%
%          Note that each sample is represented as a row vector rather
%          than a column vector
%
%   Ctrn : M-by-1 label vector for Xtrn.
%
%          Ctrn(i) is the class number of Xtrn(i, :)
%
%   Xtst : N-by-D test data matrix, where N is the number of test samples.
%
%   Ks   : L-by-1 vector of the numbers of nearest neighbours in Xtrn
%
% Output:
%  Cpreds : N-by-L matrix of predicted labels for Xtst.
%
%           Cpreds(i, j) is the predicted class for Xtst(i, :) with the
%           number of nearest neighbours being Ks(j).
    

end
