function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

lowest_e = 10000000;

for _C = C_vec
  for _sigma = sigma_vec
    fprintf("\n Trying with C = %f, sigma = %f \n", _C, _sigma);
    model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));
    predictions = svmPredict(model, Xval);
    e = mean(double(predictions ~= yval))
    if (e < lowest_e)
      best_C = _C
      best_sigma = _sigma
      lowest_e = e
    endif
  end
end

C = best_C
sigma = best_sigma

% =========================================================================

end
