function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
m = size(X, 1); % Number of training examples
% You need to return the following variables correctly
p = zeros(m, 1);

p = arrayfun( @(s) s > 0.5, sigmoid(X * theta) );

end
