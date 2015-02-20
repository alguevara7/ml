function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
m = length(y); % number of training examples
% You need to return the following variables correctly 
h = sigmoid(X * theta);

J = (1/m) * sum(-y .* log(h) - (ones(m,1) - y) .* log(ones(m,1) - h));
% =============================================================
grad = (X' * (h - y)) .* (1/m);

end
