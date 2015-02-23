function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
h = sigmoid(X * theta);

J = (1/m) * sum(-y .* log(h) - (ones(m,1) - y) .* log(ones(m,1) - h)) + (lambda / (2*m)) * sum(theta(2:end) .^ 2);

grad = zeros(size(theta));

grad(1,:) = (X'(1,:) * (h - y)) .* (1/m);
grad(2:end,:) = (X'(2:end,:) * (h - y)) .* (1/m) + ((lambda / m) * theta(2:end));

grad = grad(:);

end
