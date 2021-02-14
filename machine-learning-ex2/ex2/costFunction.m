function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% calculation for J

y_equals_1 = y .* log(sigmoid( X * theta ));
y_equals_0 = ( 1 - y ) .* log( 1 - (sigmoid(X*theta)));

sum_variable = sum(y_equals_0 + y_equals_1);

J = -1 * sum_variable / m;

before_sum = sigmoid( X * theta ) - y;

for i = 1:length(theta)
  grad(i) = sum(before_sum .* X(:,i)) / m;
endfor

% =============================================================

end
