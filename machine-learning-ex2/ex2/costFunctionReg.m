function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


y_equals_1 = y .* log(sigmoid( X * theta ));
y_equals_0 = ( 1 - y ) .* log( 1 - (sigmoid(X*theta)));

sum_variable = sum(y_equals_0 + y_equals_1);

first_part_J = -1 * sum_variable / m;

squared_theta = theta(2:length(theta)).^2;

second_part_J = lambda * sum(squared_theta) / (2 * m); 

J = first_part_J + second_part_J;

before_sum = sigmoid( X * theta ) - y;

grad(1) = sum(before_sum .* X(:,1)) / m;
for i = 2:length(theta)
  first_part_grad = sum(before_sum .* X(:,i)) / m;
  second_part_grad = lambda * theta(i) /m;
  grad(i) = first_part_grad + second_part_grad;
endfor



% =============================================================

end
