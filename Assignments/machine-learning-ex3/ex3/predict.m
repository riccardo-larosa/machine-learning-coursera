function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add x_0 = 1
X = [ones(m,1) X];

A_2 = sigmoid(X * Theta1');

% add x_1 = 1
A_2 = [ones(size(A_2),1) A_2];

A_3 = sigmoid(A_2 * Theta2');

% calculate max val for each row and find the column value for that max
[max_val, max_val_idx] = max(A_3, [], 2);

% return the column values -> labels
p = max_val_idx;




% =========================================================================


end
