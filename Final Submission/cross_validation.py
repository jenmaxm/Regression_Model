import numpy as np
from splitting import split_and_scale_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

np.random.seed(42)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def he_initialisation(input_size, output_size):
    limit = np.sqrt(2 / input_size)  # He initialization formula
    return np.random.randn(input_size, output_size) * limit

def rmse_loss(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def rmse_loss_derivative(y_true, y_pred):
    return (y_pred - y_true) / (y_true.size * np.sqrt(((y_true - y_pred) ** 2).mean()))


class NeuralNetwork(BaseEstimator, RegressorMixin):  # Extend scikit-learn classes for compatibility
    def __init__(self, input_size, hidden_size=16, output_size=1, momentum=0.85, epochs=500, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.momentum = momentum
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scaler_Y = None  # Placeholder for scaler

        # Initialize weights and biases
        self.weights_input_hidden = he_initialisation(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = he_initialisation(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        
        # Initialize velocity for momentum
        self.velocity_wih = np.zeros_like(self.weights_input_hidden)
        self.velocity_bh = np.zeros_like(self.bias_hidden)
        self.velocity_who = np.zeros_like(self.weights_hidden_output)
        self.velocity_bo = np.zeros_like(self.bias_output)

    def forward(self, X):
        self.input_layer = X
        self.hidden_layer_sum = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer = leaky_relu(self.hidden_layer_sum)
        self.output_layer_sum = np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output
        self.output_layer = leaky_relu(self.output_layer_sum)
        return self.output_layer

    def backward(self, X, y, learning_rate):
        lambda_wd = 0.0005  # Weight decay coefficient
        
        output_error = rmse_loss_derivative(y, self.output_layer)
        output_delta = output_error * leaky_relu_derivative(self.output_layer_sum)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * leaky_relu_derivative(self.hidden_layer_sum) 

        # wih = weight_input_hidden , bh = bias_hidden , who = weight_output_hidden , bo = bias_output 
        grad_wih = X.T.dot(hidden_delta)
        grad_bh = np.sum(hidden_delta, axis=0, keepdims=True) 
        grad_who = self.hidden_layer.T.dot(output_delta)
        grad_bo = np.sum(output_delta, axis=0, keepdims=True)

        # Update with momentum and weight decay
        self.velocity_wih = self.momentum * self.velocity_wih + (1 - self.momentum) * (grad_wih + lambda_wd * self.weights_input_hidden)
        self.velocity_who = self.momentum * self.velocity_who + (1 - self.momentum) * (grad_who + lambda_wd * self.weights_hidden_output)
        self.velocity_bo = self.momentum * self.velocity_bo + (1 - self.momentum) * grad_bo
        self.velocity_bh = self.momentum * self.velocity_bh + (1 - self.momentum) * grad_bh
        
        self.weights_input_hidden -= learning_rate * self.velocity_wih
        self.weights_hidden_output -= learning_rate * self.velocity_who
        self.bias_output -= learning_rate * self.velocity_bo
        self.bias_hidden -= learning_rate * self.velocity_bh

    def fit(self, X_train, y_train):
        self.scaler_Y = StandardScaler()
        y_train_scaled = self.scaler_Y.fit_transform(y_train.reshape(-1, 1))

        num_samples = X_train.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_scaled[indices]

            for i in range(0, num_samples, 32):  
                X_batch = X_train_shuffled[i:i+32]
                y_batch = y_train_shuffled[i:i+32]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, self.learning_rate)

        return self

    def predict(self, X):
        y_pred_scaled = self.forward(X)
        return self.scaler_Y.inverse_transform(y_pred_scaled)


def cross_validate_nn(X, Y, k=5):
    """
    Performs k-fold cross-validation on the NeuralNetwork model.
    
    Parameters:
    X (numpy.ndarray): Features dataset.
    Y (numpy.ndarray): Target dataset.
    k (int): Number of folds for cross-validation.

    Returns:
    None: Prints the cross-validation scores.
    """
    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=16, output_size=1, epochs=500, learning_rate=0.1)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    scores = cross_val_score(nn, X, Y, cv=kf, scoring='r2')  # You can also use 'neg_mean_squared_error' for RMSE
    print(f"Cross-validation R² Scores: {scores}")
    print(f"Average R²: {np.mean(scores):.6f} ± {np.std(scores):.6f}")


def train_and_evaluate(file_path):
    # Get the train, validation, and test sets
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_and_scale_data(file_path)

    # Convert to NumPy arrays
    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()
    Y_train, Y_valid, Y_test = Y_train.to_numpy(), Y_valid.to_numpy(), Y_test.to_numpy()

    # Initialize and train the model
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=16, output_size=1, epochs=500, learning_rate=0.1)
    nn.fit(X_train, Y_train)

    # Perform cross-validation
    cross_validate_nn(X_train, Y_train, k=5)
    
    # Evaluate on the test set
    Y_test_pred = nn.predict(X_test)
    test_rmse = rmse_loss(Y_test, Y_test_pred)
    print(f"Test RMSE: {test_rmse:.6f}")

# Specify the file path to your dataset
file_path = "Ouse93-96-Student.xlsx"

# Train, cross-validate, and evaluate on the test data
train_and_evaluate(file_path)



# Specify the file path to your dataset
file_path = "Ouse93-96-Student.xlsx"

# Train and cross-validate the model
train_and_cross_validate(file_path)
