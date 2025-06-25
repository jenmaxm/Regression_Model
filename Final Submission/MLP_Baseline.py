import numpy as np
from splitting import split_and_scale_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def he_initialization(input_size, output_size):
    limit = np.sqrt(2 / input_size)  # He initialization formula
    return np.random.randn(input_size, output_size) * limit


# Mean Squared Error loss function and its derivative
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network with He-initialized weights and zero biases."""
        self.weights_input_hidden = he_initialization(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = he_initialization(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        """Perform a forward pass through the network."""
        self.input_layer = X
        
        # Compute hidden layer activation
        self.hidden_layer_sum = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer = leaky_relu(self.hidden_layer_sum)
        
        # Compute output layer activation
        self.output_layer_sum = np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output
        self.output_layer = leaky_relu(self.output_layer_sum)

        return self.output_layer

    def backward(self, X, y, learning_rate):
        """Perform backpropagation to compute gradients."""
        # Compute error (difference between predicted and actual values)
        output_error = mse_loss_derivative(y, self.output_layer)
        output_delta = output_error * leaky_relu_derivative(self.output_layer_sum)

        # Compute error propagated back to the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * leaky_relu_derivative(self.hidden_layer_sum) 
        
        # Compute gradients for updating weights and biases
        grad_weights_input_hidden = X.T.dot(hidden_delta)
        grad_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True) 
        grad_weights_hidden_output = self.hidden_layer.T.dot(output_delta)
        grad_bias_output = np.sum(output_delta, axis=0, keepdims=True)

        return grad_weights_input_hidden, grad_bias_hidden, grad_weights_hidden_output, grad_bias_output

    def train(self, X_train, y_train_scaled, X_valid, y_valid_scaled, scaler_Y, epochs, learning_rate):
        """Train the neural network using gradient descent."""
        for epoch in range(epochs):
            # Forward pass: predict output
            y_pred_scaled = self.forward(X_train)
            
            # Compute gradients for weights and biases using backpropagation
            grad_weights_input_hidden, grad_bias_hidden, grad_weights_hidden_output, grad_bias_output = self.backward(X_train, y_train_scaled, learning_rate)
            
            # Update weights and biases using the computed gradients
            self.weights_input_hidden -= grad_weights_input_hidden * learning_rate  # Adjusting weights
            self.bias_hidden -= grad_bias_hidden * learning_rate  # Adjusting biases
            self.weights_hidden_output -= grad_weights_hidden_output * learning_rate
            self.bias_output -= grad_bias_output * learning_rate  

            # Print training error every 100 epochs to track progress
            if epoch % 100 == 0:
                y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)  # Convert predictions back to original scale
                y_train_real = scaler_Y.inverse_transform(y_train_scaled)
                train_error = mse_loss(y_train_real, y_pred_real)  # Compute loss

                print(f"Epoch {epoch}, Train Error: {train_error:.6f}")
        
        return y_pred_real, y_train_real

def plot_predictions_and_correlation(nn, X_test, Y_test_scaled, scaler_Y):
    """
    Plots both the predicted values against the actual values and the correlation between them, 
    with a 45-degree line indicating the ideal case.
    
    Parameters:
    nn (NeuralNetwork): Trained neural network model.
    X_test (numpy.ndarray): Test feature set.
    Y_test_scaled (numpy.ndarray): Scaled test target values.
    scaler_Y (StandardScaler): Scaler used to inverse transform predictions.
    """
    # Generate predictions
    Y_pred_scaled = nn.forward(X_test)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)  # Convert back to original scale
    Y_test = scaler_Y.inverse_transform(Y_test_scaled)  # Inverse transform the true values
    
    # 1. Plot Actual vs Predicted (Scatter Plot)
    plt.figure(figsize=(14, 6))

    # Subplot 1: Scatter plot of actual vs predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(Y_test)), Y_test, label='Actual', marker='.', alpha=0.8, color='dodgerblue', s=60)
    plt.scatter(range(len(Y_pred)), Y_pred, label='Predicted', marker='.', alpha=0.8, color='crimson', s=60)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Target Value', fontsize=12)
    plt.title('Neural Network Predictions vs Actual Test Data', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)

    # 2. Plot Correlation: Predicted vs Actual with 45-degree line
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, Y_pred, color='dodgerblue', marker='.' ,label='Predicted vs Actual', alpha=0.8, s=60)
    plt.title('Correlation: Predicted vs Actual Values', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    
    # Plot the 45-degree line (ideal case where predicted = actual)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Ideal (y=x)')
    
    plt.text(
        min(Y_test) + (max(Y_test) - min(Y_test)) * 0.05,  # Slightly shift right for readablity
        max(Y_test) - (max(Y_test) - min(Y_test)) * 0.1,  # Slightly shift down for readablity
        f'Correlation = {correlation:.4f}', 
        fontsize=12, 
        color='black'
    )

    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_and_predict(file_path):

    # Step 1: Load and preprocess the dataset
    # The function `split_and_scale_data` handles data loading, splitting, and scaling
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_and_scale_data(file_path)

    # Step 2: Convert the data to NumPy arrays (required for matrix operations)
    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()

    # Reshape Y values to ensure they have the correct format
    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_valid = Y_valid.to_numpy().reshape(-1, 1)
    Y_test = Y_test.to_numpy().reshape(-1, 1)

    # Step 3: Standardize the target variable (Y) using StandardScaler
    # This ensures that all target values have a mean of 0 and a standard deviation of 1
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_valid_scaled = scaler_Y.transform(Y_valid)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # Step 4: Initialize the neural network
    # The input size is determined by the number of features in X_train
    input_size = X_train.shape[1]  # Number of features in the dataset
    nn = NeuralNetwork(input_size=input_size, hidden_size=12, output_size=1)

    # Step 5: Train the neural network
    # The training function updates the model’s weights using gradient descent
    nn.train(X_train, Y_train_scaled, X_valid, Y_valid_scaled, scaler_Y, epochs=40000, learning_rate=0.1)

    # Step 6: Evaluate the trained model
    # This function plots the predicted values vs. actual values to assess performance
    plot_predictions_and_correlation(nn, X_test, Y_test_scaled, scaler_Y)

# Specify the file path to your dataset
file_path = "Ouse93-96-Student.xlsx"

# Call the function to train the model and plot predictions
train_and_predict(file_path)
