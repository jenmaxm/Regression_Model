import numpy as np
from splitting import split_and_scale_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

# 5,000 to even out learning, 0.9 correlation

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

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, momentum=0.9):
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
        
        self.momentum = momentum

    def forward(self, X):
        self.input_layer = X
        self.hidden_layer_sum = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer = leaky_relu(self.hidden_layer_sum)
        self.output_layer_sum = np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output
        self.output_layer = leaky_relu(self.output_layer_sum)
        return self.output_layer

    def backward(self, X, y, learning_rate):
        output_error = rmse_loss_derivative(y, self.output_layer)
        output_delta = output_error * leaky_relu_derivative(self.output_layer_sum)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * leaky_relu_derivative(self.hidden_layer_sum) 

        # wih = weight_input_hidden , bh = bias_hidden , who = weight_output_hidden , bo = bias_output 
        grad_wih = X.T.dot(hidden_delta)
        grad_bh = np.sum(hidden_delta, axis=0, keepdims=True) 
        grad_who = self.hidden_layer.T.dot(output_delta)
        grad_bo = np.sum(output_delta, axis=0, keepdims=True)
        
        # Update velocities with momentum
        self.velocity_wih = self.momentum * self.velocity_wih + (1 - self.momentum) * grad_wih
        self.velocity_bh = self.momentum * self.velocity_bh + (1 - self.momentum) * grad_bh
        self.velocity_who = self.momentum * self.velocity_who + (1 - self.momentum) * grad_who
        self.velocity_bo = self.momentum * self.velocity_bo + (1 - self.momentum) * grad_bo
        
        # Update weights and biases using velocity
        self.weights_input_hidden -= learning_rate * self.velocity_wih
        self.bias_hidden -= learning_rate * self.velocity_bh
        self.weights_hidden_output -= learning_rate * self.velocity_who
        self.bias_output -= learning_rate * self.velocity_bo

    def train(self, X_train, y_train_scaled, X_valid, y_valid_scaled, scaler_Y, epochs, learning_rate, batch_size=64):
        prev_loss = float('inf')
        loss_history = []
        val_loss_history = []  # History of validation losses
        num_samples = X_train.shape[0]  # Get the number of training samples

        for epoch in range(epochs):  # Loop over the number of epochs
            indices = np.random.permutation(num_samples)  # Shuffle indices to randomize training order
            X_train_shuffled = X_train[indices]  # Shuffle the training data using the randomized indices
            y_train_shuffled = y_train_scaled[indices]  # Shuffle the corresponding target values

            for i in range(0, num_samples, batch_size):  # Iterate over mini-batches
                X_batch = X_train_shuffled[i:i+batch_size]  # Extract a batch of input features
                y_batch = y_train_shuffled[i:i+batch_size]  # Extract the corresponding batch of target values

                self.forward(X_batch)  # Perform the forward pass to compute predictions
                self.backward(X_batch, y_batch, learning_rate)  # Compute gradients and update weights

            if epoch % 100 == 0:
                y_pred_real = scaler_Y.inverse_transform(self.forward(X_train))
                y_train_real = scaler_Y.inverse_transform(y_train_scaled)
                train_error = rmse_loss(y_train_real, y_pred_real)
                
                # Validation loss calculation
                y_valid_pred_real = scaler_Y.inverse_transform(self.forward(X_valid))
                y_valid_real = scaler_Y.inverse_transform(y_valid_scaled)
                val_error = rmse_loss(y_valid_real, y_valid_pred_real)
                
                loss_history.append(train_error)
                val_loss_history.append(val_error)

                print(f"Epoch {epoch}, Train Error: {train_error:.6f}, Validation Error: {val_error:.6f}, LR: {learning_rate:.6f}")

                if train_error < prev_loss:
                    learning_rate *= 1.05  
                else:
                    learning_rate *= 0.85

                prev_loss = train_error

        # Plotting both training and validation loss history
        plt.figure(figsize=(8, 5))
        plt.plot(range(0, len(loss_history) * 100, 100), loss_history, linestyle='-', color='b', label='Training Loss')
        plt.plot(range(0, len(val_loss_history) * 100, 100), val_loss_history, linestyle='-', color='orange', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE Loss')
        plt.title('Training and Validation Loss vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

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
    
    # Display the correlation coefficient (Pearson's r)
    correlation = np.corrcoef(Y_test.flatten(), Y_pred.flatten())[0, 1]
    plt.text(min(Y_test), max(Y_pred), f'Correlation = {correlation:.4f}', fontsize=12, color='black')
    
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_and_predict(file_path):
    # Get the train, validation, and test sets
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_and_scale_data(file_path)

    # Convert to NumPy arrays
    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()
    Y_train, Y_valid, Y_test = Y_train.to_numpy().reshape(-1, 1), Y_valid.to_numpy().reshape(-1, 1), Y_test.to_numpy().reshape(-1, 1)

    # Standardize the target variable (Y_train, Y_valid, Y_test)
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_valid_scaled = scaler_Y.transform(Y_valid)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # Initialize the neural network with the correct input size
    input_size = X_train.shape[1]  # Number of features
    nn = NeuralNetwork(input_size=input_size, hidden_size=8, output_size=1, momentum=0.75)

    # Train the network with validation loss tracking
    nn.train(X_train, Y_train_scaled, X_valid, Y_valid_scaled, scaler_Y, epochs= 7500, learning_rate=0.05)

    # Plot the predictions
    plot_predictions_and_correlation(nn, X_test, Y_test_scaled, scaler_Y)

# Specify the file path to your dataset
file_path = "Ouse93-96-Student.xlsx"

# Call the function to train the model and plot predictions
train_and_predict(file_path)
