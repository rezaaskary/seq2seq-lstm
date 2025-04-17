import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Custom dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, input_window, output_window):
        """
        Args:
            x_data: Input time series data of shape [total_time_steps, n_features]
            y_data: Target time series data of shape [total_time_steps, m_features]
            input_window: Length of input sequence
            output_window: Length of output sequence
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.x_data) - (self.input_window + self.output_window) + 1

    def __getitem__(self, idx):
        # Get input sequence
        x_seq = self.x_data[idx:idx + self.input_window]
        # Get target sequence
        y_seq = self.y_data[idx + self.input_window:idx + self.input_window + self.output_window]
        return x_seq, y_seq


# Encoder LSTM
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        outputs, (hidden, cell) = self.lstm(x)
        # outputs shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]
        # cell shape: [num_layers, batch_size, hidden_dim]
        return outputs, hidden, cell


# Decoder LSTM
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # The decoder input is the output dimension (for autoregressive prediction)
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x shape: [batch_size, 1, output_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]
        # cell shape: [num_layers, batch_size, hidden_dim]

        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # output shape: [batch_size, 1, hidden_dim]

        prediction = self.fc(output.squeeze(1))
        # prediction shape: [batch_size, output_dim]

        return prediction, hidden, cell


# Sequence-to-Sequence Model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, input_dim, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Project the last input value to the output dimension for initial decoder input
        self.project = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt_len):
        # src shape: [batch_size, src_seq_len, input_dim]
        batch_size = src.shape[0]

        # Get the encoder outputs and states
        _, hidden, cell = self.encoder(src)

        # Initialize the first input to decoder by projecting last input to output dimension
        last_input = src[:, -1, :]  # [batch_size, input_dim]
        decoder_input = self.project(last_input).unsqueeze(1)  # [batch_size, 1, output_dim]

        # Store all decoder outputs
        outputs = []

        # Autoregressive decoding
        for t in range(tgt_len):
            # Pass through decoder
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # Store output
            outputs.append(output)

            # Create a new input for next step using current prediction
            decoder_input = output.unsqueeze(1)  # [batch_size, 1, output_dim]

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        # outputs shape: [batch_size, tgt_len, output_dim]

        return outputs


# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt.size(1))

        # Calculate loss
        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            output = model(src, tgt.size(1))

            # Calculate loss
            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# Function to run the full pipeline
def run_seq2seq_model(x_data, y_data, input_window, output_window, n_features, m_features,
                      hidden_dim=64, num_layers=2, batch_size=32, learning_rate=0.001,
                      epochs=100, train_ratio=0.8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Full pipeline to train and evaluate a seq2seq model

    Args:
        x_data: Input time series data of shape [total_time_steps, n_features]
        y_data: Target time series data of shape [total_time_steps, m_features]
        input_window: Length of input sequence
        output_window: Length of output sequence
        n_features: Number of input features
        m_features: Number of output features
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        train_ratio: Ratio of data to use for training
        device: Device to run the model on

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    # Create dataset
    dataset = TimeSeriesDataset(x_data, y_data, input_window, output_window)

    # Split into train and validation sets
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize encoder and decoder
    encoder = Encoder(input_dim=n_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2)
    decoder = Decoder(output_dim=m_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2)

    # Initialize model
    model = Seq2SeqModel(encoder, decoder, input_dim=n_features, output_dim=m_features).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_seq2seq_model.pt')

    # Load best model
    model.load_state_dict(torch.load('best_seq2seq_model.pt'))

    return model, train_losses, val_losses


# Function to make predictions
def predict(model, x_data, input_window, output_window, device):
    """
    Make predictions using the trained model

    Args:
        model: Trained seq2seq model
        x_data: Input time series data of shape [total_time_steps, n_features]
        input_window: Length of input sequence
        output_window: Length of output sequence
        device: Device to run the model on

    Returns:
        predictions: Predicted time series
    """
    model.eval()

    # Convert to torch tensor
    x_tensor = torch.FloatTensor(x_data).to(device)

    # Initialize predictions list
    predictions = []

    with torch.no_grad():
        for i in range(0, len(x_data) - input_window + 1):
            # Get input sequence
            src = x_tensor[i:i + input_window].unsqueeze(0)  # Add batch dimension

            # Make prediction
            prediction = model(src, output_window)

            # Store prediction
            predictions.append(prediction.squeeze(0).cpu().numpy())

    return predictions


# Example usage
if __name__ == "__main__":
    # Example parameters
    n_features = 3  # Dimensionality of input features
    m_features = 2  # Dimensionality of output features
    total_time_steps = 1000  # Total time steps
    input_window = 48  # Input sequence length
    output_window = 50  # Output sequence length

    # Generate synthetic data for demonstration
    time = np.arange(total_time_steps)

    # Generate x data (nD)
    x_data = np.zeros((total_time_steps, n_features))
    x_data[:, 0] = np.sin(0.01 * time)
    x_data[:, 1] = np.cos(0.02 * time)
    x_data[:, 2] = np.sin(0.03 * time) * np.cos(0.01 * time)

    # Generate y data (mD)
    y_data = np.zeros((total_time_steps, m_features))
    y_data[:, 0] = np.sin(0.02 * time + 10) + 0.1 * x_data[:, 0]
    y_data[:, 1] = np.cos(0.01 * time + 5) + 0.2 * x_data[:, 1] - 0.1 * x_data[:, 2]

    # Add some noise
    x_data += 0.1 * np.random.randn(*x_data.shape)
    y_data += 0.1 * np.random.randn(*y_data.shape)

    # Normalize the data
    x_mean, x_std = np.mean(x_data, axis=0), np.std(x_data, axis=0)
    y_mean, y_std = np.mean(y_data, axis=0), np.std(y_data, axis=0)

    x_data = (x_data - x_mean) / x_std
    y_data = (y_data - y_mean) / y_std

    # Run the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, train_losses, val_losses = run_seq2seq_model(
        x_data, y_data, input_window, output_window, n_features, m_features,
        hidden_dim=64, num_layers=2, batch_size=32, learning_rate=0.001, epochs=100,
        device=device
    )

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Make predictions
    predictions = predict(model, x_data, input_window, output_window, device)

    # Plot a sample prediction
    sample_idx = 100

    plt.figure(figsize=(12, 10))

    # Plot each output dimension
    for i in range(m_features):
        plt.subplot(m_features, 1, i + 1)

        # True values
        true_vals = y_data[input_window + sample_idx:input_window + sample_idx + output_window, i]
        plt.plot(range(output_window), true_vals, 'b-', label='True')

        # Predicted values
        pred_vals = predictions[sample_idx][:, i]
        plt.plot(range(output_window), pred_vals, 'r--', label='Predicted')

        plt.title(f'Feature {i + 1} Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()