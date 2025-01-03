import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
    def prepare_data(self, df):
        """Prepare data for training and testing."""
        try:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            features = df[required_columns].values
            train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)
            
            self.scaler.fit(train_data)
            scaled_train = self.scaler.transform(train_data)
            scaled_test = self.scaler.transform(test_data)
            
            sequence_length = 10
            X_train, y_train = self._create_sequences(scaled_train, sequence_length)
            X_test, y_test = self._create_sequences(scaled_test, sequence_length)
            
            return X_train, y_train, X_test, y_test, test_data
            
        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            raise

    def _create_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 3])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, X_test, y_test, batch_size=32, num_epochs=100):
        try:
            train_dataset = StockDataset(X_train, y_train)
            test_dataset = StockDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            input_size = X_train.shape[2]
            hidden_size = 64
            self.model = LSTMModel(input_size, hidden_size).to(self.device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters())
            best_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                val_loss = self._validate(test_loader, criterion)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info("Early stopping triggered")
                    break

                if epoch % 10 == 0:
                    logging.info(f'Epoch [{epoch}/{num_epochs}], '
                               f'Training Loss: {total_loss/len(train_loader):.4f}, '
                               f'Validation Loss: {val_loss:.4f}')

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def _validate(self, test_loader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                total_loss += loss.item()
        return total_loss / len(test_loader)

    def predict(self, input_data, days_ahead=5):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        try:
            self.model.eval()
            predictions = []
            current_sequence = input_data[-10:].copy()

            for _ in range(days_ahead):
                with torch.no_grad():
                    sequence = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                    pred = self.model(sequence).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    current_sequence[-1] = pred

            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions.flatten()

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

    def plot_predictions(self, last_known_prices, predictions, dates=None):
        """Plot the last known prices and predictions."""
        plt.figure(figsize=(12, 6))
        
        # Plot last known prices
        last_dates = dates[-len(last_known_prices):] if dates is not None else range(len(last_known_prices))
        plt.plot(last_dates, last_known_prices, label='Historical Prices', color='blue')
        
        # Generate future dates
        if dates is not None:
            future_dates = [dates[-1] + timedelta(days=i+1) for i in range(len(predictions))]
        else:
            future_dates = range(len(last_known_prices), len(last_known_prices) + len(predictions))
        
        # Plot predictions
        plt.plot(future_dates, predictions, label='Predictions', color='red', linestyle='--')
        
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if dates is not None:
            plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        self.model = None

    @staticmethod
    def main():
        try:
            # Load data
            df = pd.read_csv('GOOG.csv')
            
            # Handle date column
            date_columns = df.filter(like='date').columns.tolist()
            if 'Date' in df.columns:
                date_col = 'Date'
            elif len(date_columns) > 0:
                date_col = date_columns[0]
            else:
                raise ValueError("No date column found in the dataset")
            
            df[date_col] = pd.to_datetime(df[date_col])
            dates = df[date_col].values
            df.set_index(date_col, inplace=True)

            # Initialize and train predictor
            predictor = StockPredictor()
            X_train, y_train, X_test, y_test, test_data = predictor.prepare_data(df)
            predictor.train(X_train, y_train, X_test, y_test)

            # Make predictions
            last_sequence = X_test[-1]
            predictions = predictor.predict(last_sequence)
            
            # Plot results
            last_known_prices = test_data[-30:, 3]  # Last 30 days of closing prices
            predictor.plot_predictions(last_known_prices, predictions, dates[-30:])
            
            # Clean up
            predictor.cleanup()
            
            return predictions

        except Exception as e:
            logging.error(f"Error in main execution: {str(e)}")
            return None

if __name__ == "__main__":
    predictions = StockPredictor.main()
    if predictions is not None:
        logging.info(f"Predictions for next 5 days: {predictions}")