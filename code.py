import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class StockDataset(Dataset):
    
    def __init__(self, data):
      self.data = data

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx,0:1], dtype=torch.float32), torch.tensor(self.data[idx,1], dtype=torch.float32)
class LinearRegressionModel(nn.Module):
    

    def __init__(self, input_size=1, output_size=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def predict_stock_price(data, days_to_predict=5):
    

    # Prepare Data
    # Ensure the data is a NumPy array
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    price_shift = np.roll(data[:,2], 1)
    price_shift[0] = data[0,2]
    prepared_data = np.stack([price_shift, data[:,2]], axis=1)[1:,:]
    #Split into training and testing
    train_ratio = 0.8
    train_size = int(len(prepared_data)*train_ratio)
    train_data, test_data = prepared_data[:train_size], prepared_data[train_size:]
    # Create PyTorch Dataset and DataLoader
    train_dataset = StockDataset(train_data)
    test_dataset = StockDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 2. Set up device, model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 3. Training Loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dataloader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          
          # forward pass
          outputs = model(x_batch)
          loss = criterion(outputs.squeeze(), y_batch)

          # backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 4. Evaluate the Model
    model.eval()  # Put model in evaluation mode
    with torch.no_grad(): # Disable gradient calculation
      total_loss = 0
      for x_test, y_test in test_dataloader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pred = model(x_test)
        loss = criterion(y_pred.squeeze(), y_test)
        total_loss+=loss.item()
      mse = total_loss/len(test_dataloader)
      print(f"\nMean Squared Error (MSE) on test set: {mse:.2f}")

    # 5. Prediction for future days:
    last_price = torch.tensor([prepared_data[-1][1]], dtype=torch.float32).to(device)
    predicted_prices = []
    current_prediction = last_price
    for _ in range(days_to_predict):
        with torch.no_grad():
          next_prediction = model(current_prediction.unsqueeze(0)).squeeze()
        predicted_prices.append(next_prediction.item())
        current_prediction = next_prediction

    # 6. Plotting and Display
    print("\nPredicted prices:")
    for i, price in enumerate(predicted_prices):
        print(f"Day {i+1}: {price:.2f}")

    # Plotting the data:
    data = prepared_data
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data[-50:])), data[-50:,1], label='Actual Closing Prices (Last 50 days)', color='blue')
    future_days = range(len(data[-1:]), len(data[-1:]) + days_to_predict)
    plt.plot(future_days, predicted_prices, label='Predicted Closing Prices', color='red', linestyle='--')

    plt.xlabel('Day Index')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load the data:
file_path = "GOOG.csv"
data = pd.read_csv(file_path)

# Get Input from User
if __name__ == "__main__":
  days = int(input("Enter number of days to predict: "))
  predict_stock_price(data, days)