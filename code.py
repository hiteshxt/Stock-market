import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# [Previous StockDataset, LSTMPredictor, and StockPredictor classes remain the same]
# ... (Keep all the previous code from the first artifact)

def analyze_and_display_results(df, predictions):
    """Display detailed analysis and predictions"""
    
    # Calculate basic statistics
    last_known_price = df['Close'].iloc[-1]
    avg_predicted_price = predictions.mean()
    max_predicted_price = predictions.max()
    min_predicted_price = predictions.min()
    
    # Print analysis
    print("\n=== Stock Price Analysis ===")
    print(f"Last Known Price: ${last_known_price:.2f}")
    print(f"Average Predicted Price: ${avg_predicted_price:.2f}")
    print(f"Maximum Predicted Price: ${max_predicted_price:.2f}")
    print(f"Minimum Predicted Price: ${min_predicted_price:.2f}")
    
    # Calculate trend
    if avg_predicted_price > last_known_price:
        trend = "UPWARD"
        change_percent = ((avg_predicted_price - last_known_price) / last_known_price) * 100
    else:
        trend = "DOWNWARD"
        change_percent = ((last_known_price - avg_predicted_price) / last_known_price) * 100
    
    print(f"\nPredicted Trend: {trend} ({change_percent:.2f}% change)")
    
    # Print detailed predictions
    print("\n=== Daily Predictions ===")
    for date, price in predictions.items():
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Historical and Predicted Prices
    plt.subplot(2, 1, 1)
    plt.plot(df.index[-60:], df['Close'][-60:], label='Historical Data', color='blue')
    plt.plot(predictions.index, predictions.values, label='Predictions', color='red', linestyle='--')
    plt.title('Stock Price Predictions vs Historical Data')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Price Distribution
    plt.subplot(2, 1, 2)
    plt.hist(predictions.values, bins=20, color='green', alpha=0.7)
    plt.axvline(last_known_price, color='red', linestyle='--', label='Current Price')
    plt.title('Distribution of Predicted Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    print("Loading and preparing data...")
    df = pd.read_csv('GOOG.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Initialize predictor
    print("\nInitializing model...")
    predictor = StockPredictor(window_size=10, hidden_size=64, num_layers=2)
    
    # Train model
    print("\nTraining model...")
    train_losses, test_losses = predictor.train(df, epochs=100)
    print("Training completed!")
    
    # Generate predictions
    print("\nGenerating predictions...")
    future_predictions = predictor.predict_future(df, days_ahead=30)
    
    # Analyze and display results
    analyze_and_display_results(df, future_predictions)
    
    return future_predictions, train_losses, test_losses

if __name__ == "__main__":
    predictions, train_losses, test_losses = main()