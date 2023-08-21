


# Number storage/manipulation imports
import pandas as pd
import numpy as np


# Graphing related imports
import matplotlib.pyplot as plt


# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler



class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers) -> None:
        super().__init__()
        '''
        input_size: Number of features. In this case the only feature is closing price, so the input_size is 1. 
        hidden_layer: Number of hidden layers inbetween the input and output layer. 
        num_stacked_layers: As LSTMs run, they will produce their own sequence. Stacked layers will use sequence data of previous layers. 
        '''
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        # Determine device. 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Create LSTM model 
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

        # Fully connected layers. Takes the amount of hidden layers. Since we are only predicting one output "close price", 1 is passed in the second argument.
        self.fc = nn.Linear(hidden_size, 1)
    '''------------------------------------'''
    def forward(self, x):
        '''
        '''
        batch_size = x.size(0)
        # Initialize variables required for LSTM.
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        # Get output.
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])










class TimeSeriesData(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class PyTorchPredictor:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    '''------------------------------------'''
    def format_data(self, df: pd.DataFrame):
        # Turn date column to pandas date type.
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "close"]]
        # Determine how many previous candles the model will take into account.
        lookback = 7

        shifted_df = self.prepare_dataframe_for_LSTM(df, n_steps=lookback)
        shifted_df = shifted_df.to_numpy()

        # Create a scaler with a range of -1 to 1. 
        scaler = MinMaxScaler(feature_range=(-1,1))
        # Fit the data to be within the feature range. 
        shifted_df = scaler.fit_transform(shifted_df)

        # Get all of the rows from the second column + any columns that follow. 
        X = shifted_df[:, 1:]
        # Get all of the rows from only first column. This is our predictor column.
        # Another note: X.shape = (1000 rows, 7 columns) *if look back is 7. 
        # y.shape (1000 rows,) because there is only 1 column. 
        y = shifted_df[:, 0]
        # Flip the dataframe horizontally. 
        # Originally the dataframe reads close(t-1), close(t-2), etc. 
        # After flipping it will read. close(t-7), close(t-6), etc. 
        # This is because the data needs to be in a sequential order. 
        X = deepcopy(np.flip(X, axis=1))
        # Create a index to determine length of training data.
        split_index = int(len(X) * 0.95)
        # Create training data that includes every index up to "split_index"
        X_train = X[:split_index]
        # Create test data that includes the elements at "split_index" + any elements that follow it. 
        X_test = X[split_index:]
        # Create the same variables for y.
        y_train = y[:split_index]
        y_test = y[split_index:]

        # LSTMs require an extra dimension at the end. Reshape matrixes to add extra dimension.
        X_train = X_train.reshape((-1, lookback, 1))
        X_test = X_test.reshape((-1, lookback, 1))
        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        # Create PyTorch tensors.
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        # Create datasets from the training and testing data
        train_dataset = TimeSeriesData(X_train, y_train)
        test_dataset = TimeSeriesData(X_test, y_test)

        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        for _, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            break


        

    '''------------------------------------'''
    def prepare_dataframe_for_LSTM(self, df: pd.DataFrame, n_steps: int):
        df = deepcopy(df)
        # Set the index of the timestamp.
        df.set_index("time", inplace=True)
        # Shift data points.
        for i in range(1, n_steps+1):
            df[f"close(t-{i})"] = df["close"].shift(i)
        # Drop any NaN values.
        df.dropna(inplace=True)

        return df
    '''------------------------------------'''
    def create_batches(self, loader: DataLoader):
        for _, batch in enumerate(loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            break
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''


if __name__ == "__main__":

    model = PredictionModel(1, 4, 1)
    model.to(model.device)
    print(f"Model: {model}")