


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
        return out










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
        # Number of candles to search before the current candle. 
        self.lookback = 7
        # Number of elements to be included in one batch.
        self.batch_size = 16

        # Set train_looder & test_loader to None
        self.train_loader = None
        self.test_loader = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None
        self.optimizer = None
        # Create a scaler with a range of -1 to 1. 
        self.scaler = MinMaxScaler(feature_range=(-1,1))

        # Set the loss function 
        self.loss_function = nn.MSELoss()

        # Set the learning rate
        self.learning_rate = 0.001
    '''------------------------------------'''
    def set_parameters(self, df: pd.DataFrame):
        # Turn date column to pandas date type.
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "close"]]
        # Determine how many previous candles the model will take into account.
        lookback = 7

        shifted_df = self.prepare_dataframe_for_LSTM(df, n_steps=lookback)
        shifted_df = shifted_df.to_numpy()

        
        # Fit the data to be within the feature range. 
        shifted_df = self.scaler.fit_transform(shifted_df)

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
        self.X_train = torch.tensor(X_train).float()
        self.y_train = torch.tensor(y_train).float()
        self.X_test = torch.tensor(X_test).float()
        self.y_test = torch.tensor(y_test).float()

        # Create datasets from the training and testing data
        train_dataset = TimeSeriesData(self.X_train, self.y_train)
        test_dataset = TimeSeriesData(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)


        

    '''------------------------------------'''
    def set_model(self, input_layer, hidden_layer, stacked_layers):
        self.model = PredictionModel(input_size=input_layer, hidden_size=hidden_layer, num_stacked_layers=stacked_layers)
        if self.optimizer == None:
            self.set_optimizer()
    '''------------------------------------'''
    def get_model(self):
        return self.model
    '''------------------------------------'''
    def set_optimizer(self):
        if self.model != None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise("[Error] Model has not been initiated yet. Optimizer cannot be created.")
    '''------------------------------------'''
    def get_optimizer(self):
        return self.optimizer
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
    def train_epochs(self, num_epochs: int = 10):
        
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch=epoch)
            self.validate_one_epoch()

    '''------------------------------------'''
    def train_one_epoch(self, epoch: int):

        if self.train_loader != None:
            self.model.train(True)
            print(f"Epoch: {epoch+1}")
            running_loss = 0.0

            for batch_index, batch in enumerate(self.train_loader):
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()
                self.optimizer.zero_grad()
                # Iterate a backward pass through the loss to get the gradient. 
                loss.backward()
                # Step in the direction of the gradient to improve model. 
                self.optimizer.step()

                if batch_index % 100 == 0: # print every 100 batches.
                    avg_loss_accross_batches = running_loss / 100
                    print(f"Batch: {batch_index+1}, Loss: {'{0:.3f}'.format(avg_loss_accross_batches)}")

                    running_loss = 0.0
        else:
            raise("[Error] Model parameters need to be set first.")
    '''------------------------------------'''
    def validate_one_epoch(self):
        if self.test_loader != None:
            # Put it in false since we are not training, but evaluating. 
            self.model.train(False)
            running_loss = 0.0

            for batch_index ,batch in enumerate(self.test_loader):
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                with torch.no_grad():
                    output = self.model(x_batch)
                    loss = self.loss_function(output, y_batch)
                    running_loss += loss
            avg_loss_across_batches = running_loss / len(self.test_loader)
            print(f"Val loss: {'{0:.3f}'.format(avg_loss_across_batches)}")
            print(f"**********************************************************")
        else:
            raise("[Error] Model parameters need to be set first.")
    '''------------------------------------'''
    def plot_predictions_train(self):

        if self.X_train != None and self.y_train != None:

            with torch.no_grad():
                predicted = self.model(self.X_train.to(self.device)).to("cpu").numpy()
                train_predictions = predicted.flatten()

                # Create an empty numpy array for X_train that is the shape that the scaler accepts. 
                x_array = np.zeros((self.X_train.shape[0], self.lookback+1))
                x_array[:, 0] = train_predictions
                x_array = self.scaler.inverse_transform(x_array)

                # Copy data back
                train_predictions = deepcopy(x_array[:, 0])

                # Create another empty array for y_train
                y_array = np.zeros((self.X_train.shape[0], self.lookback+1))
                y_array[:, 0] = self.y_train.flatten()
                y_array = self.scaler.inverse_transform(y_array)

                new_y_train = deepcopy(y_array[:, 0])

            plt.plot(new_y_train, label="Actual Close")
            plt.plot(train_predictions, label="Predicted Close")
            plt.xlabel("Day")
            plt.ylabel("Close")
            plt.legend()
            plt.show()
        else:
            raise("[Error] Parameters have not been set, and model has not been trained. ")

    '''------------------------------------'''
    def plot_predictions_test(self):
        if self.X_test != None and self.y_test != None:
            test_predictions = self.model(self.X_test.to(self.device)).detach().cpu().numpy().flatten()
            # Create empty numpy array for X_test to fit into scaler. 
            # The scaler will inverse the scaling and take the value within the feature range to it's original value. 
            x_array = np.zeros((self.X_test.shape[0], self.lookback+1))
            x_array[:, 0] = test_predictions
            x_array = self.scaler.inverse_transform(x_array)
            test_predictions = deepcopy(x_array[:, 0])
            # Create empty numpy array for y_test to fit into scaler. 
            y_array = np.zeros((self.X_test.shape[0], self.lookback+1))
            y_array[:, 0] = self.y_test.flatten()
            y_array = self.scaler.inverse_transform(y_array)
            new_y_test = deepcopy(y_array[:, 0])

            plt.plot(new_y_test, label="Actual Close")
            plt.plot(test_predictions, label="Predicted Close")
            plt.xlabel("Day")
            plt.ylabel("Close")
            plt.legend()
            plt.show()

            

        else:
            raise("[Error] Parameters have not been set, and model has not been trained. ")
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