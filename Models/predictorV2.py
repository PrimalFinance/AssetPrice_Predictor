# PyTorch related imports. 
import torch 
import torch.nn as nn

# Pandas/numpy related imports
import pandas as pd
import numpy as np

# Sci-kit Learn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

path_to_model = "D:\\Coding\\VisualStudioCode\\Projects\\Python\\AssetPrice_Predictor\\Models\\ModelFiles\\{}"

class PredictionNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, 64) # Output size of 64.
        self.fc2 = nn.Linear(64, 32) # Input size of 64. Output size of 32.
        self.fc3 = nn.Linear(32, 1) # Input size of 32. Output size of 1. (Since we are predicting one value, the close price)
        # Determine device. 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    '''------------------------------------'''
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Apply ReLU activation to fc1 output. 
        x = torch.relu(self.fc2(x)) # Apply ReLU activation to fc2 output. 
        x = self.fc3(x) # Output layer, no activation for regression. 
        return x
    






class NetworkManager:
    def __init__(self, model_name: str = "prediction_model") -> None:
        self.model_file = path_to_model.format(f"{model_name}.pth")
        # Create a scaler with a range of -1 to 1. Meaning all data fed to it will be scaled within that feature range. 
        self.default_features = ["rsi", "rsi_oversold", "rsi_overbought", "macd", "signal", "macd_over"]
        self.scaler = StandardScaler()
    '''------------------------------------'''
    def save_model(self):
        # Save the model's state_dict to a file.
        torch.save(self.model.state_dict(), self.model_file)
    '''------------------------------------'''
    def load_model(self):
        # Load the saved state_dict
        self.model = PredictionNetwork(3)
        self.model.load_state_dict(torch.load(self.model_file))

    '''------------------------------------'''
    def set_training_data(self, df: pd.DataFrame, input_features: list = []):
        # Turn date column to pandas date type.
        df["time"] = pd.to_datetime(df["time"])
        
        # Prepare the input features (past close prices) and target (next close price)
        past_periods = 5
        # Create a new DataFrame to store the data with input features and target
        data = pd.DataFrame(columns=['input_features', 'target'])

        # Prepare the input features (past close prices) and target (next close price)
        for i in range(past_periods, len(df)):
            # Past closing price values. 
            past_close_prices = df['close'].iloc[i - past_periods:i].tolist()

            # Get the volume.
            past_volume_values = df["$volume"].iloc[i - past_periods:i].to_list()

            # Past RSI related values. 
            past_rsi_values = df["rsi"].iloc[i - past_periods:i].to_list()
            past_rsi_oversold_values = df["rsi_oversold"].iloc[i - past_periods:i].to_list()
            past_rsi_overbought_values = df["rsi_overbought"].iloc[i - past_periods:i].to_list()
            
            # Past MACD related values.
            past_macd_values = df["macd"].iloc[i - past_periods:i].to_list()
            past_signal_values = df["signal"].iloc[i - past_periods:i].to_list()
            past_histogram_values = df["histogram"].iloc[i - past_periods:i].to_list()
            past_macd_over_values = df["macd_over"].iloc[i - past_periods:i].to_list()
            next_close_price = df['close'].iloc[i]
            data.loc[i - past_periods] = [past_close_prices + past_volume_values + past_rsi_values + past_rsi_oversold_values + past_rsi_overbought_values + past_macd_values + past_signal_values + past_histogram_values + past_macd_over_values, next_close_price]
        
        # Convert boolean lists in "input_features" to binary values
        data['input_features'] = data['input_features'].apply(lambda x: [int(item) if isinstance(item, bool) else item for item in x])
        

        #print(f"Data: {data[-10:]}")
        
        # Flatten the nested lists within the "input_features" column
        # Flatten the "input_features" column
        flat_input_array = [item if isinstance(item, bool) else subitem for sublist in data['input_features'] for item in sublist for subitem in (item if isinstance(item, list) else [item])]

        # Reshape array to fit in scaler. Ex: (32175,) -> (32175, 1)
        flat_input_array = np.array(flat_input_array).reshape(-1, 1)

        print(f"Flat: {flat_input_array.shape}")
        scaled_data = self.scaler.fit_transform(flat_input_array)
        
        print(f"X: {scaled_data}")

        for i in scaled_data:
            print(f"I: {i}")
        
        # Get the desired columns from the list.
        col_names = df.columns[4:]
        col_names = list(col_names)
        # Remove unneeded columns.
        col_names.remove("$volume")
        col_names.remove("close_pct_change")
        cols = []
        for col in col_names:
            for i in range(past_periods):
                cols.append(f"{col}_{i+1}")
        
       


        #scaled_input_features = self.scaler.fit_transform(flat_input_array)
        
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