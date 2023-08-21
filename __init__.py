

# Datasource imports
from DataSources.crypto_data import CryptoData

# Model imports
from Models.predictor_pytorch import PyTorchPredictor
from Models.predictor_pytorch import PredictionModel







if __name__ == "__main__":

    # Create instances of classes.  
    c = CryptoData("BTC")
    ptorch = PyTorchPredictor()

    # Get the open, high, low, close, volume data and set the data in "ptorch".
    ohlcv = c.get_OHLCV_data()
    ptorch.set_parameters(ohlcv)

    # Set the model.
    ptorch.set_model(1,4,1)


    ptorch.train_epochs(num_epochs=10)

    ptorch.plot_predictions_test()



