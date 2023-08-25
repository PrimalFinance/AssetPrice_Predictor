

# Datasource imports
from DataSources.crypto_data import CryptoData

# Model imports
from Models.predictor_pytorch import PyTorchPredictor
from Models.predictor_pytorch import PredictionModel
from Models.predictorV2 import PredictionNetwork, NetworkManager




def test1(c: CryptoData):
    ptorch = PyTorchPredictor(lookback=7, batch_size=16, train_size=0.95)
    
    # Get the open, high, low, close, volume data and set the data in "ptorch".
    ohlcv = c.get_OHLCV_data(timeframe="30m")
    
    #d = c.get_alpha_vantage_candles("BTC")
    ptorch.set_parameters(ohlcv)

    # Set the model.
    ptorch.load_model()
    ptorch.train_epochs(num_epochs=1000)
    #ptorch.plot_predictions_train()
    ptorch.save_model()

def test2(c: CryptoData):
    # Create network object.
    ptorch = NetworkManager()


    ohlcv = c.get_OHLCV_data(timeframe="30m")
    ohlcv = c.ta.get_RSI(ohlcv)
    ohlcv = c.ta.get_MACD(ohlcv)

    ptorch.set_training_data(ohlcv)

if __name__ == "__main__":

    # Create instances of classes.  
    c = CryptoData("LINK", exchange="kraken")
    


    #test1(c)
    test2(c)



    



