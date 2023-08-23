
# Number storage/manipulation related imports
import numpy as np
import pandas as pd

# Tulip related imports 
import tulipy as ti



class TechnicalIndicators:
    def __init__(self) -> None:
        pass
    '''------------------------------------'''
    def get_macd(self, data: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        '''
        data: Pandas dataframe containing close price data.
        short_period: Number of periods used to calculate the fast EMA in the MACD calculation.
        long_period: Number of periods used to calculate the slow EMA in MACD calculation. 
        signal_period: Determine the number of periods to use when calculating the EMA on the MACD itself. 
                       Running the EMA on the MACD gives us the signal line.
        '''
        # Convert to numpy array.
        closing_prices = np.array(data["close"])
        # Calculate the MACD and signal line.
        macd, signal, _ = ti.macd(closing_prices, short_period=short_period, long_period=long_period, signal_period=signal_period)
        return macd, signal
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