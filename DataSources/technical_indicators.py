
# Number storage/manipulation related imports
import numpy as np
import pandas as pd
import pandas_ta as pta

# Tulip related imports 
import tulipy as ti



class TechnicalIndicators:

    # Predetermined values best fit for variables. 
    rsi_oversold = 30
    rsi_overbought = 70
    



    def __init__(self) -> None:
        pass
    '''------------------------------------'''
    def get_RSI(self, data: pd.DataFrame, rsi_period: int = 14):
        '''
        data: Pandas dataframe containing close prices.
        rsi_period: Number of candles to use in RSI calculation.
        '''
        # Calculate the RSI.
        rsi = pta.rsi(data["close"], length=rsi_period)
        data["rsi"] = rsi
        data["rsi_oversold"] = data["rsi"] < 30
        data["rsi_overbought"] = data["rsi"] > 70
        data["rsi_neutral"] = (data["rsi"] < 70) & (data["rsi"] > 30)
        return data
    '''------------------------------------'''
    def get_MACD(self, data: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        '''
        data: Pandas dataframe containing close price data.
        short_period: Number of periods used to calculate the fast EMA in the MACD calculation.
        long_period: Number of periods used to calculate the slow EMA in MACD calculation. 
        signal_period: Determine the number of periods to use when calculating the EMA on the MACD itself. 
                       Running the EMA on the MACD gives us the signal line.
        ''' 
        # Calculate the MACD and signal line.
        macd_results = pta.macd(data["close"], fast=short_period, slow=long_period, signal=signal_period)
        # Assign values to new columns.
        macd_col = f"MACD_{short_period}_{long_period}_{signal_period}"
        signal_col = f"MACDs_{short_period}_{long_period}_{signal_period}"
        histogram_col = f"MACDh_{short_period}_{long_period}_{signal_period}"
        data["macd"] = macd_results[macd_col]
        data["signal"] = macd_results[signal_col]
        data["histogram"] = macd_results[histogram_col]
        # Create a column that is a boolean representing when the MACD is over the signal line. 
        data["macd_over"] = data["macd"] > data["signal"]
        return data
    '''------------------------------------'''
    def support(self, df: pd.DataFrame, l: int, n1: int, n2: int) -> bool:
        '''
        df: DataFrame with price data.
        l: The current index of the candle.
        n1: Number of candles before index.
        n2: Number of candles after index.

        returns: Boolean describing if there is a support level at the current candle index.
        '''
        for i in range(l-n1+1, l+1):
            if (df.low[i]>df.low[i-1]):
                return 0 
        for i in range(l+1, l+n2+1):
            if(df.low[i] < df.low[i-1]):
                return 0
        return 1
    '''------------------------------------'''
    def support(self, df: pd.DataFrame, l: int, n1: int, n2: int) -> bool:
        '''
        df: DataFrame with price data.
        l: The current index of the candle.
        n1: Number of candles before index.
        n2: Number of candles after index.

        returns: Boolean describing if there is a resistance level at the current candle index.
        '''
        for i in range(l-n1+1, l+1):
            if (df.high[i]>df.high[i-1]):
                return 0 
        for i in range(l+1, l+n2+1):
            if(df.high[i] < df.high[i-1]):
                return 0
        return 1
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