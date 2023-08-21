

# Time & Date related imports
import datetime as dt
from zoneinfo import ZoneInfo
from dateutil import tz
from dateutil import parser


# Pandas related imports
import pandas as pd


# CCXT related imports
import ccxt
from ccxt.base.errors import BadSymbol








class CryptoData:
    def __init__(self, ticker: str, exchange: str = "Kraken") -> None:
        self.ticker = ticker
        self.exchange = getattr(ccxt, exchange.lower())()
    '''------------------------------------'''
    def get_OHLCV_data(self, market:str = "USD", timeframe: str = "5m", limit: int = 1000, convert_to_local_tz: bool = True):
        try:
            # Create a trading pair to retrieve. Ex: BTC/USDT
            trading_pair = f"{self.ticker}/{market}"
            # Column names for the dataframe.
            columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            # Get the candles data at the desired timeframe. 
            candles = self.exchange.fetch_ohlcv(trading_pair, timeframe, limit=limit)
            # Convert list of lists to dataframe.
            candles_df = pd.DataFrame(candles)
            candles_df.columns = columns
            # Convert volume into local market currency. 
            candles_df['$volume'] = candles_df['volume'] * candles_df['close']
            # Format timestamps column. 
            candles_df['time'] = pd.to_datetime(candles_df['time'], unit='ms')
            # Convert timestamps to local time.
            if convert_to_local_tz:
                candles_df['time'] = [self.convert_to_local_timezone(i) for i in candles_df['time']] 

            return candles_df
        except BadSymbol as e:
            print(f"[OHLCV Error] {e}")
    '''------------------------------------'''
    '''------------------------------------'''
    def convert_to_local_timezone(self, timestamp):
        '''
        This function assumes your timestamps are in UTC format.
        It will convert the timestamp to your local timezone. '''

        # Assign from_zone to the UTC timezone, since the timestamps are expected to be in UTC format. 
        from_zone = tz.tzutc()
        to_zone = tz.tzlocal()
        # Add the timezone information to the timestamp.
        timestamp = timestamp.replace(tzinfo=from_zone)
        # Convert the timezone.
        converted_time = timestamp.astimezone(to_zone)
        return converted_time
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
