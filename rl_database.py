from .database import *
from pykalman import KalmanFilter
from finta import TA
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import warnings
warnings.simplefilter("ignore")

class RLDatabase():
    def __init__(self,
                 start_train: str = None,
                 end_train: str = None,
                 end_test: str = None,
                 tickers: [str] = None,
                 number_of_tickers_to_consider:int = 30,
                 test_time_window:int=0,
                 same_index:bool=False) -> None:
        if end_test is None:
            end_test = date.today()
        end_test = pd.to_datetime(end_test)
        if end_train is None:
            end_train = end_test - timedelta(days=1*365)
        end_train = pd.to_datetime(end_train)
        if start_train is None:
            start_train = end_train - timedelta(days=10*365)
        start_train = pd.to_datetime(start_train)
        self.start_train = start_train
        self.end_train = end_train
        self.end_test = end_test
        self.test_time_window = test_time_window
        self.same_index = same_index
        self.data = Database()
        self.close_df = pd.DataFrame()
        self.tickers = tickers
        self.number_of_tickers_to_consider = number_of_tickers_to_consider
        self.tickers = tickers
        self.base_df_columns = ['USD/BRL',  'IBOV',  'SPX', 'DJI', 'NASDAQ',
                                'CDI', 'IPCA', 'SELIC', 'IGPM']
        self.minimum_of_years = 0.8*(self.end_train - 
                                     self.start_train).days/365
        self.ohlcv_dfs= {}
        self.TA_dfs = {}
        self.specif_dfs={}
        self.general_df = None
        self.test_df = None

    def _create_ohlcv_dfs(self):
        ohlcv = self.data.get_info(
            tickers=self.tickers,
            open_date=self.start_train, 
            close_date=self.end_test, 
            info = 'ohlcv')
        for ticker in self.tickers:
            cols = [ticker+'_open', 
                    ticker+'_high', 
                    ticker+'_low', 
                    ticker+'_close', 
                    ticker+'_volume' ]
            try:
                ticker_df = ohlcv[cols]                
                ticker_df.columns =["open","high","low","close","volume"]
                ticker_df = ticker_df.dropna()
                if len(ticker_df.index) > self.minimum_of_years*252:
                    self.ohlcv_dfs[ticker] = ticker_df
                    self.close_df[ticker] = ticker_df['close']
            except:
                pass
        self.tickers = list(self.ohlcv_dfs.keys())        

    def _create_TA_dfs(self):
        ta_info={
            'rsi': {'method': TA.RSI},
            'mom': {'method': TA.MOM},
            'cci': {'method': TA.CCI},
            'stoch': {'method': TA.STOCH},
            'williams': {'method': TA.WILLIAMS},
        }
        if len(self.ohlcv_dfs.keys()) == 0:
            self._create_ohlcv_dfs()
        for ticker in self.tickers:
            try:
                ohlcv = self.ohlcv_dfs[ticker]
                dfs_to_concat =[]
                for ta in ta_info.keys():
                    method = ta_info[ta]['method']
                    df = method(ohlcv)
                    df.name = ta
                    df.replace(-np.inf, np.nan, inplace=True)
                    df.replace(np.inf, np.nan, inplace=True)
                    dfs_to_concat.append(df)
                macd = TA.MACD(ohlcv, 12, 26, 9)
                macd.columns = ['macd', 'macd_signal']
                dfs_to_concat.append(macd)
                bbands = TA.BBANDS(ohlcv).dropna()
                bbands['bband'] = bbands['BB_UPPER'] - bbands['BB_LOWER']
                bbands = bbands[['bband']]
                dfs_to_concat.append(bbands)
                ta_df = pd.concat(dfs_to_concat, axis=1)
                self.TA_dfs[ticker]=ta_df.dropna()
            except:
                pass
        self.tickers = list(self.TA_dfs.keys())

    def df_MA(self, df,
              ma_type: str = 'SMA' or 'EMA',
              periods: [int] = [2, 4, 5, 7, 10, 14, 30, 45, 90],
              drop_index: bool = True):
        if type(periods) is int:
            periods = [periods]
        index = df.columns[-1]
        ma_type = ma_type.upper()
        df = df.dropna()
        if ma_type == 'SMA':
            method = df[index].rolling
        else:
            method = df[index].ewm
        for period in periods:
            df[ma_type+'_'+str(period) + '_' + index] = method(period).mean()
        if drop_index:
            df = df.drop(columns=index)
        return df

    def df_kalman(self, df,
                  covariances: [float] = [0.005, 0.01, 0.03, 0.05],
                  drop_index: bool = True):
        if type(covariances) is float:
            covariances = [covariances]
        index = df.columns[-1]
        df = df.dropna()
        initial = df[index][0]
        df_to_filter = df[[index]]
        for covariance in covariances:
            try:
                kf = KalmanFilter(transition_matrices=[1],
                                  observation_matrices=[1],
                                  initial_state_mean=initial,
                                  initial_state_covariance=1,
                                  observation_covariance=1,
                                  transition_covariance=covariance)
                state_means, _ = kf.filter(df_to_filter)
                df['KLM_'+str(covariance)+'_'+index] = state_means
            except:
                pass
        if drop_index:
            df = df.drop(columns=index)
        return df
    
    def create_moving_info(self, df, drop_index: bool = False):
        sma_df = self.df_MA(df, 'SMA', drop_index=True)
        ema_df = self.df_MA(df, 'EMA', drop_index=True)
        kalman_df = self.df_kalman(df, drop_index=True)
        return pd.concat([sma_df, ema_df, kalman_df], axis=1)
    
    def _create_personalized_base_df(self):
        personalized_df = pd.DataFrame()
        close_info = self.close_df
        for col in self.base_df_columns:
            base_info = self.base_df[[col+'_close']]
            moving_info = self.create_moving_info(base_info)
            values_corr = None
            for ticker in close_info.columns:
                ticker_info = close_info[[ticker]]
                df = pd.concat([ticker_info,base_info, moving_info], axis=1)
                df = df.dropna()      
                corr = df.loc[self.start_train:self.end_train].corr()
                if values_corr is None:
                    values_corr = corr.values[0,]
                else:
                    values_corr += corr.values[0,]
            values_corr = np.absolute(values_corr[1:])
            max_index = np.argmax(values_corr)
            best_moving_df = df[df.columns[max_index+1]]
            best_moving_df.name = col
            personalized_df = pd.concat([personalized_df, best_moving_df], axis=1)
        self.personalized_df = personalized_df.dropna()
    
    def create_general_database(self):
        if self.general_df is not None:
            return self.general_df
        if self.tickers is None:
            self.tickers = self.data.get_most_traded(maximum_date=self.end_train, 
                    previous_days_to_consider = 30, 
                    number_of_tickers = self.number_of_tickers_to_consider+10)
        self.base_df = self.data.get_info(self.base_df_columns,
            open_date=self.start_train,
            close_date=self.end_test)
        self._create_ohlcv_dfs()
        self._create_TA_dfs()
        self._create_personalized_base_df()
        dfs_to_unite = []
        if len(self.tickers) >= self.number_of_tickers_to_consider:
            self.tickers = self.tickers[:self.number_of_tickers_to_consider]
        for ticker in self.tickers:
            ohlcv = self.ohlcv_dfs[ticker]
            ta_df = self.TA_dfs[ticker]
            df = pd.concat([ohlcv, ta_df, self.personalized_df], axis=1)
            df.index = pd.to_datetime(df.index)
            df["tic"] = ticker
            df['date'] = df.index
            wd = pd.to_datetime(df.index)
            df['day'] = wd.weekday
            df['day'] = df['day'].values.astype(float)
            df = df.dropna()
            if len(df.index) > self.minimum_of_years*252:
                dfs_to_unite.append(df)
        self.default_index = dfs_to_unite[0].index
        for df in dfs_to_unite[1:]:
            self.default_index = df.index.intersection(self.default_index)
        for i in range(len(dfs_to_unite)):
            dfs_to_unite[i] = dfs_to_unite[i].loc[self.default_index]
            dfs_to_unite[i] = dfs_to_unite[i].reset_index(drop=True)
        general_df = pd.concat(dfs_to_unite, axis=0, join='outer')
        general_df = general_df.sort_values(by=['date', 'tic'], ascending=[True, True])
        general_df = general_df.sort_index()
        self.general_df = general_df
        self.tickers = list(general_df.tic.unique())
        if not self.same_index:
            self.general_df = self.general_df.reset_index(drop=True)
        return self.general_df
    
    def create_train_database(self):
        general_df = self.create_general_database()
        train_df = general_df.loc[general_df['date'] <= self.end_train]
        return train_df
    
    def create_test_database(self):
        general_df = self.create_general_database()
        test_df = general_df.loc[general_df['date'] > self.end_train]
        first_index_to_consider = test_df.index[0]
        if self.test_time_window>0:
            time_window_multiplier = len(self.tickers)
            if self.same_index:
                time_window_multiplier = 1
            first_index_to_consider -= (self.test_time_window - 1)*time_window_multiplier
        test_df = general_df.loc[first_index_to_consider:]
        test_df.index = test_df.index - first_index_to_consider
        return test_df
    
    def create_train_test_database(self):
        train_df = self.create_train_database()
        test_df = self.create_test_database()
        info ={
            "train":train_df,
            "test":test_df
        }
        return info