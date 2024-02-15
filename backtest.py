from datetime import timedelta
import gym
from .database import *
from .pvm import PVM
import pandas as pd
import numpy as np

class Backtest:
    def __init__(self,
                test_environment:gym.Env,
                portfolio_vector:PVM
                ) -> None:
        self.env = test_environment
        self.dates = test_environment._date_memory
        self.tickers = test_environment._tic_list
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]
        self.data = Database()
        self.df_returns = self.data.get_info(
            ['RET_'+ticker for ticker in self.tickers], 
            open_date=self.start_date, close_date=self.end_date)
        self.df_returns = self.df_returns.loc[self.dates]
        self.df_returns.columns = self.tickers
        self.cash_df = pd.DataFrame(index=self.dates, columns=['CASH'])
        self.cash_df = self.cash_df.fillna(0)
        self.df_returns = pd.concat([self.cash_df,self.df_returns], axis=1)
        self.portfolio_vector = portfolio_vector
        weights =[]
        for row in np.vstack(portfolio_vector.memory[:-1]):
            weights.append(row/np.sum(row))
        self.weights = np.array(weights)
            
        self.ibov_data, self.cdi_data = None, None
        self._preprocessed = False

        
    def _preprocess(self):
        if self._preprocessed:
            return
        returns = self.df_returns.to_numpy()
        self._daily_returns = (np.sum(returns*self.weights, axis=1))
        self._cumulative_returns = np.cumprod(1+self._daily_returns)
        self._preprocessed = True

    def daily_returns(self, df=True):
        self._preprocess()
        if not df:
            return self._daily_returns
        df = pd.DataFrame(index=self.dates, columns=['Daily Returns'])
        df['Daily Returns'] = self._daily_returns
        return df
    
    def hit_ratio(self):
        positive_returns = (self._daily_returns > 0)
        return sum(positive_returns)/len(self._daily_returns)
    
    def cumulative_daily_returns(self, df=True):
        self._preprocess()
        if not df:
            return self._cumulative_returns
        df = pd.DataFrame(index=self.dates, columns=['Cumulative Daily Returns'])
        df['Cumulative Daily Returns'] = self._cumulative_returns
        return df
    
    def hit_ratio(self):
        self._preprocess()
        positive_returns = (self._daily_returns > 0)
        hit_ratio_value = sum(positive_returns)/len(self._daily_returns)
        return hit_ratio_value
    
    def sharpe_ratio(self, risk_free_rate=0.02, annualize:bool=True, periods:int = 252):
        self._preprocess()
        sharpe_ratio_value = (self._daily_returns.mean() - risk_free_rate) \
             / self._daily_returns.std()
        if annualize:
            sharpe_ratio_value*= np.sqrt(1 if periods is None else periods)
        return sharpe_ratio_value

    def general_cumulative_daily_returns(self):
        portfolio_df = self.cumulative_daily_returns()
        self.data.reset()
        if self.ibov_data is None:
            ibov_data = self.data.get_info('CRET_IBOV', self.start_date,self.end_date)
            ibov_data.columns=['IBOV']
            ibov_data['IBOV'] = ibov_data[['IBOV']].values +1
            ibov_data = ibov_data.dropna()
            ibov_data['IBOV']=ibov_data['IBOV']
            self.ibov_data = ibov_data
        if self.cdi_data is None:
            cdi_data = self.data.get_info('CDI', self.start_date,self.end_date)
            cdi_data.columns=['CDI']
            cdi_data = cdi_data.dropna()
            cdi_data['CDI'] = ((cdi_data['CDI']/100 + 1).cumprod())
            self.cdi_data = cdi_data
        cumulative_df = pd.concat([portfolio_df,self.ibov_data,self.cdi_data], axis=1)
        return cumulative_df.dropna()