import gymnasium as gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

class Finance(gym.Env):
    def __init__(
        self,
        df,
        initial_amount,
        #order_df=True,
        #return_last_action=False,
        #normalize_df="by_previous_time",
        reward_scaling=1,
        #comission_fee_model="trf",
        #comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        time_window=1,
        verbose=True,
        metrics_period=100,
        #cwd="./",
        #new_gym_api=False
        ):
        self.df = df
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.features = features
        self.valuation_feature = valuation_feature
        self.time_column = time_column
        self.tic_column = tic_column
        self.time_window = time_window 
        self.verbose = verbose
        self.metrics_period = metrics_period
        
        self.tickers = df[tic_column].unique()
        self.dates = sorted(df[time_column].unique())
        self.last_date = self.dates[-1]
        
        self.portfolio_size = len(self.tickers)
        action_space = 1 + self.portfolio_size
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        self.episode_length = len(self.dates) - time_window -1
        self.df_returns = None
        self.observation_space = spaces.Box(
                low=-(10E10),
                high=(10E10),
                shape=(len(self.features), self.portfolio_size, self.time_window),
            )
        self.reset()
        self._create_returns_df()
        
        
        
    def reset(self):
        self.time_index = self.time_window -1
        self.memory = {}
        self.portifolio_value = self.initial_amount
        self.temp_returns =[]
        self.returns =[]
        current_date, _ = self._get_current_date()        
        self.state = self._get_current_state_from_date(current_date)
        self.terminal= False
        self.reward = 0
        return self.state
        
    def _get_current_state_from_date(self, date):
        df_selected = self.df[self.df[
            self.time_column]<=date].tail((self.time_window)*len(self.tickers))
        df_selected = df_selected[self.features]
        state_list =[]
        for index in set(df_selected.index):
            data_2d = df_selected.loc[index]
            state_list.append(data_2d.values)
        state_list =np.array(state_list).T  #### ver se precisa T
        return state_list 
        
    def _create_returns_df(self):
        df_returns = pd.DataFrame()
        for ticker in self.tickers:
            df_to_check = self.df[self.df[self.tic_column] == ticker]
            df_to_check= df_to_check.set_index(self.time_column)
            returns = df_to_check[self.valuation_feature].pct_change()
            df_returns[ticker] = returns
        df_returns = df_returns.fillna(0)
        cols = df_returns.columns
        df_returns['cash'] = 0
        df_returns = df_returns[['cash']+list(cols)]
        self.df_returns = df_returns
    
    def cumulative_daily_returns(self, returns):
        returns = np.array(returns)
        cumulated_returns = (returns+1).cumprod()
        return cumulated_returns[-1]
    
    def hit_ratio(self, returns):
        returns = np.array(returns)
        positive_returns = (returns > 0)
        hit_ratio_value = np.sum(positive_returns)/len(returns)
        return hit_ratio_value
    
    def sharpe_ratio(self,returns, risk_free_rate=0.002):
        returns = np.array(returns)
        average_return = np.mean(returns)
        std_dev_return = np.std(returns)
        sharpe_ratio_value = ((average_return - risk_free_rate) / std_dev_return) * np.sqrt(252)
        return sharpe_ratio_value
    
    def maximum_drawdown(self,returns):
        returns = np.array(returns)
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.argmax(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
        trough = np.argmax(cumulative_returns[:peak])
        max_drawdown = cumulative_returns[peak] - cumulative_returns[trough]
        return max_drawdown

    def _get_current_date(self):
        current_date = self.dates[self.time_index]
        return (current_date, self.time_index)
    
    def _softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output
    
    def get_metrics(self):
        temp_info ={
            "Sharpe Ratio": self.sharpe_ratio(self.temp_returns),
            "Hit Ratio": self.hit_ratio(self.temp_returns),
            "Cumulative Returns": self.cumulative_daily_returns(self.temp_returns),
            "Maximum Drawdown": self.maximum_drawdown(self.temp_returns)
        }
        general_info ={
            "Sharpe Ratio": self.sharpe_ratio(self.returns),
            "Hit Ratio": self.hit_ratio(self.returns),
            "Cumulative Returns": self.cumulative_daily_returns(self.returns),
            "Maximum Drawdown": self.maximum_drawdown(self.returns),
        }
        info = {'temp': temp_info, 'general':general_info}
        return info
    
    def print_info(self,current_date):
        metrics = self.get_metrics()
        temp_info = metrics['temp']                
        print("\n==========================")
        print(f'Info for date: {current_date}')
        print("==========================")
        print("CURRENT METRICS")
        for key in temp_info.keys():
            print(f'{key}: {round(temp_info[key],3)}')
        general_info = metrics['general']
        print("\nGENERAL METRICS")
        for key in general_info.keys():
            print(f'{key}: {round(general_info[key],3)}')
        print("==========================")

    
    def step(self,actions):
        actions = np.array(actions, dtype=np.float32)
        actions = self._softmax_normalization(actions)
        actions = actions/np.sum(actions)
        current_date, current_index = self._get_current_date()
        if self.verbose:
            if (self.time_index-self.time_window +2)%(self.metrics_period)==0:
                self.print_info(current_date)
                self.temp_returns = []
        self.state = self._get_current_state_from_date(current_date)
        if current_date ==self.last_date:
            self.terminal=True
            return self.state, self.reward, self.terminal, self.info
        current_returns = self.df_returns.loc[current_date].values 
        total_returns = np.sum(current_returns*actions)
        self.temp_returns.append(total_returns)
        self.returns.append(total_returns)
        self.portifolio_value*=(1+total_returns)
        self.reward = total_returns*self.reward_scaling
        self.info ={
            "reward" : self.reward,
            "portifolio_value" : self.portifolio_value,
            "actions" : actions,
            "current_index" : current_index,
            "current_returns" : current_returns,
        }
        self.memory[current_date] = self.info
        self.time_index+=1
        return self.state, self.reward, self.terminal, self.info
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
        