import numpy as np
import pandas as pd
import time
from vars import *
import random
import datetime
from torch.utils.tensorboard import SummaryWriter

from math import log10, exp
from numpy import zeros
from scipy.optimize import fsolve
from matplotlib.pyplot import plot, title, show
from electrolyzer import Electrolyzer
from datetime import timedelta
from datetime import datetime
import gym
from gym import spaces

writer = SummaryWriter()


class EMS(gym.Env):
    def __init__(self, dynamic=True, eval=False, pv_gen=True, wind_gen=True, discrete=False, multiagent=False, n_agents=3):
                
        self.multiagent = multiagent
        self.discrete = discrete
        self.eval = False
        self.dynamic = True
        self.pv_gen = pv_gen
        self.wind_gen = wind_gen
        self.electrolyzer = Electrolyzer()
        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()

      
        if self.eval:
            self.random_day = 37946
        else:
            self.random_day = np.minimum(0, 37944 - NUM_TIME_STEPS - 24 * random.randint(0,1581))
            #self.random_day = np.maximum(0, 7945 - NUM_TIME_STEPS - random.randint(0,1553))  


        self.sun_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,1]
        self.sun_power = self.sun_powers[self.random_day]

        self.wind_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,7]
        self.wind_power = self.wind_powers[self.random_day]

        self.prices = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,5]
        self.price = self.prices[self.random_day]

        self.dates = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,0]
        self.date = self.dates[self.random_day]


        # reset parameters
        random.seed(time.perf_counter())
        self.natural_gas_price = 0
        self.natural_gas = 0      
        self.storage = 0
        self.hydrogen = 0
        self.hydrogen_total = 0
        self.power_from_grid = 0 
        self.time = 0
        self.moles = 0
        self.gas_consumption = 0
        self.pv_generation = 0
        self.wind_generation = 0
        self.ammonia = 0
        self.ammonia_produced = 0
        self.ammonia_total = 0  
        self.cost = 0
        self.paid_price = 0
        self.natural_gas_total = 0
        self.profit = 0
        self.done = False
 
    def step(self, action):
        #print(action)
        #action = np.clip(action, 0, 1)
        '''
        
        if type(action) is np.int64:
            print("discrete")
        
            action = action.item()
            action = str(action)
            action = action.rjust(4, '0')
            action = list(action)
            
            print(action)
            #action = np.clip(action, 0, 1)
            
        else:
            action = np.clip(action, 0, 1)
        '''
        ##############
        # RENEWABLES #
        ##############

        if self.pv_gen:
            pv_generation = self.sun_power
        else:
            pv_generation = 0 

        self.pv_generation = pv_generation

        if self.wind_gen:
            self.wind_generation = self.windkraft_ertrag(self.wind_power) * 2000
        else:
            self.wind_generation = 0 
       
        self.wind_generation = 0
        ###################
        # BATTERY STORAGE #
        ###################
        
        Pel = action[2] * ELECTROLYZER_POWER
        self.p_el = Pel
        load = STORAGE_CAPACITY * action[0]
        diff = load - self.storage
        Cbattery = 0
        Dbattery = 0
        
        
        if diff >= 0:
            diff = np.minimum(diff, C_MAX)
            Cbattery = np.minimum(diff, (STORAGE_CAPACITY - self.storage) / ETA)
            self.storage += Cbattery * ETA
        else:
            diff = np.maximum(diff, -D_MAX)
            Dbattery = np.maximum(diff, (MIN_STORAGE - self.storage) * ETA)
            self.storage += Dbattery / ETA
        ################
        # AMMONIA PRODUCTION #
        ################
        
        self.ammonia = 10000 * action[1]                                        # (mol)
        self.ammonia_produced += self.ammonia                                   # (mol)
        # NH3 -> 3 mol hydrogen in 1 mol ammonia
        mol_hydrogen_needed = self.ammonia * 3                                  # (mol)

        ################
        # ELECTROLYZER #
        ################

        
        self.p_el = Pel
        self.hydrogen, Wcomp, P_tank, self.moles = self.electrolyzer.run(Pel)   # (kg), (W)
        
        # covert mol in t 
        self.hydrogen_total += self.hydrogen / 496.031                          # (tH2)
        
        ################
        # HYDROGEN STORAGE #
        ################

        PTank = np.minimum(mol_hydrogen_needed, self.electrolyzer.get_moles())  # (mol)
        self.electrolyzer.consume_moles_from_tank(PTank)                        # (mol)
        hydrogen_from_natural_gas_mol = (mol_hydrogen_needed - PTank)           # (mol)


   
        # Calculate the volume of natural gas needed to produce the required amount of hydrogen
        # (1.17 m3 of natural gas is needed to produce 1 mol of hydrogen)
        #self.natural_gas = hydrogen_from_natural_gas_mol * 0.0224 * 1.17       # (m³)
        

        # 1 mol Natural gas = 0.01604 kg Natural gas
        self.natural_gas = hydrogen_from_natural_gas_mol * 0.01604
        self.natural_gas_total += hydrogen_from_natural_gas_mol
        

        self.power_from_grid = Pel + Cbattery - Dbattery - pv_generation + np.abs(Wcomp*1e-3)



        self.power_from_grid *= 1e-6  

        self.moles = self.electrolyzer.get_moles()
        # calculate rewards
      
        r = self.reward(self.power_from_grid)
        


        # tensorboard scalars

        writer.add_scalar('DDPG/Battery+', action[0], self.time)  
        writer.add_scalar('DDPG/Ammonia', action[1], self.time)
        writer.add_scalar('DDPG/Electrolyzer', action[2], self.time)


        writer.add_scalar('States/Hydrogen Storage', self.electrolyzer.get_moles(), self.time)    
        writer.add_scalar('States/External Energy Source/Power From Grid', self.power_from_grid, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas (kg)', self.natural_gas, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas Price', self.natural_gas_price, self.time)
        writer.add_scalar('States/External Energy Source/Wind Generation', self.wind_generation, self.time)
        writer.add_scalar('States/Compressor', np.abs(Wcomp), self.time)
        writer.add_scalar('Actions/PV Generation', pv_generation * 1e-6, self.time)
        writer.add_scalar('Actions/Hydrogen', self.hydrogen, self.time)
        writer.add_scalar('Actions/Storage', self.storage, self.time)
        
        writer.add_scalar('DDPG/AmmoniaTotal', self.ammonia_total - self.ammonia_produced, self.time)
        writer.add_scalar('DDPG/Moles', self.electrolyzer.get_moles(), self.time)
        writer.add_scalar('Price/Grid Price', self.price, self.time)
        writer.add_scalar('Price/Natural Gas Price', self.natural_gas_price, self.time)
        

        
        self.time +=1
       
        if self.dynamic:
           
            self.sun_power = self.sun_powers[self.random_day + self.time]
            self.sun_power = np.maximum(0, self.sun_power)
            self.wind_power = self.wind_powers[self.random_day + self.time]
            self.price = self.prices[self.random_day + self.time]
            
            self.date = self.dates[self.random_day + self.time]
            date = datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S')
            date = date.strftime('%Y-%m-%d')
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == date];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=1)).strftime("%Y-%m-%d")];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=2)).strftime("%Y-%m-%d")];       
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=3)).strftime("%Y-%m-%d")];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=4)).strftime("%Y-%m-%d")];              
                      
            self.natural_gas_price = row['Price'].iloc[0]

        
        #if self.ammonia_produced >= self.ammonia_total:
        #    self.done = True
        
        if self.time >= NUM_TIME_STEPS:
            if self.eval:
                print("Hydrogen produced", self.hydrogen_total)
                print("Natural gas Total", self.natural_gas_total)
                print("Ammonia Produced", self.ammonia_produced*3)
            self.done = True


    
            
        
        info = self._get_info()
        
        return [self.sun_power, self.price, self.storage, self.moles, self.natural_gas_price], r, self.done

    def _get_info(self):
        return {
            "distance": 3
            
        }
    
    def reward(self, P_grid):
        
        reward = 0
        if P_grid >=0:
            paid_price = - P_grid*self.price * 1
            self.cost = paid_price
        else:
            paid_price = P_grid*self.price * 0.2

            
            # paid_price = P_grid*self.price * 0.5
            # 8 cent/kwh * 1000 * 0.08

        # pen
        if self.pv_generation > 0 and self.storage == 0:
            reward -= 100
      
        # 1 mmbtu 0,0278 kg natural gas
        price_natural_gas = self.natural_gas * 0.0278 * self.natural_gas_price
        self.cost -= price_natural_gas
        # 0.00681
        price_ammonia = self.ammonia * 0.00681
        self.profit = self.ammonia * 0.00681
        # price_ammonia = self.ammonia * 0.00681 * 10
        too_much = np.minimum(0, self.ammonia_total - self.ammonia_produced)

        over_under_production = np.abs(self.ammonia_total - self.ammonia_produced) 
        too_much_penalty = -1 * over_under_production if self.ammonia_produced > self.ammonia_total else 0
        
        #too_less_penalty = -0.01 * over_under_production if self.ammonia_produced < self.ammonia_total else 0

        
        reward += paid_price - price_natural_gas + price_ammonia + self.hydrogen

        return reward

    def reset(self):
      
     
        if self.eval:
            self.random_day = 37945
            self.ammonia_total = 10000 * NUM_TIME_STEPS * 0.5
        else:
            
            #self.random_day = np.maximum(0,37945 - NUM_TIME_STEPS - random.randint(0,1553))  
            
            self.random_day = np.maximum(0,37944 - NUM_TIME_STEPS - 24 * random.randint(0,1581))
            self.ammonia_total = 10000 * NUM_TIME_STEPS * 0.5
         
        self.sun_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,1]
        self.sun_power = self.sun_powers[self.random_day]

        self.wind_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,7]
        self.wind_power = self.wind_powers[self.random_day]

        self.prices = pd.read_csv('data/environment/prices/data.csv', header=0,delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,5]
        self.price = self.prices[self.random_day]

        self.dates = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,0]
        self.date = self.dates[self.random_day]

        date = datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S')
        date = date.strftime('%Y-%m-%d')
     
        self.natural_gas_prices = pd.read_csv('data/environment/gas/data.csv', header=0, delimiter=';')
        self.natural_gas_prices['Date'] = pd.to_datetime(self.natural_gas_prices['Date'])  
      
        row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == date];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=1)).strftime("%Y-%m-%d")];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=2)).strftime("%Y-%m-%d")];       
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=3)).strftime("%Y-%m-%d")];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=4)).strftime("%Y-%m-%d")];
       
        
        self.natural_gas_price = row['Price'].iloc[0]  
        self.storage = INITIAL_STORAGE
        self.pv_generation = 0
        self.wind_generation = 0
        self.natural_gas = 0
        self.time = 0
        self.hydrogen = 0
        self.hydrogen_total = 0
        self.gas_consumption = 0
        self.done = False
        self.ammonia = 0
        self.ammonia_produced = 0
        self.soc = 0
        self.profit = 0
        self.moles = 0
        self.cost = 0
        self.natural_gas_total = 0
        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()
        self.paid_price = 0
        
    
        
        return [self.sun_power, self.price, self.storage, self.moles, self.natural_gas_price] 

    def windkraft_ertrag(self, windgeschwindigkeit):
        rho = 1.225 # Luftdichte in kg/m^3
        A = 100 # Fläche der Rotorblätter in m^2
        Cp = 0.45 # Leistungsbeiwert
        lambd = 1.0 # Spitzenausnutzungsgrad
        ertrag = 0.5 * rho * A * Cp * lambd * (windgeschwindigkeit**3)
        return ertrag
