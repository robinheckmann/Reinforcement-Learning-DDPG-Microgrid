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

writer = SummaryWriter()


class EMS:
    def __init__(self, dynamic=False, eval=False, pv_gen=True, wind_gen=False):
                
        self.eval = eval
        self.dynamic = dynamic
        self.pv_gen = pv_gen
        self.wind_gen = wind_gen

        self.electrolyzer = Electrolyzer()


        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()
        
        if self.eval:
            #self.random_day = random.randint(43825, 52608 - NUM_TIME_STEPS)
            self.random_day = 37946
        else:
            #self.random_day = random.randint(0, 37945 - NUM_TIME_STEPS)
            range = 37945 - NUM_TIME_STEPS

            self.random_day = random.randint(0, 37945 - NUM_TIME_STEPS)
            #self.random_day = 37945 - 8760


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
        self.power_from_grid = 0 
        self.time = 0
        self.gas_consumption = 0
        self.pv_generation = 0
        self.wind_generation = 0
        self.ammonia = 0
        self.ammonia_total = 0
        self.cost = 0
        self.paid_price = 0
        self.done = False
 


    def step(self, action):
    

        ##############
        # RENEWABLES #
        ##############

        if self.pv_gen:
            pv_generation = self.sun_power
        else:
            pv_generation = 0 

        self.pv_generation = pv_generation
        self.wind_generation = self.windkraft_ertrag(self.wind_power) * 1

        ###################
        # BATTERY STORAGE #
        ###################

        if action[1] >= 0:        
            Pbattery = np.minimum(action[1] * C_MAX, (STORAGE_CAPACITY - self.storage) / ETA)
            if not np.isnan(Pbattery):
                self.storage += int(Pbattery * ETA)
        else:           
            Pbattery = np.maximum(action[1] * D_MAX, (MIN_STORAGE - self.storage) * ETA)
            if not np.isnan(Pbattery):
                self.storage += int(Pbattery / ETA)

        ################
        # ELECTROLYZER #
        ################

        
        self.ammonia = random.uniform(141400, 145000)                           # hydrogen consumotion (moles)
        self.ammonia_total += self.ammonia
        hydrogen_needed = self.ammonia * 0.667                                  # (mol)

        #Pel = np.minimum(action[0] * ELECTROLYZER_POWER, np.maximum(0,- Pbattery + pv_generation + self.wind_generation + ELECTROLYZER_POWER * 0.01))
        Pel = action[0] * ELECTROLYZER_POWER
        self.hydrogen, Wcomp, P_tank, self.moles = self.electrolyzer.run(Pel)         # (kg), (W)
       
        
        PTank = np.minimum(1 * hydrogen_needed, self.electrolyzer.get_moles())  # (mol)
        self.electrolyzer.consume_moles_from_tank(PTank)                        # (mol)
        hydrogen_from_natural_gas_mol = (hydrogen_needed - PTank)                   # (mol)

        # Calculate the volume of natural gas needed to produce the required amount of hydrogen
        # (1.17 m3 of natural gas is needed to produce 1 mol of hydrogen)
        self.natural_gas = hydrogen_from_natural_gas_mol * 0.0224 * 1.17        # (m³)



        self.power_from_grid = Pel + Pbattery - pv_generation - self.wind_generation
        self.power_from_grid *= 1e-6  

        # calculate rewards
        r = self.reward(self.power_from_grid)
        
        # tensorboard scalars
        writer.add_scalar('States/Hydrogen Storage', self.moles, self.time)    
        writer.add_scalar('States/Price', self.price, self.time)
        writer.add_scalar('States/External Energy Source/Power From Grid', self.power_from_grid, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas Consumption', self.natural_gas, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas Price', self.natural_gas_price, self.time)
        writer.add_scalar('States/External Energy Source/Wind Generation', self.wind_generation, self.time)
        writer.add_scalar('Actions/PV Generation', pv_generation * 1e-6, self.time)
        writer.add_scalar('Actions/Electrolyzer', Pel, self.time)
        writer.add_scalar('Actions/Storage', self.storage, self.time)
        writer.add_scalar('DDPG/Electrolyzer', action[0], self.time)
        writer.add_scalar('DDPG/Battery', action[1], self.time)
        writer.add_scalar('DDPG/Ammonia', self.ammonia, self.time)
        
        
      
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

        if self.time >= NUM_TIME_STEPS:
            self.done = True
            self.cost = self.ammonia_total/self.cost
         

        return [self.sun_power, self.price, self.storage, self.natural_gas_price, self.ammonia, self.wind_generation], r, self.done 


    def reward(self, P_grid):
        
        if P_grid >=0:
            paid_price = - P_grid*self.price      # buy mal 2
            self.cost += P_grid*self.price   
        else:
            paid_price = P_grid*self.price * 1.5  # sell 1.5


           
       
        sell_green_energy_penalty = STORAGE_CAPACITY - self.storage
        # 1 mmbtu zu m3 erdgas
        price_natural_gas = self.natural_gas / 28.32 * self.natural_gas_price
        
        self.cost += self.natural_gas / 28.32 * self.natural_gas_price 
              
       
        reward = paid_price - price_natural_gas + self.hydrogen


        return reward

    def reset(self):
      
        if self.eval:
            self.random_day = 37946
        else:
            self.random_day = random.randint(0, 37945 - NUM_TIME_STEPS)

        ## Resetting parameters

        
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
        
        self.storage = 0
        self.pv_generation = 0
        self.wind_generation = 0
        self.natural_gas = 0
        self.time = 0
        self.hydrogen = 0
        self.gas_consumption = 0
        self.done = False
        self.ammonia = 0
        self.ammonia_total = 0
        self.soc = 0
        self.cost = 0
        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()
        self.paid_price = 0
        

        return [self.sun_power, self.price, self.storage, self.natural_gas_price, self.ammonia, self.wind_generation] 

    def windkraft_ertrag(self, windgeschwindigkeit):
        rho = 1.225 # Luftdichte in kg/m^3
        A = 100 # Fläche der Rotorblätter in m^2
        Cp = 0.45 # Leistungsbeiwert
        lambd = 1.0 # Spitzenausnutzungsgrad
        ertrag = 0.5 * rho * A * Cp * lambd * (windgeschwindigkeit**3)
        return ertrag
