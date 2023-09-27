from environment import EMS
from matplotlib import style
style.use('ggplot')
from vars import *
from itertools import count
import pickle as pkl
import os
import argparse
import torch
import pandas as pd
import numpy as np
from train_ddpg import train_ddpg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='DDPG')
    parser.add_argument("--dynamic", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=True,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model_type", default='DDPG')
    parser.add_argument("--noisy", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt, model_name, dynamic, soft, eval, model_type, noisy):

    if not eval:
        train_ddpg(ckpt, model_name, dynamic, noisy)
            
    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0
            brain.add_noise = False
            env = EMS(dynamic=True, eval=True)
            state = env.reset()    
            hydrogen_produced = [env.hydrogen]
            storage_state = [env.storage]
            prices = [env.price]
            power_from_grid = [env.power_from_grid]
            sun_power = [env.sun_power]
            wind_power = [env.wind_generation]
            moles = [env.moles]
            natural_gas = [env.natural_gas]
            dates = [env.date]
            pv_generation = [env.pv_generation]
            natural_gas_prices = [env.natural_gas_price]
            gas_consumptions = [env.gas_consumption]
            ammonia_produced = [env.ammonia]
            action1 = [0]
            action2 = [0]
            action3 = [0]
            actions = [[0,0,0]]
            rewards = [0]
            print('Starting evaluation of the model')
            
            state = torch.tensor(state, dtype=torch.float).to(device)
            # Normalizing data using an online algo
            brain.normalizer.observe(state)
            state = brain.normalizer.normalize(state).unsqueeze(0) 
                  
            for t_episode in range(NUM_TIME_STEPS):
                action = brain.select_action(state).type(torch.FloatTensor)
                prices.append(env.price) # Will be replaced with environment price in price branch

                action1.append(action[0].numpy())
                action2.append(action[1].numpy())
                action3.append(action[2].numpy())
                
                actions.append(action.numpy())
                next_state, reward, done = env.step(action.numpy())
                rewards.append(reward)
                moles.append(env.moles)
                pv_generation.append(env.pv_generation)
                natural_gas_prices.append(env.natural_gas_price)
                gas_consumptions.append(env.gas_consumption)
                hydrogen_produced.append(env.hydrogen)
                ammonia_produced.append(env.ammonia)
                natural_gas.append(env.natural_gas)
                storage_state.append(env.storage)
                sun_power.append(env.sun_power)
                wind_power.append(env.wind_generation)
                dates.append(env.date)
                power_from_grid.append(env.power_from_grid)
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                    # normalize data using an online algo
                    brain.normalizer.observe(next_state)
                    next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
                else:
                    next_state = None
                # Move to the next state
                state = next_state

            eval_data = pd.DataFrame()

            eval_data['Actions'] = actions
            eval_data['Rewards'] = rewards


            eval_data['Action1'] = action1
            eval_data['Action2'] = action2
            eval_data['Action3'] = action3


            eval_data['PV Generation'] = pv_generation
            eval_data['Datetime'] = dates
            eval_data['Gas Consumption'] = gas_consumptions
            eval_data['Prices'] = prices
            eval_data['Ammonia'] = ammonia_produced
            eval_data['Prices Natural gas'] = natural_gas_prices
            eval_data['Moles'] = moles
            
            eval_data['Storage'] = storage_state
            eval_data['Power'] = power_from_grid
            eval_data['Sun Power'] = sun_power
            eval_data['Hydrogen'] = hydrogen_produced
            eval_data['Wind Power'] = wind_power
            eval_data['Natural Gas'] = natural_gas
            with open(os.getcwd() + '/data/output/' + model_type + '/' + model_name + '_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            print('Finished evaluation!')
            print('evaluating the policy...')

            ammonia = 0
            battery_actions = []
            electrolyzer_action = []
            grid_prices = []
            battery_levels = [] 
            sun_powers = []
            
            times = []
            
            actions = []
            

            gas_price = 0
            gas_prices = []
            moles = 0
            battery_level = 0
            gas_price = 0
            sun_power = 0
            wind_generation = 0
            ammonia_total = 0
            action1 = []
            action2 = []
            action3 = []
            action4 = []


            
            for sun_power in np.arange(0,100000000,10000000):
                print("Sun Power", sun_power)
                for price in np.arange(-10,50,1):
                    for battery_level in np.arange(-10,50,1):
                        for moles in np.arange(-10,50,1):
                       
                            state = [sun_power, price, battery_level, moles, gas_price, ammonia, wind_generation, ammonia_total] #, time
                            
                            
                            state = torch.tensor(state, dtype=torch.float).to(device)
                            state = brain.normalizer.normalize(state).unsqueeze(0)
                            
                            action = brain.select_action(state).type(torch.FloatTensor).numpy()
                                  
                            action1.append(action[0])
                            action2.append(action[1])
                            action3.append(action[2])
                            action4.append(action[3])
                            
                            grid_prices.append(price)
                            actions.append(action)
                            
                            battery_levels.append(battery_level)
                            
                            sun_powers.append(sun_power)
                            gas_prices.append(gas_price)

            eval_data = pd.DataFrame()

            eval_data['Actions'] = actions
            eval_data['Grid Prices'] = grid_prices


            eval_data['Battery Level'] = battery_levels
            eval_data['action1'] = action1
            eval_data['action2'] = action2
            eval_data['action3'] = action3
            eval_data['action4'] = action4


            
           

            with open(os.getcwd() + '/data/output/' + model_type + '/' + model_name + '_policy_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

        else:
            print('If no training should be performed, then please choose a model that should be evaluated')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))