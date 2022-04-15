"""
Run a sweep over a host of hyper parameters in an attempt to tune the results
"""

import itertools

import numpy as np

def write_h_params_to_file():
    pass

def launch_training():
    pass

def create_combinations():

    # number of nodes in a layer
    nodes = [150]

    # learning rate
    lr = [0.001,0.0005,0.0001]

    # discount factor
    gamma = [0.9]

    # drop out
    drop_out = [0.0]

    # crashing reward
    reward_crash = [-100.0]
    # success_reward
    reward_success =[100.0]
    # reward closing distance to destination
    reward_dist = [1.0]
    # reward for being near an obstacle
    reward_prox = [-0.1,-1.0,-5.0]
    # reward norm for fuel
    reward_fuel = [10.0]
    # reward normalization for decreasing angle to destination
    reward_angle = [0.05,0.1]
    # reward normalization for swerving around the obstacle
    reward_obs = [100.0,200.0,500.0]

    info = [nodes,lr,gamma,drop_out,reward_crash,reward_success,reward_dist,reward_prox,reward_fuel,reward_angle,reward_obs]
    combinations = list(itertools.product(*info))

    length =len(combinations)
    print(length)


if __name__ == '__main__':

    create_combinations()