
# native python libraries
from abc import ABC, abstractmethod
from collections import OrderedDict
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 3rd party libraries
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

# own libraries
from Learn_Algorithms import DQN, DDPG
from Movers import SailBoat, StaticCircle, UtilityBoat
from Sensors import Lidar


class Environment(ABC):

    def __init__(self):

        self.h_params = None  # dictionary of hyperparameters the setup and define the training
        self.mover_dict = OrderedDict()  # orderd dictionary of movers that are in the simulation
        self.learning_agent = None  # the learning algorithm that acts on the movers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trial_num = None # interger for the trial number of the training

    def check_for_trial(self,trial_num):
        """
        checks if there is an existing trial. Prompts the user if they want to overwrite the trial or cancel if a trial
        exists
        :param trial_num: integer of the trial number
        :return:
        """
        self.trial_num = trial_num
        try:
            os.mkdir('Output\\Trial_' + str(trial_num))
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\episodes')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\episodes\\graph')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\episodes\\data')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\baseline')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\baseline\\graph')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\baseline\\data')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\models')
            os.mkdir('Output\\Trial_' + str(trial_num) + '\\history')
        except FileExistsError:
            inp = input('Trial already exists. Continueing will overwrite the data. Continue? [y/n]')
            if inp != 'y':
                sys.exit()

    @abstractmethod
    def run_episode(self):
        pass

    @abstractmethod
    def run_baseline(self):
        pass

    @staticmethod
    def load_hyper_params(file_name):
        """
        Given a yaml file that has configuration details for the environment and hyperparameters for the agents,
        a dictionary is built and returned to the user

        :param file_name:
        :return:
        """
        with open(file_name, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @abstractmethod
    def write_history(self):
        pass

    #@abstractmethod
    #def animate_episode(self):
    #    pass

    @abstractmethod
    def reset(self):
        pass


class NoObstacles(Environment, ABC):

    def __init__(self):

        super().__init__()
        self.history = pd.DataFrame()
        self.old = dict()

    def launch_training(self, trial_num, input_file):
        """
        launches and runs the training for the two_obstacle scenario
        :return:
        """

        # check the trial number is acceptable and notify user if overwriting old trials
        self.check_for_trial(trial_num)

        # load in the hyperparameters for the training
        self.h_params = self.load_hyper_params(input_file)
        self.write_hyper_params(trial_num)

        # get the mover
        self.create_boat()

        # create learning algorithm
        self.get_learning_algorithm()
        # set the close memory buffer to none
        self.learning_agent.memory_close = None

        # set episode history header
        action_size = self.learning_agent.action_size
        if self.h_params['learning_agent']['type'] == 'DDPG':
            q_header = ['q_val_0']
        else:
            q_header = ['q_val_' + str(i) for i in range(action_size)]
        header = ['time', 'reward', 'done', 'is_crashed', 'is_reached'] + q_header + ['action']
        self.header = header

        # run the training
        episode_number = 0
        while episode_number <= self.h_params['scenario']['num_episodes']:

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])), len(header))),
                                        columns=header)

            is_baseline = False
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(episode_number,is_baseline)

            # write history out to
            self.write_history(episode_number)

            # render single episode
            self.render_episode(episode_number)

            # train the policy network
            self.learning_agent.train_agent()

            # run a baseline episode to measure progress over time
            if episode_number % self.h_params['scenario']['baseline_frequency'] == 0:
                self.run_baseline(episode_number)

            # update the target network
            if episode_number > 0 and episode_number % self.h_params['learning_agent']['update_target'] == 0:
                self.learning_agent.update_target_network()
                # save the interim network
                torch.save(self.learning_agent.policy_network.state_dict(),
                           'Output\\Trial_' + str(trial_num) + '\\models\\actor' + str(episode_number) + '.pymdl')
                #torch.save(self.learning_agent.actor_policy_net.state_dict(),'Output\\Trial_' + str(trial_num) + '\\models\\actor' + str(episode_number) + '.pymdl')
                #torch.save(self.learning_agent.critic_net.state_dict(),'Output\\Trial_' + str(trial_num) + '\\models\\critic' + str(episode_number) + '.pymdl')

            #
            print('Episode {} out of {} episodes: Reward = {:0.3f}: Crashed = {}: Success = {}: Far Length {}'.format(episode_number,self.h_params['scenario']['num_episodes'],cumulative_reward.cpu().detach().numpy()[0][0],is_crashed,is_destination_reached,len(self.learning_agent.memory_far)))

            episode_number += 1

        # render episodes and overall information about the training
        self.render_training(trial_num)

    def create_boat(self):
        """
        creates the boat mover and adds it to the mover list
        :return:
        """

        boat_h_params = self.h_params['boat']

        boat = None
        boat_name = None
        if boat_h_params['type'] == 'sail_boat':
            boat_name = 'sail_boat_1'
            boat = SailBoat.challenger(boat_name,self.h_params['scenario']['time_step'])
        elif boat_h_params['type'] == 'utility':
            # load in a utility boat
            boat_name = 'utility_boat_1'
            boat = UtilityBoat.get_default('utility_boat_1',self.h_params['scenario']['time_step'], self.h_params['boat']['mode'])
        else:
            raise ValueError('Not a valid boat type designation')

        self.mover_dict[boat.name] = boat

    def get_learning_algorithm(self):
        """
        gets the learning algorithm, to train the agent
        :return:
        """

        alg_type = self.h_params['learning_agent']['type']

        agent = None
        if alg_type == 'DQN':
            # use DQN to train the agent
            state_size, action_size = self.get_state_and_action_size('discrete')
            agent = DQN(state_size, action_size, self.h_params, self.device)
        elif alg_type == 'DDPG':
            # use DDPG to train the agent
            state_size, action_size = self.get_state_and_action_size('continous')
            agent = DDPG(state_size, action_size, self.h_params, self.device)
        else:
            raise ValueError('The learning algorithm {} is not supported'.format(alg_type))

        self.learning_agent = agent

    def get_state_and_action_size(self,action_type):
        # generate the state size
        state_size = 0
        action_size = 0
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                state_size += len(mover.get_state(False))
                action_size += mover.get_action_size(action_type)

                # check the sensors measurements for the state
                for sensor in mover.sensors:
                    state_size += len(sensor.get_state())
        return state_size, action_size

    def get_normalized_state(self, mover):
        """
        gets the state vector in its normalized state
        :return:
        """
        return mover.normalize_state()

    def convert_numpy_to_tensor(self,arr):
        """
        converts a numpy array to a tensor for use with pytorch
        :param arr:
        :return:
        """
        tmp = torch.tensor([arr], device=self.device,dtype=torch.float)
        return tmp.view(tmp.size(), -1)

    def run_episode(self,ep_num,is_baseline):
        """
        runs one complete episode of for the simulation. The history of the simulation is saved for graphing and for
        making animations if desired.
        :return:
        """

        # reset the state of the simulation. Place the obstacles, destination, and boat at random locations and
        # orientations
        if not is_baseline:
            self.reset()

        delta_t = self.h_params['scenario']['time_step']  # time step to march the simulation forward in
        t = 0  # current time of the simulation
        max_t = self.h_params['scenario']['max_time']  # maximum time to run episode
        step_num = 0  # number of steps taken over the episode
        # copy initial state to episode history

        # reset the history of the movers
        for name, mover in self.mover_dict.items():
            mover.reset_history(int(np.ceil(max_t/delta_t)))

        # simulation loop
        done = False
        cumulative_reward = 0
        while t < max_t and done == False:

            # step
            state, action,  new_state, reward, done, is_crashed, is_reached_dest, q_vals = self.step(ep_num, t, max_t,is_baseline)

            # add to memory
            if not is_baseline and t > 0.001:
                # if boat is close to an obstacle put it in memory_close, else in memory far

                self.learning_agent.memory_far.push(state, action, new_state, reward, done)

            # add to history of
            for name, mover in self.mover_dict.items():
                mover.add_step_history(step_num)

            # epsiode history
            q_vals = np.reshape(q_vals,(len(q_vals[0]),))
            if isinstance(action,np.ndarray):
                telemetry = np.concatenate(([t, reward.cpu().detach().numpy()[0][0], done, is_crashed, is_reached_dest],
                                            q_vals, action))
            else:
                telemetry = np.concatenate(([t,reward.cpu().detach().numpy()[0][0],done,is_crashed,is_reached_dest],q_vals,action.cpu().detach().numpy()[0]))
            self.history.iloc[step_num] = telemetry
            cumulative_reward += reward

            # increase time
            t += delta_t
            step_num += 1

        # trim the histories
        self.history.drop(range(step_num,len(self.history)),inplace=True)
        for name, mover in self.mover_dict.items():
            mover.trim_history(step_num)

        return is_crashed, is_reached_dest, cumulative_reward

    def step(self,n_steps, t, max_t, is_baseline):
        """
        preform one time step of the simulation
        :param - the current number of steps for the overall training
        :return:
        """

        # sensors are assumed to be up to date from either the last step or from the reset action

        # step each of the movers
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                # this is the boats

                # Trud for getting the state that includes the sensors
                norm_state = mover.normalize_state(True)
                norm_state_tensor = self.convert_numpy_to_tensor(list(norm_state.values()))

                # build the state for the mover

                # returns the action, and all of the q vals
                if is_baseline:
                    action, q_vals = self.learning_agent.get_action_eval(norm_state_tensor)
                else:
                    action, q_vals = self.learning_agent.get_action_train(norm_state_tensor, n_steps, self.bg_noise )

                if self.h_params['learning_agent']['type'] == 'DDPG':
                    #tmp_power = mover.power + action[1]
                    action_np = action.cpu().detach().numpy()
                    mover.set_control(mover.power + action_np[0][1], mover.delta + action_np[0][0])
                else:
                    delta_adj, power_adj = mover.action_to_command(action.cpu().detach().numpy()[0][0])
                    if self.h_params['boat']['type'] == 'utility':
                        mover.set_control(mover.power + power_adj, mover.delta + delta_adj)
                    else:
                        # this says power but it is the sail angle
                        mover.set_control(mover.beta + power_adj, mover.delta + delta_adj)

                self.old['mu'] = mover.mu
                self.old['dest_dist'] = mover.destination_distance
                if self.h_params['boat']['type'] == 'utility':
                    self.old['fuel'] = mover.fuel
                self.old['phi'] = mover.phi
                self.old['x'] = mover.x
                self.old['y'] = mover.y
                for sensor in mover.sensors:
                    tmp_state = sensor.get_state()
                    self.old['theta_obs_1'] = tmp_state['angle_center_0']

            mover.step()

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

        # get state prime or new state
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                new_norm_state = mover.normalize_state(True)
                new_norm_state_tensor = self.convert_numpy_to_tensor(list(new_norm_state.values()))

        # get reward
        reward, done, is_crashed, is_reached_dest = self.reward(t, max_t)
        reward_tensor = self.convert_numpy_to_tensor([reward])

        # set done-ness for next state
        z = np.zeros(self.learning_agent.state_size)
        if done:
            new_norm_state_tensor = torch.tensor([z], device=self.device,
                                                 dtype=torch.float)  # .view(self.state_size,1,-1)
        new_norm_state_tensor = new_norm_state_tensor.view(new_norm_state_tensor.size(), -1)

        done = np.float(done)

        return norm_state_tensor, action,  new_norm_state_tensor, reward_tensor, done, is_crashed, is_reached_dest, q_vals

    def run_baseline(self,ep_num):
        """
        runs a set of baseline episodes were no random actions are used and
        :return:
        """

        # position_x
        x = [0,0,0,110,110,110]
        # position_y
        y = [30,50,70,30,50,70]
        # destination
        dest = [150,50]
        #obstacles
        obs_1 = [70,50,20.0]
        #power of the propller
        wind = np.pi/2.0

        # sail angle
        beta = 0.0

        for i, x_tmp in enumerate(x):

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])),len(self.header))), columns=self.header)

            self.reset()

            # set destination and intial orientation
            for name, mover in self.mover_dict.items():
                if 'boat' in mover.name:
                    # set destination
                    mover.destination = dest

                    mover.x = x[i]
                    mover.y = y[i]

                    mover.phi = 0
                    mover.delta = 0
                    mover.gamma = wind
                    mover.beta = beta

            # negative 1 is given as the episode number as a arbitrary place holder
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(-1,True)

            # write the history of the baseline episode with a number for
            self.write_history(str(ep_num)+'-'+str(i),True)
        #

    def reset(self):
        """
        resets the state of the boats and the simulations. Random locations for the obstacles, boat, and destinations
        are all drawn to initialize the simulation
        :return:
        """

        # domain
        domain = self.h_params['scenario']['domain']

        # set boat state variables

        # destination location
        x3 = np.random.random() * domain
        y3 = np.random.random() * domain

        # current position
        x4 = np.random.random() * domain
        y4 = np.random.random() * domain
        dist_34 = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3))
        while dist_34 < 3.0:
            x4 = np.random.random() * domain
            y4 = np.random.random() * domain
            dist_34 = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3))

        for name, mover in self.mover_dict.items():
            if 'boat' in mover.name:
                # set destination
                mover.destination = [x3, y3]
                # set the power of the simulation
                if self.h_params['boat']['type'] == 'utility':
                    if self.h_params['boat']['mode'] == 'propeller_only':
                        mover.power = 10000.0 # 10k [kilo-watt]
                    elif self.h_params['boat']['mode'] == 'propeller_and_power':
                        mover.power = np.random.random() * 20000.0
                    # add random amount of initial fuel
                    mover.fuel = 1.0 + np.random.random()  # 1-2.0 kilos of fuel
                else:
                    # set wind angle
                    #mover.gamma = np.random.random() * np.pi * 2.0
                    mover.gamma = np.pi/2.0
                    # set wind velocity
                    mover.v_wind = 10.0 # 10 [m/s]
                # set propeller or rudder angle
                mover.delta = (np.random.random()-0.5)*2.0*np.pi
                # set position of the boat itself
                mover.x = x4
                mover.y = y4
                # set hull angle:
                mover.phi = np.random.random() * np.pi * 2.0
                # set initial velocity
                v_boat = np.random.random() * 2.0
                mover.v_boat_x = v_boat * np.cos(mover.phi)
                mover.v_boat_y = v_boat * np.sin(mover.phi)
                # reset the acceleration to zero
                mover.a_x = 0.0
                mover.a_y = 0.0
                mover.a_boat_theta = 0.0
                # calculate distance metrics
                mover.calc_dest_metrics()
                if self.h_params['boat']['type'] == 'utility':
                    # bound the hull angle
                    mover.bound_hull_angle()
                else:
                    mover.check_angles()

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

        # reset noise for DDPG
        self.bg_noise = np.zeros(self.learning_agent.action_size)

    def reward(self, t, max_t):
        """
        calculates the reward for the given state of the system. The reward denotes how good or bad the reward is.
        :param t:
        :param max_t:
        :return:
        """
        done = False

        reward = 0.0

        # get the boat
        for key, mover in self.mover_dict.items():
            if 'boat' in key:
                boat = mover

        # add reward for better angle
        boat.calc_dest_metrics()

        # add negative reward for traveling backwards longitudinally
        if boat.v_boat_x_p < 0:
            reward -= 0.1

        # reward for closing the distance to the destination
        if boat.destination_distance < self.old['dest_dist'] and boat.v_boat_x_p > 0:
            reward += (self.old['dest_dist'] - boat.destination_distance) * (
                    self.old['dest_dist'] - boat.destination_distance)/self.h_params['reward']['distance_norm']

        # extra reward for closeness to destination
        if boat.destination_distance <= 2.01:
            reward += self.h_params['reward']['success']
        # elif self.boat.destination_distance <= 10.0:
        #    reward += self.h_params['reward_prox']


        # add negative reward for running into an obstacle
        is_crashed = False

        # if the boat has reached the destination
        is_reached_dest = False
        if boat.destination_distance <= 2.0:
            is_reached_dest = True

        if t >= max_t - 1 or boat.destination_distance < 2.0 or boat.destination_distance >= 300.0:
            # stop simulation at max time, when it reaches its destination, or it runs into an obstacle
            done = True
        return reward, done, is_crashed, is_reached_dest

    def reward_angle_to_obstacle(self, boat, obs_1, reward):

        buffer = 5.0

        # calculate distance to obstacles
        dist_to_1 = np.sqrt(
            (boat.x - obs_1.pos[0]) * (boat.x - obs_1.pos[0]) + (boat.y - obs_1.pos[1]) * (boat.y - obs_1.pos[1]))

        # reward for angle to destination and angle to obstacles
        unit_x = np.cos(boat.phi)
        unit_y = np.sin(boat.phi)
        slope = unit_y / unit_x
        slope_obs = -unit_x / unit_y
        b = boat.y - slope * boat.x
        # obstacle 1
        b_obs_1 = obs_1.pos[1] - slope_obs * obs_1.pos[0]
        x_obs_1_int = (b_obs_1 - b) / (slope - slope_obs)
        y_obs_1_int = slope_obs * x_obs_1_int + b_obs_1
        close_dist_obs_1 = np.sqrt(
            (x_obs_1_int - obs_1.pos[0]) * (x_obs_1_int - obs_1.pos[0]) + (y_obs_1_int - obs_1.pos[1]) * (
                        y_obs_1_int - obs_1.pos[1]))
        angle_to_intersect_1 = np.arctan2((y_obs_1_int - boat.y) ,(x_obs_1_int - boat.x))

        is_consider = False
        if close_dist_obs_1 <= (obs_1.radius+buffer):
            # boat's current line instersects with the obstacle
            if boat.phi - np.pi/2.0 <= angle_to_intersect_1 <= boat.phi + np.pi/2.0:
                # obstacle is infront of the boat, not behind
                if boat.destination_distance > (dist_to_1 - (obs_1.radius+buffer)):
                    # need to consider obstacle. gain reward for increassing angle to obstacle
                    is_consider = True

                    for sensor in boat.sensors:
                        tmp_state = sensor.get_state()
                        theta = tmp_state['angle_center_0']

                    # correct theat
                    theta_old = self.old['theta_obs_1']
                    if theta_old > np.pi:
                        theta_old -= 2.0*np.pi
                    if theta > np.pi:
                        theta -= 2.0*np.pi

                    # correct phi
                    phi_old = self.old['phi']
                    if phi_old > np.pi:
                        phi_old -= 2.0*np.pi
                    phi = boat.phi
                    if phi > np.pi:
                        phi -= 2.0*np.pi

                    x_old = self.old['x']
                    y_old = self.old['y']
                    x_phi = np.cos(self.old['phi'])
                    y_phi = np.sin(self.old['phi'])
                    x_theta = np.cos(self.old['theta_obs_1'])
                    y_theta = np.sin(self.old['theta_obs_1'])
                    top = (x_phi-x_old)*(x_theta-x_old)+(y_phi-y_old)*(y_theta-y_old)
                    bottom = np.sqrt((x_phi-x_old)*(x_phi-x_old) + (y_phi-y_old)*(y_phi-y_old)) * np.sqrt((x_theta-x_old)*(x_theta-x_old) + (y_theta-y_old)*(y_theta-y_old))
                    delta_old = np.arccos(top/bottom)

                    x_new = boat.x
                    y_new = boat.y
                    x_phi = np.cos(boat.phi)
                    y_phi = np.sin(boat.phi)
                    x_theta = np.cos(theta)
                    y_theta = np.sin(theta)
                    top = (x_phi - x_new) * (x_theta - x_new) + (y_phi - y_new) * (y_theta - y_new)
                    bottom = np.sqrt((x_phi - x_new) * (x_phi - x_new) + (y_phi - y_new) * (y_phi - y_new)) * np.sqrt(
                        (x_theta - x_new) * (x_theta - x_new) + (y_theta - y_new) * (y_theta - y_new))
                    delta_new = np.arccos(top/bottom)
                    delta_theta_obs = delta_new - delta_old

                    sign_delta = np.sign(delta_theta_obs)
                    r_slope = 10.0/15.0
                    r_int = 1
                    if np.rad2deg(delta_theta_obs) > 10.0:
                        tmp_reward = 10.0
                    else:
                        tmp_reward = r_slope*np.rad2deg(delta_theta_obs)+r_int
                    reward += tmp_reward*sign_delta

                    #reward += delta_theta_obs * self.h_params['reward']['obs']

        if not is_consider:
            # gain reward for reducing angle to destination
            #if boat.v_boat_x_p > 0 and (np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0)):
            if np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0):
                reward += (np.pi - abs(boat.mu)) * (np.pi - abs(boat.mu)) * self.h_params['reward']['angle']  # normalize reward to bound it

        return reward

    def write_history(self, episode_number,is_baseline=False):
        """
        writes the telemetry for the episode out to a csv file for later processing
        :param episode_number: integer for the episode number
        :param is_baseline: boolean for if the episode is a base line episode or a training episode
        :return:
        """

        # combine all of the histories together
        total_history = self.history
        for _, mover in self.mover_dict.items():
            total_history = pd.concat([total_history,mover.history], axis=1)

        if is_baseline:
            # the baseline episode is placed into a different folder
            total_history.to_csv(
                'Output\\Trial_' + str(self.trial_num) + '\\baseline\\data\\' + str(episode_number) + '.csv',index=False)
        else:
            total_history.to_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_number)+'.csv',index=False)

    def write_hyper_params(self, trial_num):
        """
        make a copy of the hyperparameters and save them to the output directory for saving later
        :return:
        """
        with open('Output\\Trial_' + str(trial_num)+'\\hyper_parameters.yml', 'w') as file:
            yaml.safe_dump(self.h_params,file)

    def render_episode(self,episode_num):

        df = pd.read_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_num)+'.csv')

        # graph episode
        sns.set_theme()
        fig = plt.figure(0, figsize=(14, 12))
        gs = GridSpec(5, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        ax4 = fig.add_subplot(gs[2, 2])
        ax5 = fig.add_subplot(gs[2, 3])
        ax6 = fig.add_subplot(gs[3, 2])
        ax7 = fig.add_subplot(gs[3,3])
        ax8 = fig.add_subplot(gs[4, 0])
        ax9 = fig.add_subplot(gs[4, 1])
        ax10 = fig.add_subplot(gs[4, 2])
        ax11 = fig.add_subplot(gs[4, 3])

        # graph trajectory
        circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[0]), radius=2.0, alpha=1.0,
                                color='tab:green')
        ax1.add_patch(circle)

        sc = ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none')
        plt.colorbar(sc, ax=ax1)
        # get the points for each 20% of the trajectory
        spacing = int(len(df) / 4)
        idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
        for j in range(len(idx)):
            ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

        #ax1.legend()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')

        # graph the q values
        if self.h_params['boat']['type'] == 'utility':
            if self.h_params['boat']['mode'] == 'propeller_and_power':

                if self.h_params['learning_agent']['type'] == 'DQN':
                    select_q = np.zeros(len(df))
                    for i in range(len(select_q)):
                        select_q[i] = df['q_val_' + str(int(df['action'].iloc[i]))].iloc[i]
                    ax2.plot(df['time'], select_q)
                    ax2.set_ylabel('q value')
                    ax2.set_xlabel('time [s]')
                elif self.h_params['learning_agent']['type'] == 'DDPG':


                    qv = [ 1 if 'q_val' in item else 0 for item in self.history.columns ]
                    qv = np.sum(qv)

                    for i in range(qv):

                        ax2.plot(df['time'], df['q_val_'+str(i)])
                    ax2.set_xlabel('time [s]')
                    ax2.set_ylabel('critic value')
                #ax2.legend()

            else:
                action_size = self.learning_agent.action_size
                for j in range(action_size):
                    ax2.plot(df['time'], df['q_val_' + str(j)], label=str(j))
                ax2.set_xlabel('time [s]')
                ax2.set_ylabel('q value')
                ax2.legend()
        else:
            select_q = np.zeros(len(df))
            for i in range(len(select_q)):
                select_q[i] = df['q_val_' + str(int(df['action'].iloc[i]))].iloc[i]
            ax2.plot(df['time'], select_q)
            ax2.set_ylabel('q value')
            ax2.set_xlabel('time [s]')

        # graph angles
        ax3.plot(df['time'], np.rad2deg(df['phi']), label='hull_angle')
        ax3.plot(df['time'], np.rad2deg(df['delta']), label='rudder angle')
        ax3.plot(df['time'], np.rad2deg(df['mu']), label='hull to dest')
        ax3.plot(df['time'], np.rad2deg(df['theta']), label='to dest')
        ax3.plot(df['time'], np.rad2deg(df['gamma']), label='wind')
        ax3.plot(df['time'], np.rad2deg(df['beta']), label='sail')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angles [deg]')
        ax3.legend()

        # graph velocities
        #ax4.plot(df['time'], df['v_boat_x'], label='x')
        #ax4.plot(df['time'], df['v_boat_y'], label='y')
        ax4.plot(df['time'], df['v_boat_x_p'], label='x_p')
        ax4.plot(df['time'], df['v_boat_theta'], label='theta')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Velocity')
        ax4.legend()

        # graph power
        #ax5.plot(df['time'],df['power'])
        #ax5.set_xlabel('Time [s]')
        #ax5.set_ylabel('Power [watt]')

        #
        ax6.plot(df['time'], df['v_boat_x'], label='x')
        ax6.plot(df['time'], df['v_boat_y'], label='y')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Velocity')
        ax6.legend()

        # fuel
        #ax7.plot(df['time'], df['fuel'])
        #ax7.set_xlabel('Time [s]')
        #ax7.set_ylabel('Fuel [kg]')

        # graph the reward
        ax8.plot(df['time'], df['reward'])
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Reward')

        # graph distance to the obstacles
        #ax9.plot(df['time'], df['dist_center_0'])
        #ax9.set_xlabel('Time [s]')
        #ax9.set_ylabel('Dist to obstacle [m]')

        # angle to the obstacles
        #ax10.plot(df['time'], np.rad2deg(df['angle_center_0']))
        #ax10.set_xlabel('Time [s]')
        #ax10.set_ylabel('Angle to obstacle [deg]')

        # distance to the destination
        ax11.plot(df['time'], df['destination_distance'])
        ax11.plot([0, max(df['time'])], [2.0, 2.0])
        ax11.set_xlabel('Time [s]')
        ax11.set_ylabel('Dist to dest [m]')

        # tidy up the graph
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\episodes\\graph\\' + str(episode_num) + '.png')
        plt.close('all')

    def render_training(self, trial_num):

        # render each episode
        reward = np.zeros(self.h_params['scenario']['num_episodes'])
        is_crashed = np.zeros_like(reward)
        is_reached = np.zeros_like(reward)
        min_dist = np.zeros_like(reward)
        for i in range(self.h_params['scenario']['num_episodes']):
            df = pd.read_csv('Output\\Trial_'+str(trial_num)+'\\episodes\\data\\'+str(i)+'.csv')



            # gather data for trend across the training
            reward[i] = np.sum(df['reward'])
            is_crashed[i] = max(df['is_crashed'])
            is_reached[i] = max(df['is_reached'])
            min_dist[i] = min(df['destination_distance'])

        # overall training graphs
        reward_df = pd.DataFrame()
        fig = plt.figure(0,figsize=(14,8))
        windows = [1,51,101,201]
        #windows = [1, 3, 5]
        avg = np.zeros_like(reward)
        for window in windows:
            avg, alpha = self.moving_average(reward,window)
            reward_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])],avg,label='window='+str(window),alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward.png')
        reward_df.to_csv('Output\\Trial_'+str(trial_num)+'\\history\\Reward.csv')
        plt.close('all')

        is_crashed_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_crashed)
        for window in windows:
            avg, alpha = self.moving_average(is_crashed, window)
            is_crashed_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Crash Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.png')
        is_crashed_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.csv')
        plt.close('all')

        is_reached_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_reached)
        for window in windows:
            avg, alpha = self.moving_average(is_reached, window)
            is_reached_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Success Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.png')
        is_reached_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.csv')
        plt.close('all')

        min_dist_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(min_dist)
        for window in windows:
            avg, alpha = self.moving_average(min_dist, window)
            min_dist_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Minimum Distance to Destination [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Min_dist.png')
        min_dist_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Min_Dist.csv')
        plt.close('all')

        # baseline graphs

        baseline_data_path = 'Output\\Trial_'+str(trial_num)+'\\baseline\\data'
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(baseline_data_path) if isfile(join(baseline_data_path, f))]
        first_char = onlyfiles[0][0]
        num_iters = 0
        for i in range(len(onlyfiles)):
            if onlyfiles[i][0] == first_char:
                num_iters += 1
            else:
                break

        base_eps = []
        reward = []
        tmp_reward = []
        is_crashed = []
        tmp_crashed = []
        is_destination = []
        tmp_destination = []
        min_dist =[]
        tmp_min_dist = []
        baseline_number = 0
        #while baseline_number <= len(onlyfiles):
        #for file in onlyfiles:
        while baseline_number < self.h_params['scenario']['num_episodes']:

            for i in range(num_iters):
                df = pd.read_csv(baseline_data_path + '\\' + str(baseline_number)+'-'+str(i)+'.csv')

                if i == 0:
                    fig = plt.figure(0, figsize=(14, 8))
                    # gs = GridSpec(5, 4, figure=fig)
                    ax1 = fig.add_subplot(111)

                    circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[1]), radius=2.0,
                                            alpha=1.0,
                                            color='tab:green')
                    ax1.add_patch(circle)
                    base_eps.append(baseline_number)
                tmp_reward.append(np.sum(df['reward']))
                tmp_crashed.append(np.max(df['is_crashed']))
                tmp_destination.append(np.max(df['is_reached']))
                tmp_min_dist.append(np.min(df['destination_distance']))

                label = str(i)
                ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none', label=label)
                # get the points for each 20% of the trajectory
                spacing = int(len(df) / 4)
                idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
                for j in range(len(idx)):
                    ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

            plt.tight_layout()
            plt.savefig('Output\\Trial_' + str(trial_num) + '\\baseline\\graph\\' + str(baseline_number) + '.png')
            plt.close('all')

            reward.append([min(tmp_reward), np.mean(tmp_reward), np.max(tmp_reward)])
            tmp_reward = []
            is_crashed.append([min(tmp_crashed), np.mean(tmp_crashed), np.max(tmp_crashed)])
            tmp_crashed = []
            is_destination.append([min(tmp_destination), np.mean(tmp_destination), np.max(tmp_destination)])
            tmp_destination = []
            min_dist.append([min(tmp_min_dist), np.mean(tmp_min_dist), np.max(tmp_min_dist)])
            tmp_min_dist = []

            baseline_number += self.h_params['scenario']['baseline_frequency']

        # graph trends in the baseline over time
        plt.close('all')
        reward = np.reshape(reward,(len(reward),3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps,reward[:,0],label='min')
        plt.plot(base_eps, reward[:, 1], label='avg')
        plt.plot(base_eps, reward[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward_Baseline.png')
        plt.close('all')

        is_crashed = np.reshape(is_crashed, (len(is_crashed), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_crashed[:, 0], label='min')
        plt.plot(base_eps, is_crashed[:, 1], label='avg')
        plt.plot(base_eps, is_crashed[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Crash Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Crash_Rate_Baseline.png')
        plt.close('all')

        is_destination = np.reshape(is_destination, (len(is_destination), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_destination[:, 0], label='min')
        plt.plot(base_eps, is_destination[:, 1], label='avg')
        plt.plot(base_eps, is_destination[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Success_Rate_Baseline.png')
        plt.close('all')

        min_dist = np.reshape(min_dist, (len(min_dist), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, min_dist[:, 0], label='min')
        plt.plot(base_eps, min_dist[:, 1], label='avg')
        plt.plot(base_eps, min_dist[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Minimum Distance [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Minimum_Distance_Baseline.png')
        plt.close('all')

    def moving_average(self,arr,window):
        """
        given an array, a moving average is calculated given a window to average over. The ends hold value to nearest
        possible average
        :param arr:
        :param window:
        :return:
        """
        avg = np.zeros_like(arr)
        half_window = int(window / 2.0)
        if half_window == 0:
            avg = arr
            alpha = 0.3
        else:
            alpha = 1.0
            for i in range(len(arr)):
                if i < half_window:
                    avg[i] = np.sum(arr[0:window]) / window
                elif i > (len(arr) - half_window):
                    avg[i] = np.sum(arr[len(arr) - window:len(arr)]) / window
                else:
                    avg[i] = np.sum(arr[i - half_window:i + half_window + 1]) / window
        return avg, alpha


class OneObstacles(Environment, ABC):

    def __init__(self):

        super().__init__()
        self.history = pd.DataFrame()
        self.old = dict()

    def launch_training(self, trial_num, input_file):
        """
        launches and runs the training for the two_obstacle scenario
        :return:
        """

        # check the trial number is acceptable and notify user if overwriting old trials
        self.check_for_trial(trial_num)

        # load in the hyperparameters for the training
        self.h_params = self.load_hyper_params(input_file)
        self.write_hyper_params(trial_num)

        # create obstacle movers
        self.create_obstacles()

        # get the mover
        self.create_boat()

        # create learning algorithm
        self.get_learning_algorithm()

        # set episode history header
        action_size = self.learning_agent.action_size
        if self.h_params['learning_agent']['type'] == 'DDPG':
            q_header = ['q_val_0']
        else:
            q_header = ['q_val_' + str(i) for i in range(action_size)]
        a_header = ['action_'+str(i) for i in range(action_size)]
        header = ['time', 'reward', 'done', 'is_crashed', 'is_reached'] + q_header + a_header
        self.header = header

        # run the training
        episode_number = 0
        while episode_number <= self.h_params['scenario']['num_episodes']:

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])), len(header))),
                                        columns=header)

            is_baseline = False
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(episode_number,is_baseline)

            # write history out to
            self.write_history(episode_number)

            # render single episode
            self.render_episode(episode_number)

            # train the policy network
            self.learning_agent.train_agent()

            # run a baseline episode to measure progress over time
            if episode_number % self.h_params['scenario']['baseline_frequency'] == 0:
                self.run_baseline(episode_number)

            # update the target network
            if episode_number > 0 and episode_number % self.h_params['learning_agent']['update_target'] == 0:
                self.learning_agent.update_target_network()
                # save the interim network
                torch.save(self.learning_agent.actor_policy_net.state_dict(),'Output\\Trial_' + str(trial_num) + '\\models\\actor' + str(episode_number) + '.pymdl')
                torch.save(self.learning_agent.critic_net.state_dict(),
                           'Output\\Trial_' + str(trial_num) + '\\models\\critic' + str(episode_number) + '.pymdl')

            #
            print('Episode {} out of {} episodes: Reward = {:0.3f}: Crashed = {}: Success = {}: Far Length {}: Close Length {}'.format(episode_number,self.h_params['scenario']['num_episodes'],cumulative_reward.cpu().detach().numpy()[0][0],is_crashed,is_destination_reached,len(self.learning_agent.memory_far),len(self.learning_agent.memory_close)))

            episode_number += 1

        # render episodes and overall information about the training
        self.render_training(trial_num)

    def create_obstacles(self):
        """
        Create the obstacle and adds them to the mover list
        :return:
        """

        # obstacle 1
        obs_1 = StaticCircle('obs_1',50)
        self.mover_dict[obs_1.name] = obs_1

    def create_boat(self):
        """
        creates the boat mover and adds it to the mover list
        :return:
        """

        boat_h_params = self.h_params['boat']

        boat = None
        boat_name = None
        if boat_h_params['type'] == 'sail_boat':
            boat = SailBoat.get_default(self.h_params['scenario']['time_step'])
        elif boat_h_params['type'] == 'utility':
            # load in a utility boat
            boat_name = 'utility_boat_1'
            boat = UtilityBoat.get_default('utility_boat_1',self.h_params['scenario']['time_step'], self.h_params['boat']['mode'])
        else:
            raise ValueError('Not a valid boat type designation')

        # add lidar to the boat
        if boat_h_params['sensor'] == 'lidar':
            lidar = Lidar(boat_name, 1, 100.0)
            boat.add_sensor(lidar)
        else:
            raise ValueError('Only lidar is the currently implimented sensor')

        self.mover_dict[boat.name] = boat

    def get_learning_algorithm(self):
        """
        gets the learning algorithm, to train the agent
        :return:
        """

        alg_type = self.h_params['learning_agent']['type']

        agent = None
        if alg_type == 'DQN':
            # use DQN to train the agent
            state_size, action_size = self.get_state_and_action_size('discrete')
            agent = DQN(state_size, action_size, self.h_params, self.device)
        elif alg_type == 'DDPG':
            # use DDPG to train the agent
            state_size, action_size = self.get_state_and_action_size('continous')
            agent = DDPG(state_size, action_size, self.h_params, self.device)
        else:
            raise ValueError('The learning algorithm {} is not supported'.format(alg_type))

        self.learning_agent = agent

    def get_state_and_action_size(self,action_type):
        # generate the state size
        state_size = 0
        action_size = 0
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                state_size += len(mover.get_state(False))
                action_size += mover.get_action_size(action_type)

                # check the sensors measurements for the state
                for sensor in mover.sensors:
                    state_size += len(sensor.get_state())
        return state_size, action_size

    def get_normalized_state(self, mover):
        """
        gets the state vector in its normalized state
        :return:
        """
        return mover.normalize_state()

    def convert_numpy_to_tensor(self,arr):
        """
        converts a numpy array to a tensor for use with pytorch
        :param arr:
        :return:
        """
        tmp = torch.tensor([arr], device=self.device,dtype=torch.float)
        return tmp.view(tmp.size(), -1)

    def run_episode(self,ep_num,is_baseline):
        """
        runs one complete episode of for the simulation. The history of the simulation is saved for graphing and for
        making animations if desired.
        :return:
        """

        # reset the state of the simulation. Place the obstacles, destination, and boat at random locations and
        # orientations
        if not is_baseline:
            self.reset()

        delta_t = self.h_params['scenario']['time_step']  # time step to march the simulation forward in
        t = 0  # current time of the simulation
        max_t = self.h_params['scenario']['max_time']  # maximum time to run episode
        step_num = 0  # number of steps taken over the episode
        # copy initial state to episode history

        # reset the history of the movers
        for name, mover in self.mover_dict.items():
            mover.reset_history(int(np.ceil(max_t/delta_t)))

        # simulation loop
        done = False
        cumulative_reward = 0
        while t < max_t and done == False:

            # step
            state, action,  new_state, reward, done, is_crashed, is_reached_dest, q_vals = self.step(ep_num, t, max_t,is_baseline)

            # add to memory
            if not is_baseline and t > 0.001:
                # if boat is close to an obstacle put it in memory_close, else in memory far
                dist_to_1 = None
                obs_1 = None
                for name, mover in self.mover_dict.items():
                    if 'boat' in name:
                        for sensor in mover.sensors:
                            tmp_obs = sensor.get_state()
                            dist_to_1 = tmp_obs['dist_center_0']
                    if 'obs_1' in name:
                        obs_1 = mover
                if (dist_to_1-obs_1.radius) <= 10.0:
                    self.learning_agent.memory_close.push(state, action, new_state, reward, done)
                else:
                    self.learning_agent.memory_far.push(state, action, new_state, reward, done)
                #self.learning_agent.memory.push(state, action, new_state, reward)

            # add to history of
            for name, mover in self.mover_dict.items():
                mover.add_step_history(step_num)

            # epsiode history
            q_vals = np.reshape(q_vals,(len(q_vals[0]),))
            if isinstance(action,np.ndarray):
                telemetry = np.concatenate(([t, reward.cpu().detach().numpy()[0][0], done, is_crashed, is_reached_dest],
                                            q_vals, action))
            else:
                telemetry = np.concatenate(([t,reward.cpu().detach().numpy()[0][0],done,is_crashed,is_reached_dest],q_vals,action.cpu().detach().numpy()[0]))
            self.history.iloc[step_num] = telemetry
            cumulative_reward += reward

            # increase time
            t += delta_t
            step_num += 1

        # trim the histories
        self.history.drop(range(step_num,len(self.history)),inplace=True)
        for name, mover in self.mover_dict.items():
            mover.trim_history(step_num)

        return is_crashed, is_reached_dest, cumulative_reward

    def step(self,n_steps, t, max_t, is_baseline):
        """
        preform one time step of the simulation
        :param - the current number of steps for the overall training
        :return:
        """

        # sensors are assumed to be up to date from either the last step or from the reset action

        # step each of the movers
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                # this is the boats

                # Trud for getting the state that includes the sensors
                norm_state = mover.normalize_state(True)
                norm_state_tensor = self.convert_numpy_to_tensor(list(norm_state.values()))

                # build the state for the mover

                # returns the action, and all of the q vals
                if is_baseline:
                    action, q_vals = self.learning_agent.get_action_eval(norm_state_tensor)
                else:
                    action, q_vals = self.learning_agent.get_action_train(norm_state_tensor, n_steps, self.bg_noise )

                if self.h_params['learning_agent']['type'] == 'DDPG':
                    #tmp_power = mover.power + action[1]
                    action_np = action.cpu().detach().numpy()
                    mover.set_control(mover.power + action_np[0][1], mover.delta + action_np[0][0])
                else:
                    delta_adj, power_adj = mover.action_to_command(action.cpu().detach().numpy()[0][0])
                    mover.set_control(mover.power + power_adj, mover.delta + delta_adj)

                self.old['mu'] = mover.mu
                self.old['dest_dist'] = mover.destination_distance
                self.old['fuel'] = mover.fuel
                self.old['phi'] = mover.phi
                self.old['x'] = mover.x
                self.old['y'] = mover.y
                for sensor in mover.sensors:
                    tmp_state = sensor.get_state()
                    self.old['theta_obs_1'] = tmp_state['angle_center_0']

            mover.step()

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

        # get state prime or new state
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                new_norm_state = mover.normalize_state(True)
                new_norm_state_tensor = self.convert_numpy_to_tensor(list(new_norm_state.values()))

        # get reward
        reward, done, is_crashed, is_reached_dest = self.reward(t, max_t)
        reward_tensor = self.convert_numpy_to_tensor([reward])

        # set done-ness for next state
        z = np.zeros(self.learning_agent.state_size)
        if done:
            new_norm_state_tensor = torch.tensor([z], device=self.device,
                                                 dtype=torch.float)  # .view(self.state_size,1,-1)
        new_norm_state_tensor = new_norm_state_tensor.view(new_norm_state_tensor.size(), -1)

        done = np.float(done)

        return norm_state_tensor, action,  new_norm_state_tensor, reward_tensor, done, is_crashed, is_reached_dest, q_vals

    def run_baseline(self,ep_num):
        """
        runs a set of baseline episodes were no random actions are used and
        :return:
        """

        # position_x
        x = [0,0,0,110,110,110]
        # position_y
        y = [30,50,70,30,50,70]
        # destination
        dest = [150,50]
        #obstacles
        obs_1 = [70,50,20.0]
        #power of the propller
        power = 5000.0

        for i, x_tmp in enumerate(x):

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])),len(self.header))), columns=self.header)

            self.reset()

            # set obstacle 1 location
            for name, mover in self.mover_dict.items():
                if mover.name == 'obs_1':
                    mover.pos = [obs_1[0], obs_1[1]]
                    mover.radius = obs_1[2]
                    break

            # set destination and intial orientation
            for name, mover in self.mover_dict.items():
                if 'boat' in mover.name:
                    # set destination
                    mover.destination = dest

                    mover.x = x[i]
                    mover.y = y[i]

                    mover.phi = 0
                    mover.delta = 0
                    mover.power = power

            # negative 1 is given as the episode number as a arbitrary place holder
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(-1,True)

            # write the history of the baseline episode with a number for
            self.write_history(str(ep_num)+'-'+str(i),True)
        #

    def reset(self):
        """
        resets the state of the boats and the simulations. Random locations for the obstacles, boat, and destinations
        are all drawn to initialize the simulation
        :return:
        """

        # domain
        domain = self.h_params['scenario']['domain']

        # set the location of obstacle one
        x1 = np.random.random() * domain
        y1 = np.random.random() * domain
        r1 = 20.0 # [m]
        for name, mover in self.mover_dict.items():
            if mover.name == 'obs_1':
                mover.pos = [x1, y1]
                mover.radius = r1
                break

        # set boat state variables

        # destination location
        x3 = np.random.random() * domain
        y3 = np.random.random() * domain
        dist_13 = np.sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1))
        while dist_13 < r1 + 10.0 :
            x3 = np.random.random() * domain
            y3 = np.random.random() * domain
            dist_13 = np.sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1))

        # current position
        x4 = np.random.random() * domain
        y4 = np.random.random() * domain
        dist_14 = np.sqrt((x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1))
        dist_34 = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3))
        while dist_14 < r1 + 1.0 and dist_34 > 3.0:
            x4 = np.random.random() * domain
            y4 = np.random.random() * domain
            dist_14 = np.sqrt((x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1))

        for name, mover in self.mover_dict.items():
            if 'boat' in mover.name:
                # set destination
                mover.destination = [x3, y3]
                # set the power of the simulation
                if self.h_params['boat']['mode'] == 'propeller_only':
                    mover.power = 10000.0 # 10k [kilo-watt]
                elif self.h_params['boat']['mode'] == 'propeller_and_power':
                    mover.power = np.random.random() * 20000.0
                # set propeller angle
                mover.delta = (np.random.random()-0.5)*2.0*np.pi
                # set position of the boat itself
                mover.x = x4
                mover.y = y4
                # set hull angle:
                mover.phi = np.random.random() * np.pi * 2.0
                # set initial velocity
                v_boat = np.random.random() * 2.0
                mover.v_boat_x = v_boat * np.cos(mover.phi)
                mover.v_boat_y = v_boat * np.sin(mover.phi)
                # reset the acceleration to zero
                mover.a_x = 0.0
                mover.a_y = 0.0
                mover.a_boat_theta = 0.0
                # calculate distance metrics
                mover.calc_dest_metrics()
                # bound the hull angle
                mover.bound_hull_angle()
                # add random amount of initial fuel
                mover.fuel = 1.0 + np.random.random()  # 1-2.0 kilos of fuel

                # reset the old dictionary
                #self.old['fuel'] = mover.fuel
                #self.old['mu'] = mover.mu
                #self.old['dest_dist'] = mover.destination_distance

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

        # reset noise for DDPG
        self.bg_noise = np.zeros(self.learning_agent.action_size)

    def reward(self, t, max_t):
        """
        calculates the reward for the given state of the system. The reward denotes how good or bad the reward is.
        :param t:
        :param max_t:
        :return:
        """
        done = False

        reward = 0.0

        # get the boat, obstacle 1, and obstacle 2
        for key, mover in self.mover_dict.items():
            if 'boat' in key:
                boat = mover
            elif '_1' in key:
                obs_1 = mover
            elif '_2' in key:
                obs_2 = mover

        # calculate distance to obstacles
        dist_to_1 = np.sqrt(
            (boat.x - obs_1.pos[0]) * (boat.x - obs_1.pos[0]) + (boat.y - obs_1.pos[1]) * (
                        boat.y - obs_1.pos[1]))

        # add negative reward for fuel usage
        reward += (boat.fuel - self.old['fuel'])*self.h_params['reward']['fuel_use']

        # add reward for better angle
        boat.calc_dest_metrics()

        # reward for closing the distance to the destination
        if boat.destination_distance < self.old['dest_dist'] and boat.v_boat_x_p > 0:
            reward += (self.old['dest_dist'] - boat.destination_distance) * (
                    self.old['dest_dist'] - boat.destination_distance)/self.h_params['reward']['distance_norm']

        # extra reward for closeness to destination
        if boat.destination_distance <= 2.01:
            reward += self.h_params['reward']['success']
        # elif self.boat.destination_distance <= 10.0:
        #    reward += self.h_params['reward_prox']


        # add negative reward for being close to the obstacles
        reward = self.reward_angle_to_obstacle(boat, obs_1, reward)

        if (dist_to_1-obs_1.radius) <= (obs_1.radius+10.0):
            reward += -1.0/(self.h_params['reward']['prox_crash']*(dist_to_1-obs_1.radius)+0.1)

        # add negative reward for running into an obstacle
        is_crashed = False
        if dist_to_1 <= obs_1.radius:
            is_crashed = True
            reward += self.h_params['reward']['crash']  # 50.0

        # if the boat has reached the destination
        is_reached_dest = False
        if boat.destination_distance <= 2.0:
            is_reached_dest = True

        if t >= max_t - 1 or dist_to_1 < obs_1.radius or boat.destination_distance < 2.0 or boat.destination_distance >= 300.0:
            # stop simulation at max time, when it reaches its destination, or it runs into an obstacle
            done = True
        return reward, done, is_crashed, is_reached_dest

    def reward_angle_to_obstacle(self, boat, obs_1, reward):

        buffer = 5.0

        # calculate distance to obstacles
        dist_to_1 = np.sqrt(
            (boat.x - obs_1.pos[0]) * (boat.x - obs_1.pos[0]) + (boat.y - obs_1.pos[1]) * (boat.y - obs_1.pos[1]))

        # reward for angle to destination and angle to obstacles
        unit_x = np.cos(boat.phi)
        unit_y = np.sin(boat.phi)
        slope = unit_y / unit_x
        slope_obs = -unit_x / unit_y
        b = boat.y - slope * boat.x
        # obstacle 1
        b_obs_1 = obs_1.pos[1] - slope_obs * obs_1.pos[0]
        x_obs_1_int = (b_obs_1 - b) / (slope - slope_obs)
        y_obs_1_int = slope_obs * x_obs_1_int + b_obs_1
        close_dist_obs_1 = np.sqrt(
            (x_obs_1_int - obs_1.pos[0]) * (x_obs_1_int - obs_1.pos[0]) + (y_obs_1_int - obs_1.pos[1]) * (
                        y_obs_1_int - obs_1.pos[1]))
        angle_to_intersect_1 = np.arctan2((y_obs_1_int - boat.y) ,(x_obs_1_int - boat.x))

        is_consider = False
        if close_dist_obs_1 <= (obs_1.radius+buffer):
            # boat's current line instersects with the obstacle
            if boat.phi - np.pi/2.0 <= angle_to_intersect_1 <= boat.phi + np.pi/2.0:
                # obstacle is infront of the boat, not behind
                if boat.destination_distance > (dist_to_1 - (obs_1.radius+buffer)):
                    # need to consider obstacle. gain reward for increassing angle to obstacle
                    is_consider = True

                    for sensor in boat.sensors:
                        tmp_state = sensor.get_state()
                        theta = tmp_state['angle_center_0']

                    # correct theat
                    theta_old = self.old['theta_obs_1']
                    if theta_old > np.pi:
                        theta_old -= 2.0*np.pi
                    if theta > np.pi:
                        theta -= 2.0*np.pi

                    # correct phi
                    phi_old = self.old['phi']
                    if phi_old > np.pi:
                        phi_old -= 2.0*np.pi
                    phi = boat.phi
                    if phi > np.pi:
                        phi -= 2.0*np.pi

                    x_old = self.old['x']
                    y_old = self.old['y']
                    x_phi = np.cos(self.old['phi'])
                    y_phi = np.sin(self.old['phi'])
                    x_theta = np.cos(self.old['theta_obs_1'])
                    y_theta = np.sin(self.old['theta_obs_1'])
                    top = (x_phi-x_old)*(x_theta-x_old)+(y_phi-y_old)*(y_theta-y_old)
                    bottom = np.sqrt((x_phi-x_old)*(x_phi-x_old) + (y_phi-y_old)*(y_phi-y_old)) * np.sqrt((x_theta-x_old)*(x_theta-x_old) + (y_theta-y_old)*(y_theta-y_old))
                    delta_old = np.arccos(top/bottom)

                    x_new = boat.x
                    y_new = boat.y
                    x_phi = np.cos(boat.phi)
                    y_phi = np.sin(boat.phi)
                    x_theta = np.cos(theta)
                    y_theta = np.sin(theta)
                    top = (x_phi - x_new) * (x_theta - x_new) + (y_phi - y_new) * (y_theta - y_new)
                    bottom = np.sqrt((x_phi - x_new) * (x_phi - x_new) + (y_phi - y_new) * (y_phi - y_new)) * np.sqrt(
                        (x_theta - x_new) * (x_theta - x_new) + (y_theta - y_new) * (y_theta - y_new))
                    delta_new = np.arccos(top/bottom)
                    delta_theta_obs = delta_new - delta_old

                    sign_delta = np.sign(delta_theta_obs)
                    r_slope = 10.0/15.0
                    r_int = 1
                    if np.rad2deg(delta_theta_obs) > 10.0:
                        tmp_reward = 10.0
                    else:
                        tmp_reward = r_slope*np.rad2deg(delta_theta_obs)+r_int
                    reward += tmp_reward*sign_delta

                    #reward += delta_theta_obs * self.h_params['reward']['obs']

        if not is_consider:
            # gain reward for reducing angle to destination
            #if boat.v_boat_x_p > 0 and (np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0)):
            if np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0):
                reward += (np.pi - abs(boat.mu)) * (np.pi - abs(boat.mu)) * self.h_params['reward']['angle']  # normalize reward to bound it

        return reward

    def write_history(self, episode_number,is_baseline=False):
        """
        writes the telemetry for the episode out to a csv file for later processing
        :param episode_number: integer for the episode number
        :param is_baseline: boolean for if the episode is a base line episode or a training episode
        :return:
        """

        # combine all of the histories together
        total_history = self.history
        for _, mover in self.mover_dict.items():
            total_history = pd.concat([total_history,mover.history], axis=1)

        if is_baseline:
            # the baseline episode is placed into a different folder
            total_history.to_csv(
                'Output\\Trial_' + str(self.trial_num) + '\\baseline\\data\\' + str(episode_number) + '.csv',index=False)
        else:
            total_history.to_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_number)+'.csv',index=False)

    def write_hyper_params(self, trial_num):
        """
        make a copy of the hyperparameters and save them to the output directory for saving later
        :return:
        """
        with open('Output\\Trial_' + str(trial_num)+'\\hyper_parameters.yml', 'w') as file:
            yaml.safe_dump(self.h_params,file)

    def render_episode(self,episode_num):

        df = pd.read_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_num)+'.csv')

        # graph episode
        sns.set_theme()
        fig = plt.figure(0, figsize=(14, 12))
        gs = GridSpec(5, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        ax4 = fig.add_subplot(gs[2, 2])
        ax5 = fig.add_subplot(gs[2, 3])
        ax6 = fig.add_subplot(gs[3, 2])
        ax7 = fig.add_subplot(gs[3,3])
        ax8 = fig.add_subplot(gs[4, 0])
        ax9 = fig.add_subplot(gs[4, 1])
        ax10 = fig.add_subplot(gs[4, 2])
        ax11 = fig.add_subplot(gs[4, 3])

        # graph trajectory
        circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[0]), radius=2.0, alpha=1.0,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]), radius=df['radius_obs_1'].iloc[0],
                                alpha=1.0,
                                color='tab:blue')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]), radius=df['radius_obs_1'].iloc[0]+10.0,
                                alpha=0.2,
                                color='tab:blue')
        ax1.add_patch(circle)
        sc = ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none')
        plt.colorbar(sc, ax=ax1)
        # get the points for each 20% of the trajectory
        spacing = int(len(df) / 4)
        idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
        for j in range(len(idx)):
            ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

        #ax1.legend()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')

        # graph the q values
        if self.h_params['boat']['mode'] == 'propeller_and_power':

            if self.h_params['learning_agent']['type'] == 'DQN':
                select_q = np.zeros(len(df))
                for i in range(len(select_q)):
                    select_q[i] = df['q_val_' + str(int(df['action'].iloc[i]))].iloc[i]
                ax2.plot(df['time'], select_q)
                ax2.set_ylabel('q value')
                ax2.set_xlabel('time [s]')
            elif self.h_params['learning_agent']['type'] == 'DDPG':


                qv = [ 1 if 'q_val' in item else 0 for item in self.history.columns ]
                qv = np.sum(qv)

                for i in range(qv):

                    ax2.plot(df['time'], df['q_val_'+str(i)])
                ax2.set_xlabel('time [s]')
                ax2.set_ylabel('critic value')
            #ax2.legend()

        else:
            action_size = self.learning_agent.action_size
            for j in range(action_size):
                ax2.plot(df['time'], df['q_val_' + str(j)], label=str(j))
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel('q value')
            ax2.legend()

        # graph angles
        ax3.plot(df['time'], np.rad2deg(df['phi']), label='hull_angle')
        ax3.plot(df['time'], np.rad2deg(df['delta']), label='prop angle')
        ax3.plot(df['time'], np.rad2deg(df['mu']), label='hull to dest')
        ax3.plot(df['time'], np.rad2deg(df['theta']), label='to dest')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angles [deg]')
        ax3.legend()

        # graph velocities
        #ax4.plot(df['time'], df['v_boat_x'], label='x')
        #ax4.plot(df['time'], df['v_boat_y'], label='y')
        ax4.plot(df['time'], df['v_boat_x_p'], label='x_p')
        ax4.plot(df['time'], df['v_boat_theta'], label='theta')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Velocity')
        ax4.legend()

        # graph power
        ax5.plot(df['time'],df['power'])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Power [watt]')

        #
        ax6.plot(df['time'], df['v_boat_x'], label='x')
        ax6.plot(df['time'], df['v_boat_y'], label='y')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Velocity')
        ax6.legend()

        # fuel
        ax7.plot(df['time'], df['fuel'])
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('Fuel [kg]')

        # graph the reward
        ax8.plot(df['time'], df['reward'])
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Reward')

        # graph distance to the obstacles
        ax9.plot(df['time'], df['dist_center_0'])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel('Dist to obstacle [m]')

        # angle to the obstacles
        ax10.plot(df['time'], np.rad2deg(df['angle_center_0']))
        ax10.set_xlabel('Time [s]')
        ax10.set_ylabel('Angle to obstacle [deg]')

        # distance to the destination
        ax11.plot(df['time'], df['destination_distance'])
        ax11.plot([0, max(df['time'])], [2.0, 2.0])
        ax11.set_xlabel('Time [s]')
        ax11.set_ylabel('Dist to dest [m]')

        # tidy up the graph
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\episodes\\graph\\' + str(episode_num) + '.png')
        plt.close('all')

    def render_training(self, trial_num):

        # render each episode
        reward = np.zeros(self.h_params['scenario']['num_episodes'])
        is_crashed = np.zeros_like(reward)
        is_reached = np.zeros_like(reward)
        min_dist = np.zeros_like(reward)
        for i in range(self.h_params['scenario']['num_episodes']):
            df = pd.read_csv('Output\\Trial_'+str(trial_num)+'\\episodes\\data\\'+str(i)+'.csv')



            # gather data for trend across the training
            reward[i] = np.sum(df['reward'])
            is_crashed[i] = max(df['is_crashed'])
            is_reached[i] = max(df['is_reached'])
            min_dist[i] = min(df['destination_distance'])

        # overall training graphs
        reward_df = pd.DataFrame()
        fig = plt.figure(0,figsize=(14,8))
        windows = [1,51,101,201]
        #windows = [1, 3, 5]
        avg = np.zeros_like(reward)
        for window in windows:
            avg, alpha = self.moving_average(reward,window)
            reward_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])],avg,label='window='+str(window),alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward.png')
        reward_df.to_csv('Output\\Trial_'+str(trial_num)+'\\history\\Reward.csv')
        plt.close('all')

        is_crashed_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_crashed)
        for window in windows:
            avg, alpha = self.moving_average(is_crashed, window)
            is_crashed_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Crash Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.png')
        is_crashed_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.csv')
        plt.close('all')

        is_reached_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_reached)
        for window in windows:
            avg, alpha = self.moving_average(is_reached, window)
            is_reached_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Success Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.png')
        is_reached_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.csv')
        plt.close('all')

        min_dist_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(min_dist)
        for window in windows:
            avg, alpha = self.moving_average(min_dist, window)
            min_dist_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Minimum Distance to Destination [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Min_dist.png')
        min_dist_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Min_Dist.csv')
        plt.close('all')

        # baseline graphs

        baseline_data_path = 'Output\\Trial_'+str(trial_num)+'\\baseline\\data'
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(baseline_data_path) if isfile(join(baseline_data_path, f))]
        first_char = onlyfiles[0][0]
        num_iters = 0
        for i in range(len(onlyfiles)):
            if onlyfiles[i][0] == first_char:
                num_iters += 1
            else:
                break

        base_eps = []
        reward = []
        tmp_reward = []
        is_crashed = []
        tmp_crashed = []
        is_destination = []
        tmp_destination = []
        min_dist =[]
        tmp_min_dist = []
        baseline_number = 0
        #while baseline_number <= len(onlyfiles):
        #for file in onlyfiles:
        while baseline_number < self.h_params['scenario']['num_episodes']:

            for i in range(num_iters):
                df = pd.read_csv(baseline_data_path + '\\' + str(baseline_number)+'-'+str(i)+'.csv')

                if i == 0:
                    fig = plt.figure(0, figsize=(14, 8))
                    # gs = GridSpec(5, 4, figure=fig)
                    ax1 = fig.add_subplot(111)

                    circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[1]), radius=2.0,
                                            alpha=1.0,
                                            color='tab:green')
                    ax1.add_patch(circle)
                    circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]),
                                            radius=df['radius_obs_1'].iloc[0],
                                            alpha=1.0,
                                            color='tab:blue')
                    ax1.add_patch(circle)
                    circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]),
                                            radius=df['radius_obs_1'].iloc[0]+10.0,
                                            alpha=0.2,
                                            color='tab:blue')
                    ax1.add_patch(circle)
                    base_eps.append(baseline_number)
                tmp_reward.append(np.sum(df['reward']))
                tmp_crashed.append(np.max(df['is_crashed']))
                tmp_destination.append(np.max(df['is_reached']))
                tmp_min_dist.append(np.min(df['destination_distance']))

                label = str(i)
                ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none', label=label)
                # get the points for each 20% of the trajectory
                spacing = int(len(df) / 4)
                idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
                for j in range(len(idx)):
                    ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

            plt.tight_layout()
            plt.savefig('Output\\Trial_' + str(trial_num) + '\\baseline\\graph\\' + str(baseline_number) + '.png')
            plt.close('all')

            reward.append([min(tmp_reward), np.mean(tmp_reward), np.max(tmp_reward)])
            tmp_reward = []
            is_crashed.append([min(tmp_crashed), np.mean(tmp_crashed), np.max(tmp_crashed)])
            tmp_crashed = []
            is_destination.append([min(tmp_destination), np.mean(tmp_destination), np.max(tmp_destination)])
            tmp_destination = []
            min_dist.append([min(tmp_min_dist), np.mean(tmp_min_dist), np.max(tmp_min_dist)])
            tmp_min_dist = []

            baseline_number += self.h_params['scenario']['baseline_frequency']

        # graph trends in the baseline over time
        plt.close('all')
        reward = np.reshape(reward,(len(reward),3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps,reward[:,0],label='min')
        plt.plot(base_eps, reward[:, 1], label='avg')
        plt.plot(base_eps, reward[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward_Baseline.png')
        plt.close('all')

        is_crashed = np.reshape(is_crashed, (len(is_crashed), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_crashed[:, 0], label='min')
        plt.plot(base_eps, is_crashed[:, 1], label='avg')
        plt.plot(base_eps, is_crashed[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Crash Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Crash_Rate_Baseline.png')
        plt.close('all')

        is_destination = np.reshape(is_destination, (len(is_destination), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_destination[:, 0], label='min')
        plt.plot(base_eps, is_destination[:, 1], label='avg')
        plt.plot(base_eps, is_destination[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Success_Rate_Baseline.png')
        plt.close('all')

        min_dist = np.reshape(min_dist, (len(min_dist), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, min_dist[:, 0], label='min')
        plt.plot(base_eps, min_dist[:, 1], label='avg')
        plt.plot(base_eps, min_dist[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Minimum Distance [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Minimum_Distance_Baseline.png')
        plt.close('all')

    def moving_average(self,arr,window):
        """
        given an array, a moving average is calculated given a window to average over. The ends hold value to nearest
        possible average
        :param arr:
        :param window:
        :return:
        """
        avg = np.zeros_like(arr)
        half_window = int(window / 2.0)
        if half_window == 0:
            avg = arr
            alpha = 0.3
        else:
            alpha = 1.0
            for i in range(len(arr)):
                if i < half_window:
                    avg[i] = np.sum(arr[0:window]) / window
                elif i > (len(arr) - half_window):
                    avg[i] = np.sum(arr[len(arr) - window:len(arr)]) / window
                else:
                    avg[i] = np.sum(arr[i - half_window:i + half_window + 1]) / window
        return avg, alpha


class TwoObstacles(Environment, ABC):

    def __init__(self):

        super().__init__()
        self.history = pd.DataFrame()
        self.old = dict()

    def launch_training(self, trial_num, input_file):
        """
        launches and runs the training for the two_obstacle scenario
        :return:
        """

        # check the trial number is acceptable and notify user if overwriting old trials
        self.check_for_trial(trial_num)

        # load in the hyperparameters for the training
        self.h_params = self.load_hyper_params(input_file)
        self.write_hyper_params(trial_num)

        # create obstacle movers
        self.create_obstacles()

        # get the mover
        self.create_boat()

        # create learning algorithm
        self.get_learning_algorithm()

        # set episode history header
        action_size = self.learning_agent.action_size
        if self.h_params['learning_agent']['type'] == 'DDPG':
            q_header = ['q_val_0']
        else:
            q_header = ['q_val_'+str(i) for i in range(action_size)]
        header = ['time', 'reward', 'done', 'is_crashed', 'is_reached'] + q_header + ['action']
        self.header = header

        # run the training
        episode_number = 0
        while episode_number <= self.h_params['scenario']['num_episodes']:

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])), len(header))),
                                        columns=header)

            is_baseline = False
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(episode_number,is_baseline)

            # write history out to
            self.write_history(episode_number)

            # render single episode
            self.render_episode(episode_number)

            # train the policy network
            self.learning_agent.train_agent()

            # run a baseline episode to measure progress over time
            if episode_number % self.h_params['scenario']['baseline_frequency'] == 0:
                self.run_baseline(episode_number)

            # update the target network
            if episode_number > 0 and episode_number % self.h_params['learning_agent']['update_target'] == 0:
                self.learning_agent.update_target_network()
                # save the interim network
                torch.save(self.learning_agent.target_network.state_dict(),'Output\\Trial_' + str(trial_num) + '\\models\\' + str(episode_number) + '.pymdl')

            #
            print('Episode {} out of {} episodes: Reward = {:0.3f}: Crashed = {}: Success = {}: Close Length {}'.format(episode_number,self.h_params['scenario']['num_episodes'],cumulative_reward.cpu().detach().numpy()[0][0],is_crashed,is_destination_reached,len(self.learning_agent.memory_close)))

            episode_number += 1

        # render episodes and overall information about the training
        self.render_training(trial_num)

    def create_obstacles(self):
        """
        Create the two obstacles and adds them to the mover list
        :return:
        """

        # obstacle 1
        obs_1 = StaticCircle('obs_1',50)
        self.mover_dict[obs_1.name] = obs_1

        # obstacle 2
        obs_2 = StaticCircle('obs_2', 50)
        self.mover_dict[obs_2.name] = obs_2

    def create_boat(self):
        """
        creates the boat mover and adds it to the mover list
        :return:
        """

        boat_h_params = self.h_params['boat']

        boat = None
        boat_name = None
        if boat_h_params['type'] == 'sail_boat':
            boat = SailBoat.get_default(self.h_params['scenario']['time_step'])
        elif boat_h_params['type'] == 'utility':
            # load in a utility boat
            boat_name = 'utility_boat_1'
            boat = UtilityBoat.get_default('utility_boat_1',self.h_params['scenario']['time_step'], self.h_params['boat']['mode'])
        else:
            raise ValueError('Not a valid boat type designation')

        # add lidar to the boat
        if boat_h_params['sensor'] == 'lidar':
            lidar = Lidar(boat_name, 2, 100.0)
            boat.add_sensor(lidar)
        else:
            raise ValueError('Only lidar is the currently implimented sensor')

        self.mover_dict[boat.name] = boat

    def get_learning_algorithm(self):
        """
        gets the learning algorithm, to train the agent
        :return:
        """

        alg_type = self.h_params['learning_agent']['type']

        agent = None
        if alg_type == 'DQN':
            # use DQN to train the agent
            state_size, action_size = self.get_state_and_action_size('discrete')
            agent = DQN(state_size, action_size, self.h_params, self.device)
        elif alg_type == 'DDPG':
            # use DDPG to train the agent
            state_size, action_size = self.get_state_and_action_size('continous')
            agent = DDPG(state_size,action_size,self.h_params,self.device)
        else:
            raise ValueError('The learning algorithm {} is not supported'.format(alg_type))

        self.learning_agent = agent

    def get_state_and_action_size(self,action_type):
        # generate the state size
        state_size = 0
        action_size = 0
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                state_size += len(mover.get_state(False))
                action_size += mover.get_action_size(action_type)

                # check the sensors measurements for the state
                for sensor in mover.sensors:
                    state_size += len(sensor.get_state())
        return state_size, action_size

    def get_normalized_state(self, mover):
        """
        gets the state vector in its normalized state
        :return:
        """
        return mover.normalize_state()

    def convert_numpy_to_tensor(self,arr):
        """
        converts a numpy array to a tensor for use with pytorch
        :param arr:
        :return:
        """
        tmp = torch.tensor([arr], device=self.device,dtype=torch.float)
        return tmp.view(tmp.size(), -1)

    def run_episode(self,ep_num,is_baseline):
        """
        runs one complete episode of for the simulation. The history of the simulation is saved for graphing and for
        making animations if desired.
        :return:
        """

        # reset the state of the simulation. Place the obstacles, destination, and boat at random locations and
        # orientations
        if not is_baseline:
            self.reset()

        delta_t = self.h_params['scenario']['time_step']  # time step to march the simulation forward in
        t = 0  # current time of the simulation
        max_t = self.h_params['scenario']['max_time']  # maximum time to run episode
        step_num = 0  # number of steps taken over the episode
        # copy initial state to episode history

        # reset the history of the movers
        for name, mover in self.mover_dict.items():
            mover.reset_history(int(np.ceil(max_t/delta_t)))

        # simulation loop
        done = False
        cumulative_reward = 0
        while t < max_t and done == False:

            # step
            state, action,  new_state, reward, done, is_crashed, is_reached_dest, q_vals = self.step(ep_num, t, max_t,is_baseline)

            # add to memory
            if not is_baseline and t > 0.001:
                #self.learning_agent.memory.push(state, action, new_state, reward)
                self.learning_agent.add_to_memory(state,action,new_state,reward,done)

            # add to history of
            for name, mover in self.mover_dict.items():
                mover.add_step_history(step_num)

            # epsiode history
            q_vals = np.reshape(q_vals,(len(q_vals[0]),))
            telemetry = np.concatenate(([t,reward.cpu().detach().numpy()[0][0],done,is_crashed,is_reached_dest],q_vals,action.cpu().detach().numpy()[0]))
            self.history.iloc[step_num] = telemetry
            cumulative_reward += reward

            # increase time
            t += delta_t
            step_num += 1

        # trim the histories
        self.history.drop(range(step_num,len(self.history)),inplace=True)
        for name, mover in self.mover_dict.items():
            mover.trim_history(step_num)

        return is_crashed, is_reached_dest, cumulative_reward

    def step(self,n_steps, t, max_t, is_baseline):
        """
        preform one time step of the simulation
        :param - the current number of steps for the overall training
        :return:
        """

        # sensors are assumed to be up to date from either the last step or from the reset action

        # step each of the movers
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                # this is the boats

                # Trud for getting the state that includes the sensors
                norm_state = mover.normalize_state(True)
                norm_state_tensor = self.convert_numpy_to_tensor(list(norm_state.values()))

                # build the state for the mover

                # returns the action, and all of the q vals
                if is_baseline:
                    action, q_vals = self.learning_agent.get_action_eval(norm_state_tensor)
                else:
                    action, q_vals = self.learning_agent.get_action_train(norm_state_tensor, n_steps)

                delta_adj, power_adj = mover.action_to_command(action.cpu().detach().numpy()[0][0])
                mover.set_control(mover.power + power_adj, mover.delta + delta_adj)

                self.old['x'] = mover.x
                self.old['y'] = mover.y
                self.old['mu'] = mover.mu
                self.old['dest_dist'] = mover.destination_distance
                self.old['fuel'] = mover.fuel
                self.old['phi'] = mover.phi
                for sensor in mover.sensors:
                    tmp_state = sensor.get_state()
                    self.old['theta_obs_1'] = tmp_state['angle_center_0']
                    self.old['theta_obs_2'] = tmp_state['angle_center_1']

                #self.mu_old = mover.mu
                #self.dest_dist_old = mover.destination_distance
                #self.fuel_old = mover.fuel

            mover.step()

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

        # get state prime or new state
        for name, mover in self.mover_dict.items():
            if mover.can_learn:
                new_norm_state = mover.normalize_state(True)
                new_norm_state_tensor = self.convert_numpy_to_tensor(list(new_norm_state.values()))

        # get reward
        reward, done, is_crashed, is_reached_dest = self.reward(t, max_t)
        reward_tensor = self.convert_numpy_to_tensor([reward])

        # set done-ness for next state
        z = np.zeros(self.learning_agent.state_size)
        if done:
            new_norm_state_tensor = torch.tensor([z], device=self.device,
                                                 dtype=torch.float)  # .view(self.state_size,1,-1)
        new_norm_state_tensor = new_norm_state_tensor.view(new_norm_state_tensor.size(), -1)

        # convert bool to numpy
        done = np.float(done)

        return norm_state_tensor, action,  new_norm_state_tensor, reward_tensor, done, is_crashed, is_reached_dest, q_vals

    def run_baseline(self,ep_num):
        """
        runs a set of baseline episodes were no random actions are used and
        :return:
        """

        # position_x
        x = [0,0,0,110,110,110]
        # position_y
        y = [20,50,80,20,50,80]
        # destination
        dest = [150,50]
        #obstacles
        obs_1 = [70,30,15]
        obs_2 = [70, 70, 15]

        for i, x_tmp in enumerate(x):

            # reset own history
            self.history = pd.DataFrame(data=np.zeros((int(
                np.ceil(self.h_params['scenario']['max_time'] / self.h_params['scenario']['time_step'])),len(self.header))), columns=self.header)

            self.reset()

            # set obstacle 1 location
            for name, mover in self.mover_dict.items():
                if mover.name == 'obs_1':
                    mover.pos = [obs_1[0], obs_1[1]]
                    mover.radius = obs_1[2]
                    break
            # set obstacle 2 location
            for name, mover in self.mover_dict.items():
                if mover.name == 'obs_2':
                    mover.pos = [obs_2[0], obs_2[1]]
                    mover.radius = obs_2[2]
                    break

            # set destination and intial orientation
            for name, mover in self.mover_dict.items():
                if 'boat' in mover.name:
                    # set destination
                    mover.destination = dest

                    mover.x = x[i]
                    mover.y = y[i]

                    mover.phi = 0
                    mover.delta = 0

            # negative 1 is given as the episode number as a arbitrary place holder
            is_crashed, is_destination_reached, cumulative_reward = self.run_episode(-1,True)

            # write the history of the baseline episode with a number for
            self.write_history(str(ep_num)+'-'+str(i),True)
        #

    def reset(self):
        """
        resets the state of the boats and the simulations. Random locations for the obstacles, boat, and destinations
        are all drawn to initialize the simulation
        :return:
        """

        # domain
        domain = self.h_params['scenario']['domain']

        # set the location of obstacle one
        x1 = np.random.random() * domain
        y1 = np.random.random() * domain
        r1 = 15.0 # [m]
        for name, mover in self.mover_dict.items():
            if mover.name == 'obs_1':
                mover.pos = [x1, y1]
                mover.radius = r1
                break

        # set the location of obstacle two
        x2 = np.random.random() * domain
        y2 = np.random.random() * domain
        r2 = 15.0  # [m]
        # obstacle to needs to be at least 10 meters from the other obstacle
        dist_12 = np.sqrt( (x2-x1)*(x2-x1) +(y2-y1)*(y2-y1))
        while dist_12 <= 10.0:
            x2 = np.random.random() * domain
            y2 = np.random.random() * domain
            dist_12 = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        for name, mover in self.mover_dict.items():
            if mover.name == 'obs_2':
                mover.pos = [x2, y2]
                mover.radius = r2
                break

        # set boat state variables

        # destination location
        x3 = np.random.random() * domain
        y3 = np.random.random() * domain
        dist_13 = np.sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1))
        dist_23 = np.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))
        while dist_13 < r1 + 10.0 or dist_23 < r2 + 10.0:
            x3 = np.random.random() * domain
            y3 = np.random.random() * domain
            dist_13 = np.sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1))
            dist_23 = np.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))

        # current position
        x4 = np.random.random() * domain
        y4 = np.random.random() * domain
        dist_14 = np.sqrt((x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1))
        dist_24 = np.sqrt((x4 - x2) * (x4 - x2) + (y4 - y2) * (y4 - y2))
        dist_34 = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3))
        while dist_14 < r1 + 10.0 or dist_24 < r2 + 10.0 and dist_34 > 3.0:
            x4 = np.random.random() * domain
            y4 = np.random.random() * domain
            dist_14 = np.sqrt((x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1))
            dist_24 = np.sqrt((x4 - x2) * (x4 - x2) + (y4 - y2) * (y4 - y2))

        for name, mover in self.mover_dict.items():
            if 'boat' in mover.name:
                # set destination
                mover.destination = [x3, y3]
                # set the power of the simulation
                if self.h_params['boat']['mode'] == 'propeller_only':
                    mover.power = 10000.0 # 10k [kilo-watt]
                elif self.h_params['boat']['mode'] == 'propeller_and_power':
                    mover.power = np.random.random() * 20000.0
                # set propeller angle
                mover.delta = (np.random.random()-0.5)*2.0*np.pi
                # set position of the boat itself
                mover.x = x4
                mover.y = y4
                # set hull angle:
                mover.phi = np.random.random() * np.pi * 2.0
                # set initial velocity
                v_boat = np.random.random() * 2.0
                mover.v_boat_x = v_boat * np.cos(mover.phi)
                mover.v_boat_y = v_boat * np.sin(mover.phi)
                # reset the acceleration to zero
                mover.a_x = 0.0
                mover.a_y = 0.0
                mover.a_boat_theta = 0.0
                # calculate distance metrics
                mover.calc_dest_metrics()
                # bound the hull angle
                mover.bound_hull_angle()
                # add random amount of initial fuel
                mover.fuel = 1.0 + np.random.random()  # 1-2.0 kilos of fuel

                # reset the old dictionary
                #self.old['fuel'] = mover.fuel
                #self.old['mu'] = mover.mu
                #self.old['dest_dist'] = mover.destination_distance

        # update sensors so that the state for all of the movers is accurate
        for name, mover in self.mover_dict.items():
            mover.update_sensors(self.mover_dict)

    def reward(self, t, max_t):
        """
        calculates the reward for the given state of the system. The reward denotes how good or bad the reward is.
        :param t:
        :param max_t:
        :return:
        """
        done = False

        reward = 0.0

        # get the boat, obstacle 1, and obstacle 2
        for key, mover in self.mover_dict.items():
            if 'boat' in key:
                boat = mover
            elif '_1' in key:
                obs_1 = mover
            elif '_2' in key:
                obs_2 = mover

        # add negative reward for fuel usage
        reward += (boat.fuel - self.old['fuel'])*self.h_params['reward']['fuel_use']

        # add reward for better angle
        boat.calc_dest_metrics()

        # reward for closing the distance to the destination
        if boat.destination_distance < self.old['dest_dist'] and boat.v_boat_x_p > 0:
            reward += (self.old['dest_dist'] - boat.destination_distance) * (
                    self.old['dest_dist'] - boat.destination_distance)/self.h_params['reward']['distance_norm']

        # extra reward for closeness to destination
        if boat.destination_distance <= 2.01:
            reward += self.h_params['reward']['success']
        # elif self.boat.destination_distance <= 10.0:
        #    reward += self.h_params['reward_prox']

        # add negative reward for being close to the obstacles
        reward = self.reward_angle_to_obstacle(boat, obs_1, obs_2, reward)

        # calculate distance to obstacles
        dist_to_1 = np.sqrt(
            (boat.x - obs_1.pos[0]) * (boat.x - obs_1.pos[0]) + (boat.y - obs_1.pos[1]) * (boat.y - obs_1.pos[1]))
        dist_to_2 = np.sqrt((boat.x - obs_2.pos[0]) * (boat.x - obs_2.pos[0]) + (
                boat.y - obs_2.pos[1]) * (boat.y - obs_2.pos[1]))

        if dist_to_1 <= (obs_1.radius+10.0) or dist_to_2 <= (obs_2.radius+10.0):
            reward += self.h_params['reward']['prox_crash']  # 50.0

        # add negative reward for running into an obstacle
        is_crashed = False
        if dist_to_1 <= obs_1.radius or dist_to_2 <= obs_2.radius:
            is_crashed = True
            reward += self.h_params['reward']['crash']  # 50.0

        # if the boat has reached the destination
        is_reached_dest = False
        if boat.destination_distance <= 2.0:
            is_reached_dest = True

        if t >= max_t - 1 or dist_to_1 < obs_1.radius or dist_to_2 < obs_2.radius or boat.destination_distance < 2.0 or boat.destination_distance >= 300.0:
            # stop simulation at max time, when it reaches its destination, or it runs into an obstacle
            done = True
        return reward, done, is_crashed, is_reached_dest

    def reward_angle_to_obstacle(self, boat, obs_1, obs_2, reward):
        # calculate distance to obstacles
        dist_to_1 = np.sqrt(
            (boat.x - obs_1.pos[0]) * (boat.x - obs_1.pos[0]) + (boat.y - obs_1.pos[1]) * (boat.y - obs_1.pos[1]))
        dist_to_2 = np.sqrt((boat.x - obs_2.pos[0]) * (boat.x - obs_2.pos[0]) + (
                boat.y - obs_2.pos[1]) * (boat.y - obs_2.pos[1]))

        # reward for angle to destination and angle to obstacles
        unit_x = np.cos(boat.phi)
        unit_y = np.sin(boat.phi)
        slope = unit_y / unit_x
        slope_obs = -unit_x / unit_y
        b = boat.y - slope * boat.x
        # obstacle 1
        b_obs_1 = obs_1.pos[1] - slope_obs * obs_1.pos[0]
        x_obs_1_int = (b_obs_1 - b) / (slope - slope_obs)
        y_obs_1_int = slope_obs * x_obs_1_int + b_obs_1
        close_dist_obs_1 = np.sqrt(
            (x_obs_1_int - obs_1.pos[0]) * (x_obs_1_int - obs_1.pos[0]) + (y_obs_1_int - obs_1.pos[1]) * (
                        y_obs_1_int - obs_1.pos[1]))
        angle_to_intersect_1 = np.arctan((y_obs_1_int - boat.y) / (x_obs_1_int - boat.x))
        # obstacle 2
        b_obs_2 = obs_2.pos[1] - slope_obs * obs_2.pos[0]
        x_obs_2_int = (b_obs_2 - b) / (slope - slope_obs)
        y_obs_2_int = slope_obs * x_obs_2_int + b_obs_2
        close_dist_obs_2 = np.sqrt(
            (x_obs_2_int - obs_2.pos[0]) * (x_obs_2_int - obs_2.pos[0]) + (y_obs_2_int - obs_2.pos[1]) * (
                        y_obs_2_int - obs_2.pos[1]))
        angle_to_intersect_2 = np.arctan((y_obs_2_int - boat.y) / (x_obs_2_int - boat.x))

        # buffer around the obstacle to avoid
        buffer = 5.0
        is_consider_1 = False
        is_consider_2 = False

        if close_dist_obs_1 <= (obs_1.radius + buffer):
            # boat's current line instersects with the obstacle
            if boat.phi - np.pi / 2.0 <= angle_to_intersect_1 <= boat.phi + np.pi / 2.0:
                # obstacle is infront of the boat, not behind
                if boat.destination_distance > (dist_to_1 - (obs_1.radius + buffer)):
                    # need to consider obstacle. gain reward for increassing angle to obstacle
                    is_consider_1 = True

                    for sensor in boat.sensors:
                        tmp_state = sensor.get_state()
                        theta = tmp_state['angle_center_0']

                    # correct theat
                    theta_old = self.old['theta_obs_1']
                    if theta_old > np.pi:
                        theta_old -= 2.0 * np.pi
                    if theta > np.pi:
                        theta -= 2.0 * np.pi

                    # correct phi
                    phi_old = self.old['phi']
                    if phi_old > np.pi:
                        phi_old -= 2.0 * np.pi
                    phi = boat.phi
                    if phi > np.pi:
                        phi -= 2.0 * np.pi

                    x_old = self.old['x']
                    y_old = self.old['y']
                    x_phi = np.cos(self.old['phi'])
                    y_phi = np.sin(self.old['phi'])
                    x_theta = np.cos(self.old['theta_obs_1'])
                    y_theta = np.sin(self.old['theta_obs_1'])
                    top = (x_phi - x_old) * (x_theta - x_old) + (y_phi - y_old) * (y_theta - y_old)
                    bottom = np.sqrt((x_phi - x_old) * (x_phi - x_old) + (y_phi - y_old) * (y_phi - y_old)) * np.sqrt(
                        (x_theta - x_old) * (x_theta - x_old) + (y_theta - y_old) * (y_theta - y_old))
                    delta_old = np.arccos(top / bottom)

                    x_new = boat.x
                    y_new = boat.y
                    x_phi = np.cos(boat.phi)
                    y_phi = np.sin(boat.phi)
                    x_theta = np.cos(theta)
                    y_theta = np.sin(theta)
                    top = (x_phi - x_new) * (x_theta - x_new) + (y_phi - y_new) * (y_theta - y_new)
                    bottom = np.sqrt((x_phi - x_new) * (x_phi - x_new) + (y_phi - y_new) * (y_phi - y_new)) * np.sqrt(
                        (x_theta - x_new) * (x_theta - x_new) + (y_theta - y_new) * (y_theta - y_new))
                    delta_new = np.arccos(top / bottom)
                    delta_theta_obs = delta_new - delta_old

                    sign_delta = np.sign(delta_theta_obs)
                    r_slope = 10.0 / 15.0
                    r_int = 1
                    if np.rad2deg(delta_theta_obs) > 10.0:
                        tmp_reward = 10.0
                    else:
                        tmp_reward = r_slope * np.rad2deg(delta_theta_obs) + r_int
                    reward += tmp_reward * sign_delta

        if close_dist_obs_2 <= (obs_2.radius + buffer):
            # boat's current line instersects with the obstacle
            if boat.phi - np.pi / 2.0 <= angle_to_intersect_2 <= boat.phi + np.pi / 2.0:
                # obstacle is infront of the boat, not behind
                if boat.destination_distance > (dist_to_2 - (obs_2.radius + buffer)):
                    # need to consider obstacle. gain reward for increassing angle to obstacle
                    is_consider_2 = True

                    for sensor in boat.sensors:
                        tmp_state = sensor.get_state()
                        theta = tmp_state['angle_center_1']

                    # correct theat
                    theta_old = self.old['theta_obs_2']
                    if theta_old > np.pi:
                        theta_old -= 2.0 * np.pi
                    if theta > np.pi:
                        theta -= 2.0 * np.pi

                    # correct phi
                    phi_old = self.old['phi']
                    if phi_old > np.pi:
                        phi_old -= 2.0 * np.pi
                    phi = boat.phi
                    if phi > np.pi:
                        phi -= 2.0 * np.pi

                    x_old = self.old['x']
                    y_old = self.old['y']
                    x_phi = np.cos(self.old['phi'])
                    y_phi = np.sin(self.old['phi'])
                    x_theta = np.cos(self.old['theta_obs_2'])
                    y_theta = np.sin(self.old['theta_obs_2'])
                    top = (x_phi - x_old) * (x_theta - x_old) + (y_phi - y_old) * (y_theta - y_old)
                    bottom = np.sqrt((x_phi - x_old) * (x_phi - x_old) + (y_phi - y_old) * (y_phi - y_old)) * np.sqrt(
                        (x_theta - x_old) * (x_theta - x_old) + (y_theta - y_old) * (y_theta - y_old))
                    delta_old = np.arccos(top / bottom)

                    x_new = boat.x
                    y_new = boat.y
                    x_phi = np.cos(boat.phi)
                    y_phi = np.sin(boat.phi)
                    x_theta = np.cos(theta)
                    y_theta = np.sin(theta)
                    top = (x_phi - x_new) * (x_theta - x_new) + (y_phi - y_new) * (y_theta - y_new)
                    bottom = np.sqrt((x_phi - x_new) * (x_phi - x_new) + (y_phi - y_new) * (y_phi - y_new)) * np.sqrt(
                        (x_theta - x_new) * (x_theta - x_new) + (y_theta - y_new) * (y_theta - y_new))
                    delta_new = np.arccos(top / bottom)
                    delta_theta_obs = delta_new - delta_old

                    sign_delta = np.sign(delta_theta_obs)
                    r_slope = 10.0 / 15.0
                    r_int = 1
                    if np.rad2deg(delta_theta_obs) > 10.0:
                        tmp_reward = 10.0
                    else:
                        tmp_reward = r_slope * np.rad2deg(delta_theta_obs) + r_int
                    reward += tmp_reward * sign_delta

        if not is_consider_1 and not is_consider_2:
            # gain reward for reducing angle to destination
            #if boat.v_boat_x_p > 0 and (np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0)):
            if np.abs(boat.mu) < np.abs(self.old['mu']) or np.abs(boat.mu) <= np.deg2rad(5.0):
                reward += (np.pi - abs(boat.mu)) * (np.pi - abs(boat.mu)) * self.h_params['reward']['angle']  # normalize reward to bound it

        return reward

    def write_history(self, episode_number,is_baseline=False):
        """
        writes the telemetry for the episode out to a csv file for later processing
        :param episode_number: integer for the episode number
        :param is_baseline: boolean for if the episode is a base line episode or a training episode
        :return:
        """

        # combine all of the histories together
        total_history = self.history
        for _, mover in self.mover_dict.items():
            total_history = pd.concat([total_history,mover.history], axis=1)

        if is_baseline:
            # the baseline episode is placed into a different folder
            total_history.to_csv(
                'Output\\Trial_' + str(self.trial_num) + '\\baseline\\data\\' + str(episode_number) + '.csv',index=False)
        else:
            total_history.to_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_number)+'.csv',index=False)

    def write_hyper_params(self, trial_num):
        """
        make a copy of the hyperparameters and save them to the output directory for saving later
        :return:
        """
        with open('Output\\Trial_' + str(trial_num)+'\\hyper_parameters.yml', 'w') as file:
            yaml.safe_dump(self.h_params,file)

    def render_episode(self,episode_num):

        df = pd.read_csv('Output\\Trial_'+str(self.trial_num)+'\\episodes\\data\\'+str(episode_num)+'.csv')

        # graph episode
        sns.set_theme()
        fig = plt.figure(0, figsize=(14, 12))
        gs = GridSpec(5, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        ax4 = fig.add_subplot(gs[2, 2])
        ax5 = fig.add_subplot(gs[2, 3])
        ax6 = fig.add_subplot(gs[3, 2])
        ax7 = fig.add_subplot(gs[3,3])
        ax8 = fig.add_subplot(gs[4, 0])
        ax9 = fig.add_subplot(gs[4, 1])
        ax10 = fig.add_subplot(gs[4, 2])
        ax11 = fig.add_subplot(gs[4, 3])

        # graph trajectory
        circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[0]), radius=2.0, alpha=1.0,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]), radius=df['radius_obs_1'].iloc[0],
                                alpha=1.0,
                                color='tab:blue')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]), radius=df['radius_obs_1'].iloc[0]+10.0,
                                alpha=0.2,
                                color='tab:blue')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_2'].iloc[0], df['y_obs_2'].iloc[0]), radius=df['radius_obs_2'].iloc[0],
                                alpha=1.0,
                                color='tab:orange')
        ax1.add_patch(circle)
        circle = patches.Circle((df['x_obs_2'].iloc[0], df['y_obs_2'].iloc[0]), radius=df['radius_obs_2'].iloc[0]+10.0,
                                alpha=0.2,
                                color='tab:orange')
        ax1.add_patch(circle)
        sc = ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none')
        plt.colorbar(sc, ax=ax1)
        # get the points for each 20% of the trajectory
        spacing = int(len(df) / 4)
        idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
        for j in range(len(idx)):
            ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

        #ax1.legend()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')

        # graph the q values
        if self.h_params['boat']['mode'] == 'propeller_and_power':
            select_q = np.zeros(len(df))
            for i in range(len(select_q)):
                select_q[i] = df['q_val_'+str(int(df['action'].iloc[i]))].iloc[i]
            ax2.plot(df['time'],select_q)
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel('q value')
            #ax2.legend()

        else:
            action_size = self.learning_agent.action_size
            for j in range(action_size):
                ax2.plot(df['time'], df['q_val_' + str(j)], label=str(j))
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel('q value')
            ax2.legend()

        # graph angles
        ax3.plot(df['time'], np.rad2deg(df['phi']), label='hull_angle')
        ax3.plot(df['time'], np.rad2deg(df['delta']), label='prop angle')
        ax3.plot(df['time'], np.rad2deg(df['mu']), label='hull to dest')
        ax3.plot(df['time'], np.rad2deg(df['theta']), label='to dest')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angles [deg]')
        ax3.legend()

        # graph velocities
        #ax4.plot(df['time'], df['v_boat_x'], label='x')
        #ax4.plot(df['time'], df['v_boat_y'], label='y')
        ax4.plot(df['time'], df['v_boat_x_p'], label='x_p')
        ax4.plot(df['time'], df['v_boat_theta'], label='theta')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Velocity')
        ax4.legend()

        # graph power
        ax5.plot(df['time'],df['power'])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Power [watt]')

        #
        ax6.plot(df['time'], df['v_boat_x'], label='x')
        ax6.plot(df['time'], df['v_boat_y'], label='y')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Velocity')
        ax6.legend()

        # fuel
        ax7.plot(df['time'], df['fuel'])
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('Fuel [kg]')

        # graph the reward
        ax8.plot(df['time'], df['reward'])
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Reward')

        # graph distance to the obstacles
        ax9.plot(df['time'], df['dist_center_0'])
        ax9.plot(df['time'], df['dist_center_1'])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel('Dist to obstacle [m]')

        # angle to the obstacles
        ax10.plot(df['time'], np.rad2deg(df['angle_center_0']))
        ax10.plot(df['time'], np.rad2deg(df['angle_center_1']))
        ax10.set_xlabel('Time [s]')
        ax10.set_ylabel('Angle to obstacle [deg]')

        # distance to the destination
        ax11.plot(df['time'], df['destination_distance'])
        ax11.plot([0, max(df['time'])], [2.0, 2.0])
        ax11.set_xlabel('Time [s]')
        ax11.set_ylabel('Dist to dest [m]')

        # tidy up the graph
        plt.tight_layout()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\episodes\\graph\\' + str(episode_num) + '.png')
        plt.close('all')

    def render_training(self, trial_num):

        # render each episode
        reward = np.zeros(self.h_params['scenario']['num_episodes'])
        is_crashed = np.zeros_like(reward)
        is_reached = np.zeros_like(reward)
        min_dist = np.zeros_like(reward)
        for i in range(self.h_params['scenario']['num_episodes']):
            df = pd.read_csv('Output\\Trial_'+str(trial_num)+'\\episodes\\data\\'+str(i)+'.csv')



            # gather data for trend across the training
            reward[i] = np.sum(df['reward'])
            is_crashed[i] = max(df['is_crashed'])
            is_reached[i] = max(df['is_reached'])
            min_dist[i] = min(df['destination_distance'])

        # overall training graphs
        reward_df = pd.DataFrame()
        fig = plt.figure(0,figsize=(14,8))
        windows = [1,51,101,201]
        #windows = [1, 3, 5]
        avg = np.zeros_like(reward)
        for window in windows:
            avg, alpha = self.moving_average(reward,window)
            reward_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])],avg,label='window='+str(window),alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average reward')
        plt.legend()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward.png')
        reward_df.to_csv('Output\\Trial_'+str(trial_num)+'\\history\\Reward.csv')
        plt.close('all')

        is_crashed_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_crashed)
        for window in windows:
            avg, alpha = self.moving_average(is_crashed, window)
            is_crashed_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Crash Rate')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.png')
        is_crashed_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Crashed.csv')
        plt.close('all')

        is_reached_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(is_reached)
        for window in windows:
            avg, alpha = self.moving_average(is_reached, window)
            is_reached_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Average Success Rate')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.png')
        is_reached_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Is_Reached.csv')
        plt.close('all')

        min_dist_df = pd.DataFrame()
        fig = plt.figure(0, figsize=(14, 8))
        avg = np.zeros_like(min_dist)
        for window in windows:
            avg, alpha = self.moving_average(min_dist, window)
            min_dist_df[window] = avg
            plt.plot([i for i in range(self.h_params['scenario']['num_episodes'])], avg, label='window=' + str(window),
                     alpha=alpha)
        plt.xlabel('Episode number')
        plt.ylabel('Minimum Distance to Destination [m]')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Min_dist.png')
        min_dist_df.to_csv('Output\\Trial_' + str(trial_num) + '\\history\\Min_Dist.csv')
        plt.close('all')

        # baseline graphs

        baseline_data_path = 'Output\\Trial_'+str(trial_num)+'\\baseline\\data'
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(baseline_data_path) if isfile(join(baseline_data_path, f))]
        first_char = onlyfiles[0][0]
        num_iters = 0
        for i in range(len(onlyfiles)):
            if onlyfiles[i][0] == first_char:
                num_iters += 1
            else:
                break

        base_eps = []
        reward = []
        tmp_reward = []
        is_crashed = []
        tmp_crashed = []
        is_destination = []
        tmp_destination = []
        min_dist =[]
        tmp_min_dist = []
        baseline_number = 0
        #while baseline_number <= len(onlyfiles):
        #for file in onlyfiles:
        while baseline_number < self.h_params['scenario']['num_episodes']:

            for i in range(num_iters):
                df = pd.read_csv(baseline_data_path + '\\' + str(baseline_number)+'-'+str(i)+'.csv')

                if i == 0:
                    fig = plt.figure(0, figsize=(14, 8))
                    # gs = GridSpec(5, 4, figure=fig)
                    ax1 = fig.add_subplot(111)

                    circle = patches.Circle((df['destination_x'].iloc[0], df['destination_y'].iloc[1]), radius=2.0,
                                            alpha=1.0,
                                            color='tab:green')
                    ax1.add_patch(circle)
                    circle = patches.Circle((df['x_obs_1'].iloc[0], df['y_obs_1'].iloc[0]),
                                            radius=df['radius_obs_1'].iloc[0],
                                            alpha=1.0,
                                            color='tab:blue')
                    ax1.add_patch(circle)
                    circle = patches.Circle((df['x_obs_2'].iloc[0], df['y_obs_2'].iloc[0]),
                                            radius=df['radius_obs_2'].iloc[0],
                                            alpha=1.0,
                                            color='tab:orange')
                    ax1.add_patch(circle)
                    base_eps.append(baseline_number)
                tmp_reward.append(np.sum(df['reward']))
                tmp_crashed.append(np.max(df['is_crashed']))
                tmp_destination.append(np.max(df['is_reached']))
                tmp_min_dist.append(np.min(df['destination_distance']))

                label = str(i)
                ax1.scatter(df['x_pos'], df['y_pos'], c=df['time'], cmap=cm.plasma, edgecolor='none', label=label)
                # get the points for each 20% of the trajectory
                spacing = int(len(df) / 4)
                idx = [0, spacing, 2 * spacing, 3 * spacing, len(df) - 1]
                for j in range(len(idx)):
                    ax1.text(df['x_pos'].iloc[idx[j]], df['y_pos'].iloc[idx[j]], df['time'].iloc[idx[j]], c='black')

            plt.tight_layout()
            plt.savefig('Output\\Trial_' + str(trial_num) + '\\baseline\\graph\\' + str(baseline_number) + '.png')
            plt.close('all')

            reward.append([min(tmp_reward), np.mean(tmp_reward), np.max(tmp_reward)])
            tmp_reward = []
            is_crashed.append([min(tmp_crashed), np.mean(tmp_crashed), np.max(tmp_crashed)])
            tmp_crashed = []
            is_destination.append([min(tmp_destination), np.mean(tmp_destination), np.max(tmp_destination)])
            tmp_destination = []
            min_dist.append([min(tmp_min_dist), np.mean(tmp_min_dist), np.max(tmp_min_dist)])
            tmp_min_dist = []

            baseline_number += self.h_params['scenario']['baseline_frequency']

        # graph trends in the baseline over time
        plt.close('all')
        reward = np.reshape(reward,(len(reward),3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps,reward[:,0],label='min')
        plt.plot(base_eps, reward[:, 1], label='avg')
        plt.plot(base_eps, reward[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('Output\\Trial_'+str(trial_num)+'\\history\\Reward_Baseline.png')
        plt.close('all')

        is_crashed = np.reshape(is_crashed, (len(is_crashed), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_crashed[:, 0], label='min')
        plt.plot(base_eps, is_crashed[:, 1], label='avg')
        plt.plot(base_eps, is_crashed[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Crash Rate')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Crash_Rate_Baseline.png')
        plt.close('all')

        is_destination = np.reshape(is_destination, (len(is_destination), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, is_destination[:, 0], label='min')
        plt.plot(base_eps, is_destination[:, 1], label='avg')
        plt.plot(base_eps, is_destination[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Success_Rate_Baseline.png')
        plt.close('all')

        min_dist = np.reshape(min_dist, (len(min_dist), 3))
        fig = plt.figure(0, figsize=(14, 8))
        plt.plot(base_eps, min_dist[:, 0], label='min')
        plt.plot(base_eps, min_dist[:, 1], label='avg')
        plt.plot(base_eps, min_dist[:, 2], label='max')
        plt.xlabel('Baseline Episode')
        plt.ylabel('Minimum Distance [m]')
        plt.legend()
        plt.savefig('Output\\Trial_' + str(trial_num) + '\\history\\Minimum_Distance_Baseline.png')
        plt.close('all')

    def moving_average(self,arr,window):
        """
        given an array, a moving average is calculated given a window to average over. The ends hold value to nearest
        possible average
        :param arr:
        :param window:
        :return:
        """
        avg = np.zeros_like(arr)
        half_window = int(window / 2.0)
        if half_window == 0:
            avg = arr
            alpha = 0.3
        else:
            alpha = 1.0
            for i in range(len(arr)):
                if i < half_window:
                    avg[i] = np.sum(arr[0:window]) / window
                elif i > (len(arr) - half_window):
                    avg[i] = np.sum(arr[len(arr) - window:len(arr)]) / window
                else:
                    avg[i] = np.sum(arr[i - half_window:i + half_window + 1]) / window
        return avg, alpha


if __name__ == '__main__':

    # trial 0 is used for debugging
    trial_num = 23 # the trial number to save the output information too
    input_file = 'scenario_params.yml'  # file that describes the parameters for the training

    # run a no obstalce scenario
    no = NoObstacles()
    no.launch_training(trial_num, input_file)

    # run a one obstacle scenario
    #oo = OneObstacles()
    #oo.launch_training(trial_num,input_file)