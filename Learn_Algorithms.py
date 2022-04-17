
from collections import namedtuple
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def convert_numpy_to_tensor(device, arr):
    """
    converts a numpy array to a tensor for use with pytorch
    :param arr:
    :return:
    """
    tmp = torch.tensor([arr], device=device, dtype=torch.float)
    return tmp.view(tmp.size(), -1)


class ReplayStorage:

    def __init__(self,h_params):
        """


        :param h_params:
        :param transition:
        """

        # save the hyperparameters
        self.h_params = h_params
        self.transition = None
        self.set_transition()
        self.capacity = self.h_params['replay_data']['capacity']
        self.position = dict()

        # initialize the replay buffers
        self.buffers = dict()
        self.interim_buffer = None
        self.strategy_initializer(self.transition)

    def set_transition(self):
        """
        for the configuration outlined in the input hyper-parameter files, determine the appropriate transition data to
        be stored during each step of the simulation

        :return:
        """
        strategy = self.h_params['replay_data']['replay_strategy']
        if strategy == 'all_in_one':
            self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        elif strategy == 'proximity':
            self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done','prox'))
        elif strategy == 'outcome':
            self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done','outcome'))
        else:
            raise ValueError('Invalid option given for replay strategy. Only \'all_in_one\', \'proximity\', and \'outcome\' are allowed')

    def strategy_initializer(self, transition):
        """
        intializes the replay buffer strategy, creating the buffers based on how the memory will be stored and
        organized. Currently there exists three strategies.
            1 - All data in one buffer
            2 - data split into close to an obstacle, and data far from an obstacle
            3 - data split into tree cases: a successful episode, and episode where the boat crashes, and an episode
                where the boat does not crash or succeed

        :param transition: named tuple that defines the data that is stored in the replay buffers
        :return:
        """

        # initialize the position for the interim buffer
        self.position['interim'] = 0

        # set up the replay storage based on the strategt passed in by the hyper parameters
        strategy = self.h_params['replay_data']['replay_strategy']
        if strategy == 'all_in_one':
            # all data samples are saved into one replay buffer
            tmp_buffer = ReplayMemory(self.capacity,transition,'only')
            self.buffers['only'] = tmp_buffer

            # initialize the position for the only buffer
            self.position['only'] = 0

        elif strategy == 'proximity':
            # data is saved into two buffers based on how close the boat is to an obstacle
            close_buffer = ReplayMemory(self.capacity,transition,'close')
            far_buffer = ReplayMemory(self.capacity,transition,'far')
            self.buffers['close'] = close_buffer
            self.buffers['far'] = far_buffer

            # initialize the position for the both buffers
            self.position['close'] = 0
            self.position['far'] = 0

        elif strategy == 'outcome':
            # data is saved into three replay buffers based on the outcome of the episode: crash, success, or others
            crash_buffer = ReplayMemory(self.capacity,transition,'crash')
            success_buffer = ReplayMemory(self.capacity,transition,'success')
            other_buffer = ReplayMemory(self.capacity,transition, 'other')
            self.buffers['crash'] = crash_buffer
            self.buffers['success'] = success_buffer
            self.buffers['other'] = other_buffer

            # initialize the position for the all buffers
            self.position['crash'] = 0
            self.position['success'] = 0
            self.position['other'] = 0

        else:
            raise ValueError('An invalid learning agent replay strategy was given, only \'all_in_one\', \'proximity\', or \'outcome\' are accepted')

        # initialize the buffer that is used during an episode to store data tuples before being sorted into the correct
        # replay buffer
        self.reset_interim_buffer()

    def push(self, *args):
        """
        pushes a single data point (a state, action, reward, next state, and reward) tuple to the interim replay buffer.
        The interim buffers keeps the data until it can be sorted

        :param transition:
        :return:
        """
        self.interim_buffer.push(*args)

    def sort_data_into_buffers(self):
        """
        The data that has been accumulated over the course of an episode is sorted into the correct replay buffers
        based on the storage strategy being used. It is assumed this function is called after the completion of an
        episode.

        :return:
        """
        strategy = self.h_params['replay_data']['replay_strategy']
        if strategy == 'all_in_one':
            for data in self.interim_buffer.memory:
                if len(self.buffers['only']) < self.capacity:
                    self.buffers['only'].memory.append(None)
                self.buffers['only'].memory[self.position['only']] = data
                self.position['only'] = (self.position['only'] + 1) % self.capacity
        elif strategy == 'proximity':

            # get the proximity threshold from the hyper-parameters
            prox_thresh = self.h_params['replay_data']['proximity_threshold']

            # iterate through the data to sort it into the correct buffer
            for data in self.interim_buffer.memory:

                # determine what buffer the data should be in
                if data.prox <= prox_thresh:
                    tag = 'close'
                else:
                    tag = 'far'

                # add the data point to the buffer
                buffer = self.buffers[tag].memory
                if len(buffer) < self.capacity:
                    buffer.append(None)
                buffer[self.position[tag]] = data
                self.position[tag] = (self.position[tag] + 1) % self.capacity

        elif strategy == 'outcome':

            # get the last data point to determine the outcome
            tag = self.interim_buffer.memory[-1].outcome
            # iterate through the data to sort it into the correct buffer

            for data in self.interim_buffer.memory:

                buffer = self.buffers[tag].memory

                # add the data point to the buffer
                if len(buffer) < self.capacity:
                    buffer.append(None)
                buffer[self.position[tag]] = data
                self.position[tag] = (self.position[tag] + 1) % self.capacity

        self.reset_interim_buffer()

    def reset_interim_buffer(self):
        """
        set the buffer that is used to store the data during an episode to empty to be prepared for the next episode

        :return:
        """

        # create a new buffer to 'reset' the buffer
        self.interim_buffer = ReplayMemory(self.capacity,self.transition,'interim')

        # reset the position of the interim buffer
        self.position['interim'] = 0

    def sample(self, batch_size):
        """
        given a request for the number of samples, a batch of data is taken from the replay buffers based on the
        strategy. The distributions from each buffer are specified in the input file.

        :param batch_size: the number of data tuples tp sample from the replay buffers to train over
        :return:
        """
        strategy = self.h_params['replay_data']['replay_strategy']
        if strategy == 'all_in_one':

            if len(self.buffers['only'].memory) < batch_size:
                # not enough data to fill up one batch
                return None

            return random.sample(self.buffers['only'].memory, batch_size)
        elif strategy == 'proximity':

            if len(self.buffers['close'].memory) + len(self.buffers['far'].memory) < batch_size:
                # there is not enough data in both buffers to complete one batch
                return None

            n_close = int(np.floor(self.h_params['replay_data']['close_fraction']*float(batch_size)))
            n_far = batch_size - n_close

            enough_close = False
            if n_close <= len(self.buffers['close'].memory):
                enough_close = True
            enough_far = False
            if n_far <= len(self.buffers['far'].memory):
                enough_far = True

            if enough_close and enough_far:

                close = random.sample(self.buffers['close'].memory, n_close)
                far = random.sample(self.buffers['far'].memory, n_far)
            else:
                # one of the buffers does not have enough data so some needs to be borrowed from the other buffer to
                # fill out the buffer
                # sort the buffers in descending error
                buffer_lens = [['far', len(self.buffers['far'].memory), n_far],
                               ['close', len(self.buffers['close'].memory), n_close]]

                # bubble sort
                n = len(buffer_lens)
                for i in range(n):
                    for j in range(0, n - i - 1):

                        # traverse the array from 0 to n-i-1
                        # Swap if the element found is greater
                        # than the next element
                        if buffer_lens[j][1] < buffer_lens[j + 1][1]:
                            buffer_lens[j], buffer_lens[j + 1] = buffer_lens[j + 1], buffer_lens[j]

                # determine how many extra data points are needed.
                extra_needed = 0
                while len(buffer_lens) > 0:

                    tmp_buffer_info = buffer_lens.pop()

                    if len(buffer_lens) == 0:
                        adj = extra_needed
                    else:
                        adj = int(np.floor(extra_needed / len(buffer_lens)))

                    tmp_additional_data = (tmp_buffer_info[2] + adj) - len(self.buffers[tmp_buffer_info[0]].memory)
                    if tmp_additional_data > 0:
                        # there is not enough data in this buffer to meet the requested amount, so log that more
                        # data is needed from the remaining buffers
                        extra_needed = tmp_additional_data

                        if tmp_buffer_info[0] == 'far':
                            far = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                    len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'close':
                            close = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                  len(self.buffers[tmp_buffer_info[0]]))
                    else:
                        if tmp_buffer_info[0] == 'far':
                            far = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                    tmp_buffer_info[2] + extra_needed)
                        elif tmp_buffer_info[0] == 'close':
                            close = random.sample(self.buffers[tmp_buffer_info[0]].memory,
                                                  tmp_buffer_info[2] + extra_needed)

            batch = close + far
            return batch

        elif strategy == 'outcome':

            if len(self.buffers['success'].memory)+len(self.buffers['crash'].memory)+len(self.buffers['other'].memory) < batch_size:
                # there is not enough data in all of the buffers to complete one batch
                return None

            n_success = int(np.floor(self.h_params['replay_data']['success_fraction']*float(batch_size)))
            n_crash = int(np.floor(self.h_params['replay_data']['crash_fraction']*float(batch_size)))
            n_other = batch_size - n_crash - n_success

            enough_success = False
            if n_success <= len(self.buffers['success'].memory):
                enough_success = True
            enough_crash = False
            if n_crash <= len(self.buffers['crash'].memory):
                enough_crash = True
            enough_other = False
            if n_other <= len(self.buffers['other'].memory):
                enough_other = True

            if enough_success and enough_crash and enough_other:
                # there exists enough data in each buffer to meet the desired batch size in the desired ratios
                success = random.sample(self.buffers['success'].memory, n_success)
                crash = random.sample(self.buffers['crash'].memory, n_crash)
                other = random.sample(self.buffers['other'].memory, n_other)
            else:
                # one or more of the buffers does not have enough data to meet the desired ratio, but enough data exists
                # to fill up the batch. Data is pulled from the buffers that have excess to allow for training to begin
                # before each buffer has enough

                # sort the buffers in descending error
                buffer_lens = [['success',len(self.buffers['success'].memory), n_success],
                               ['crash',len(self.buffers['crash'].memory), n_crash],
                               ['other',len(self.buffers['other'].memory), n_other]]

                # bubble sort
                n = len(buffer_lens)
                for i in range(n):
                    for j in range(0, n - i - 1):

                        # traverse the array from 0 to n-i-1
                        # Swap if the element found is greater
                        # than the next element
                        if buffer_lens[j][1] < buffer_lens[j + 1][1]:
                            buffer_lens[j], buffer_lens[j + 1] = buffer_lens[j + 1], buffer_lens[j]

                # determine how many extra data points are needed.
                extra_needed = 0
                while len(buffer_lens) > 0:

                    tmp_buffer_info = buffer_lens.pop()

                    if len(buffer_lens) == 0:
                        adj = extra_needed
                    else:
                        adj = int(np.floor(extra_needed/len(buffer_lens)))

                    tmp_additional_data = (tmp_buffer_info[2]+adj)-len(self.buffers[tmp_buffer_info[0]].memory)
                    if tmp_additional_data > 0:
                        # there is not enough data in this buffer to meet the requested amount, so log that more
                        # data is needed from the remaining buffers
                        extra_needed = tmp_additional_data

                        if tmp_buffer_info[0] == 'success':
                            success = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'crash':
                            crash = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                        elif tmp_buffer_info[0] == 'other':
                            other = random.sample(self.buffers[tmp_buffer_info[0]].memory,len(self.buffers[tmp_buffer_info[0]]))
                    else:
                        if tmp_buffer_info[0] == 'success':
                            success = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)
                        elif tmp_buffer_info[0] == 'crash':
                            crash = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)
                        elif tmp_buffer_info[0] == 'other':
                            other = random.sample(self.buffers[tmp_buffer_info[0]].memory,tmp_buffer_info[2]+extra_needed)


            batch = success + crash + other
            return batch


class ReplayMemory(object):

    def __init__(self, capacity, transition, name):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition
        self.name = name

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):

    def __init__(self, state_size, action_size, h_params):
        super(Network, self).__init__()

        # gps layers
        # self.conv_1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(4,4))
        # self.conv_2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(4,4))
        # self.conv_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4,4))
        # self.pool = nn.MaxPool2d(kernel_size=(2,2))
        # self.flatten = nn.Flatten()

        # gps layers
        '''
        self.fc1 = nn.Linear(state_size, 100)
        self.fc2 = nn.Linear(136, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, action_size)
        '''
        self.fc1 = nn.Linear(state_size, h_params['learning_agent']['nodes_in_layer'])
        self.fc2 = nn.Linear(h_params['learning_agent']['nodes_in_layer'], h_params['learning_agent']['nodes_in_layer'])
        self.out = nn.Linear(h_params['learning_agent']['nodes_in_layer'], action_size)
        self.relu = torch.nn.ReLU()
        self.drop = nn.Dropout(h_params['learning_agent']['drop_out'])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, z):
        # GPS mask usage
        '''
        y = z[0]
        y = y[:, None, : ,:]
        x = z[1]

        # gps head
        y = self.conv_1(y)
        y = self.relu(y)
        #y = self.pool(y)
        y = self.conv_2(y)
        y = self.relu(y)
        #y = self.pool(y)
        y = self.conv_3(y)
        y = self.relu(y)
        #y = self.pool(y)
        y = self.flatten(y)

        # other state information head
        x = self.fc1(x)
        x = self.relu(x)

        # combine
        x = torch.cat((y,x),1)
        #flat = x.size()
        # combined layers
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.out(x)
        #actions = x  # psudeo linear activation
        #check = self.out(x.view(x.size(0), -1))
        #return self.out(x.view(x.size(0), -1))
        #return self.head(x.view(x.size(0), -1))
        return x # psudeo linear activation
        '''
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.out(x)
        return x  # psudeo linear activation


class Actor(nn.Module):

    def __init__(self, state_size, action_size, h_params):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, h_params['learning_agent']['nodes_in_layer'])
        self.fc2 = nn.Linear(h_params['learning_agent']['nodes_in_layer'], h_params['learning_agent']['nodes_in_layer'])
        self.out = nn.Linear(h_params['learning_agent']['nodes_in_layer'], action_size)
        self.relu = torch.nn.ReLU()
        self.drop = nn.Dropout(h_params['learning_agent']['drop_out'])
        # TODO check if max action needs to be converted to a tensor
        raw_action = list(h_params['learning_agent']['max_action'].split(','))
        float_arr = [float(x) for x in raw_action]
        self.max_action = convert_numpy_to_tensor('cuda',float_arr)
        self.tanh = torch.nn.Tanh()

    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.tanh(x)
        return x


class ActorTwoHeads(nn.Module):

    def __init__(self, state_size, action_size, h_params):
        super(ActorTwoHeads, self).__init__()
        self.fc1 = nn.Linear(state_size, h_params['learning_agent']['nodes_in_layer'])
        self.fc2 = nn.Linear(h_params['learning_agent']['nodes_in_layer'], h_params['learning_agent']['nodes_in_layer'])
        self.mu_out = nn.Linear(h_params['learning_agent']['nodes_in_layer'], action_size)
        self.sigma_out = nn.Linear(h_params['learning_agent']['nodes_in_layer'], action_size)
        self.relu = torch.nn.ReLU()
        self.drop = nn.Dropout(h_params['learning_agent']['drop_out'])

    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        mu = self.mu_out(x)
        sigma = self.sigma_out(x)
        return mu, sigma


class Critic(nn.Module):

    def __init__(self, state_size, action_size, h_params):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, h_params['learning_agent']['nodes_in_layer'])
        self.fc2 = nn.Linear(h_params['learning_agent']['nodes_in_layer'], h_params['learning_agent']['nodes_in_layer'])
        self.out = nn.Linear(h_params['learning_agent']['nodes_in_layer'], 1)
        self.relu = torch.nn.ReLU()
        self.drop = nn.Dropout(h_params['learning_agent']['drop_out'])

    def forward(self, state_action):
        """
        the state and action need to be concatenated before
        :param state_action:
        :return:
        """
        x = self.fc1(state_action)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class DQN:

    def __init__(self, state_size, action_size, h_params, device):
        """
        This is a DQN agent. THis class holds the target, and policy networks for the DQN network.
        :param state_size:
        :param action_size:
        :param h_params:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.policy_network = Network(state_size, action_size,h_params).cuda()
        self.target_network = Network(state_size, action_size,h_params).cuda()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.h_params = h_params
        self.optimizer = None
        self.set_optimizer(h_params)
        self.replay_storage = ReplayStorage(self.h_params)
        self.transition = self.replay_storage.transition
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['learn_rate'],betas=(h_params['beta_1'], h_params['beta_2']))

        #self.memory_far = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        #self.memory_close = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.device = device # choose cpu or gpu

    def train_agent(self):
        """
        train/optimize the networks
        :return:
        """

        loss = None
        c = 0
        while c < self.h_params['learning_agent']['n_batches']:

            transitions = self.replay_storage.sample(self.h_params['learning_agent']['batch_size'])
            if transitions is None:
                return

            #far_sample_size = int(self.h_params['learning_agent']['batch_size'] #*self.h_params['learning_agent']['memory_split'])
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = self.transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)

            # non_final_mask_gps = torch.tensor(tuple(map(lambda s: s is not None,batch.next_gps)), device=self.device, dtype=torch.bool)
            # non_final_next_gps = torch.cat([s for s in batch.next_gpsif s is not None])

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # concatenate the far and close parts

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # state_action_values = self.policy_net([gps_batch,state_batch]).gather(1, action_batch.type(torch.int64))
            state_action_values = self.policy_network(state_batch).gather(1, action_batch.type(torch.int64))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.h_params['learning_agent']['batch_size'], device=self.device)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

            # Compute the expected Q values
            #expected_state_action_values = (next_state_values * self.h_params['learning_agent']['gamma']) + reward_batch
            expected_state_action_values = torch.add(torch.mul(torch.reshape(next_state_values,(len(next_state_values),1))  , self.h_params['learning_agent']['gamma']) , reward_batch)

            # Compute Huber loss
            if self.h_params['learning_agent']['loss'] == 'huber':
                #loss = F.smooth_l1_loss(state_action_values.type(torch.double),expected_state_action_values.unsqueeze(1).type(torch.double))
                loss = F.smooth_l1_loss(state_action_values.type(torch.double),expected_state_action_values.type(torch.double))
            elif self.h_params['learning_agent']['loss'] == 'mse':
                #loss = F.mse_loss(state_action_values.type(torch.double), expected_state_action_values.unsqueeze(1).type(torch.double))
                loss = F.mse_loss(state_action_values.type(torch.double),
                                  expected_state_action_values.type(torch.double))
            else:
                raise ValueError('Given loss is currently not supported')

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            c += 1

        return loss

    def update_target_network(self):
        """
        copies the values of the policy network into the target network
        :return:
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_action_train(self, state, n_steps, noise):
        """
        gets an action from the network and stores the gradient information. This actions is expected to be used during
        the evaluation stages
        :param state:
        :param n_steps:
        :return:
        """

        sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * n_steps / self.eps_decay)
        if n_steps > self.h_params['learning_agent']['epsilon_decay']:
            eps_threshold = self.h_params['learning_agent']['epsilon_end']
        else:
            eps_threshold = (self.h_params['learning_agent']['epsilon_end'] - self.h_params['learning_agent']['epsilon_start']) / self.h_params['learning_agent']['epsilon_decay'] \
                            * n_steps + self.h_params['learning_agent']['epsilon_start']
        self.h_params['learning_agent']['epsilon_threshold'] = eps_threshold

        if sample > eps_threshold:

            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #return self.policy_network(state).argmax().view(1, -1), self.policy_network(state).max(), self.policy_network(state).argmax()
            return self.policy_network(state).argmax().view(1, -1), self.policy_network(state).cpu().detach().numpy()
        else:

            rand_int = random.randrange(self.action_size)
            #self.policy_network(state).detach().numpy()
            #tmp = self.policy_network(state).cpu().detach().numpy()
            #return torch.tensor([[rand_int]], device=self.device, dtype=torch.int), self.policy_network(state).detach().numpy()[0, rand_int], rand_int
            return torch.tensor([[rand_int]], device=self.device, dtype=torch.int), self.policy_network(state).cpu().detach().numpy()

    def get_action_eval(self, state):
        """
        deterministically gets and action that does not influence training. THis is expected to be used in baseline
        scenarios were we want to evaluate only the network and not do exploration
        :param state:
        :return:
        """
        with torch.no_grad():
            return self.policy_network(state).argmax().view(1, -1), self.policy_network(state).cpu().detach().numpy()

    def set_optimizer(self, h_params):
        """
        set the optimizer of the learning algorithm
        :param h_params:
        :return:
        """
        optimizer = h_params['learning_agent']['optimizer']

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=h_params['learning_agent']['learn_rate'],
                                        betas=(h_params['learning_agent']['beta_1'], h_params['learning_agent']['beta_2']))
        elif optimizer == 'rms_prop':
            self.optimizer = optim.RMSprop(self.policy_network.parameters(),lr=h_params['learning_agent']['learn_rate'],
                                           alpha=h_params['learning_agent']['alpha'],momentum=h_params['learning_agent']['momentum'])
        else:
            raise ValueError('Optimizer {} is currently not supported'.format(optimizer))


class DDPG:

    def __init__(self, state_size, action_size, h_params, device):
        """
        trains an agent using DDPG
        :param state_size:
        :param action_size:
        :param h_params:
        :param device:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.h_params = h_params
        # actor network
        self.actor_policy_net = Actor(state_size,action_size,h_params).cuda()
        self.actor_target_net = Actor(state_size, action_size, h_params).cuda()
        self.actor_target_net.load_state_dict(self.actor_policy_net.state_dict())
        self.actor_target_net.eval()

        # critic network
        self.critic_net = Critic(state_size,action_size,h_params).cuda()
        self.critic_target_net = Critic(state_size, action_size, h_params).cuda()
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.critic_target_net.eval()

        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.set_optimizer(h_params)
        self.memory_far = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.memory_close = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.device = device

    def train_agent(self):
        """
        trains the network based on the data that has been collected via the episodes
        :return:
        """

        if len(self.memory_far) <= self.h_params['learning_agent']['batch_size'] or len(self.memory_close) <= \
                self.h_params['learning_agent']['batch_size']:
            return

        with torch.autograd.set_detect_anomaly(True):
            loss = None
            c = 0
            while c < self.h_params['learning_agent']['n_batches']:

                self.train_batch()
                #print(c)
                c += 1

    def train_batch(self):
        far_sample_size = int(
            self.h_params['learning_agent']['batch_size'] * self.h_params['learning_agent']['memory_split'])
        transitions_far = self.memory_far.sample(far_sample_size)
        batch_far = self.transition(*zip(*transitions_far))
        state_batch_far = torch.cat(batch_far.state)
        action_batch_far = torch.cat(batch_far.action)
        reward_batch_far = torch.cat(batch_far.reward)
        next_state_far = torch.cat(batch_far.next_state)
        #done_tensor = torch.tensor(batch_far.done)
        #done_far = torch.reshape(done_tensor, (list(done_tensor.size())[0], 1))
        done_far = torch.FloatTensor(1-np.array(batch_far.done)).cuda()

        transitions_close = self.memory_close.sample(self.h_params['learning_agent']['batch_size'] - far_sample_size)
        batch_close = self.transition(*zip(*transitions_close))
        state_batch_close = torch.cat(batch_close.state)
        action_batch_close = torch.cat(batch_close.action)  # torch.cat(batch_close.action)
        reward_batch_close = torch.cat(batch_close.reward)
        next_state_close = torch.cat(batch_close.next_state)
        #done_tensor = torch.tensor(batch_close.done)
        #done_close = torch.reshape(done_tensor, (list(done_tensor.size())[0], 1))
        done_close = torch.FloatTensor(1-np.array(batch_close.done)).cuda()

        state_batch = torch.cat((state_batch_far, state_batch_close))
        action_batch = torch.cat((action_batch_far, action_batch_close))
        reward_batch = torch.cat((reward_batch_far, reward_batch_close))
        next_state_batch = torch.cat((next_state_far, next_state_close))
        done_batch = torch.cat((done_far, done_close))

        target_Q_init = self.critic_target_net(torch.cat([next_state_batch, self.actor_target_net(next_state_batch)], dim=1))
        #target_Q_init = torch.reshape()
        done_batch = torch.reshape(done_batch,(len(done_batch),1))
        target_Q = reward_batch + (done_batch * self.h_params['learning_agent']['gamma'] * target_Q_init).detach()

        #inp = torch.cat([state_batch, action_batch], dim=1).to(torch.float)
        #current_Q = self.critic_net(torch.cat([state_batch, self.actor_policy_net(state_batch)], dim=1))
        #k =
        current_Q = self.critic_net(torch.cat([state_batch, action_batch], dim=1))
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_net(torch.cat([state_batch, self.actor_policy_net(state_batch)], dim=1)).mean()
        self.actor_policy_net.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network()

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def get_action_train(self, state, n_steps, bg_noise=[]):
        action_tensor = self.actor_policy_net(state).detach()
        action_np = action_tensor.cpu().data.numpy().flatten()
        #action_np = action_tensor.cpu().detach().numpy()

        # add some random noise to the actions
        sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * n_steps / self.eps_decay)
        if n_steps > self.h_params['learning_agent']['epsilon_decay']:
            eps_threshold = self.h_params['learning_agent']['epsilon_end']
        else:
            eps_threshold = (self.h_params['learning_agent']['epsilon_end'] - self.h_params['learning_agent'][
                'epsilon_start']) / self.h_params['learning_agent']['epsilon_decay'] \
                            * n_steps + self.h_params['learning_agent']['epsilon_start']
        self.h_params['learning_agent']['epsilon_threshold'] = eps_threshold

        # TODO check that random noise is added correctly. thr level of noise may not be correct
        if sample < eps_threshold:
            # add noise to the action

            #ma = self.actor_policy_net.max_action.cpu().detach().numpy()
            sigma = self.actor_policy_net.max_action.cpu().detach().numpy()/1.645
            sigma = np.reshape(sigma,len(sigma[0]))
            base_random = action_np + np.random.normal(0, sigma, size=self.action_size)
            #base_random = action_tensor + sigma*torch.randn(self.action_size).cuda()
            #action = base_random.clip(-self.actor_policy_net.max_action.cpu().detach().numpy(), self.actor_policy_net.max_action.cpu().detach().numpy())
            action = np.clip(base_random,-self.actor_policy_net.max_action.cpu().detach().numpy(), self.actor_policy_net.max_action.cpu().detach().numpy())

            #noise = self.ou_noise(bg_noise,dt=0.25, dim=self.action_size)
            #action = np.clip(action.cpu().detach().numpy() + noise, -self.actor_policy_net.max_action.cpu().detach().numpy(), self.actor_policy_net.max_action.cpu().detach().numpy())

            action_tensor = torch.from_numpy(action).cuda().to(torch.float32)


        state_action = torch.concat((state,action_tensor),dim=1)
        return action_tensor, self.critic_net(state_action.float()).cpu().detach().numpy()

    def get_action_eval(self, state):
        with torch.no_grad():
            action = self.actor_policy_net(state)
            #action_tensor = torch.from_numpy(action).cuda()
            state_action = torch.concat((state, action), dim=1)

        return action, self.critic_net(state_action.float()).cpu().detach().numpy()

    def update_target_network(self):
        """
        copies the values of the policy network into the target network
        :return:
        """
        tau = self.h_params['learning_agent']['tau']
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor_policy_net.parameters(), self.actor_target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def set_optimizer(self, h_params):
        """
        set the optimizer of the learning algorithm
        :param h_params:
        :return:
        """
        optimizer = h_params['learning_agent']['optimizer']

        if optimizer == 'adam':
            self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=h_params['learning_agent']['learn_rate'],
                                        betas=(h_params['learning_agent']['beta_1'], h_params['learning_agent']['beta_2']))
            self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), lr=h_params['learning_agent']['learn_rate'],
                                        betas=(
                                        h_params['learning_agent']['beta_1'], h_params['learning_agent']['beta_2']))
        elif optimizer == 'rms_prop':
            self.critic_optimizer = optim.RMSprop(self.critic_net.parameters(),lr=h_params['learning_agent']['learn_rate'],
                                           alpha=h_params['learning_agent']['alpha'],momentum=h_params['learning_agent']['momentum'])
            self.actor_optimizer = optim.RMSprop(self.actor_policy_net.parameters(), lr=h_params['learning_agent']['learn_rate'],
                                           alpha=h_params['learning_agent']['alpha'],
                                           momentum=h_params['learning_agent']['momentum'])
        else:
            raise ValueError('Optimizer {} is currently not supported'.format(optimizer))

    def add_to_memory(self,state,action,next_state,reward,done, is_close):

        if is_close:
            self.memory_close.push(state,action,next_state,reward,done)
        else:
            self.memory_far.push(state,action,next_state,reward,done)


class PPO:

    def __init__(self, state_size, action_size, h_params, device):
        """
        trains an agent using the PPO algorithm
        :param state_size:
        :param action_size:
        :param h_params:
        :param device:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.h_params = h_params
        # actor network
        self.actor_net = ActorTwoHeads(state_size,action_size,h_params).cuda()

        # critic network
        self.critic_net = Critic(state_size,action_size,h_params).cuda()

        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done', 'a_log_prob'))
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.set_optimizer(h_params)
        self.memory_far = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.memory_close = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.device = device