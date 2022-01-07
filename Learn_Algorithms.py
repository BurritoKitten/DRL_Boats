
from collections import namedtuple
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):

    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition

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
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.out(x)
        return x  # psudeo linear activation


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
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['learn_rate'],betas=(h_params['beta_1'], h_params['beta_2']))
        self.memory = ReplayMemory(self.h_params['learning_agent']['memory_capacity'], self.transition)
        self.device = device # choose cpu or gpu

    def train_agent(self):
        """
        train/optimize the networks
        :return:
        """

        loss = None
        c = 0
        while c < self.h_params['learning_agent']['n_batches']:

            if len(self.memory) < self.h_params['learning_agent']['batch_size']:
                return
            transitions = self.memory.sample(self.h_params['learning_agent']['batch_size'])
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
            # gps_batch = torch.cat(batch.gps)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

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
            #tmp = torch.mul(torch.reshape(next_state_values,(len(next_state_values),1)) , self.h_params['learning_agent']['gamma'])
            #check = torch.reshape(next_state_values,(len(next_state_values),1))
            #tmp_2 = torch.add(tmp, check)
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

    def get_action_train(self, state, n_steps):
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