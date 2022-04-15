

# native modules
from collections import namedtuple, OrderedDict
from unittest import TestCase

# third party modules
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

# own modules
import Environment
import Learn_Algorithms
import Movers
import Sensors


class TestEnvironment(TestCase):

    def test_obstacle_reward(self):
        """
        tests how the reward of a boat varies around an obstacle
        :return:
        """

        oo = Environment.OneObstacles()
        oo.h_params = dict()
        oo.h_params['reward'] = dict()
        oo.h_params['reward']['angle'] = 0.05
        oo.h_params['reward']['obs'] = 100.0

        boat = Movers.UtilityBoat.get_default('boat', 0.25,'propeller_and_power')
        boat.destination = [100.0,20.0]
        lidar = Sensors.Lidar('boat',2,100.0)
        boat.add_sensor(lidar)
        obs_1 = Movers.StaticCircle('obs_1',50)
        obs_1.pos = [70.0,20.0]
        obs_1.radius = 10.0
        #obs_2 = Movers.StaticCircle('obs_2', 50)
        #obs_2.pos = [100.0, 25.0]
        #obs_2.radius = 10.0

        mover_dict = OrderedDict()
        mover_dict[boat.name] = boat
        mover_dict[obs_1.name] = obs_1
        #mover_dict[obs_2.name] = obs_2

        x_v = np.linspace(0,100,401)
        y_v = np.linspace(0,40,201)

        reward = np.zeros((len(x_v),len(y_v)))
        x = np.zeros_like(reward)
        y = np.zeros_like(reward)
        psi = np.zeros_like(reward)
        psi_2 = np.zeros_like(reward)
        psi_3 = np.zeros_like(reward)

        for i in range(len(x_v)):
            for j in range(len(y_v)):

                x[i,j] = x_v[i]
                y[i, j] = y_v[j]

                boat.phi = 0.0
                boat.x = x[i,j]
                boat.y = y[i,j]

                if x[i,j] == 20.0 and (y[i,j] == 30.0 or y[i,j] == 20.0 or y[i,j] == 10.0):
                    check = 0

                boat.update_sensors(mover_dict)
                boat.calc_dest_metrics()
                oo.old['mu'] = boat.mu
                oo.old['phi'] = boat.phi
                for sensor in boat.sensors:
                    tmp_state = sensor.get_state()
                    oo.old['theta_obs_1'] = tmp_state['angle_center_0']
                    #to.old['theta_obs_2'] = tmp_state['angle_center_1']
                oo.old['x'] = boat.x
                oo.old['y'] = boat.y

                theta = oo.old['theta_obs_1']
                if theta > np.pi:
                    theta -= 2.0 * np.pi
                phi = boat.phi
                if phi > np.pi:
                    phi -= 2.0 * np.pi
                psi[i, j] = theta - phi

                # change the boat angle to determine if that helps the reward
                boat.phi = np.deg2rad(-10.0)
                boat.calc_dest_metrics()
                boat.update_sensors(mover_dict)

                reward[i,j] = oo.reward_angle_to_obstacle(boat,obs_1,0.0)

                theta = oo.old['theta_obs_1']
                if theta > np.pi:
                    theta -= 2.0 * np.pi
                phi = boat.phi
                if phi > np.pi:
                    phi -= 2.0 * np.pi
                psi_2[i, j] = theta - phi

                psi_3[i,j] = psi_2[i,j] - psi[i,j]

        """
        fig = plt.figure(0)
        ax1 = fig.add_subplot(1,1,1)
        cs = ax1.contour(x,y,np.rad2deg(psi),11)
        plt.colorbar(cs,ax=ax1)
        ax1.clabel(cs)

        circle = patches.Circle((boat.destination[0], boat.destination[1]), radius=2.0, alpha=0.5,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((obs_1.pos[0],obs_1.pos[1]), radius=obs_1.radius, alpha=0.5,
                                color='tab:red')
        ax1.add_patch(circle)
        #circle = patches.Circle((obs_2.pos[0], obs_2.pos[1]), radius=obs_2.radius, alpha=0.5,color='tab:red')
        #ax1.add_patch(circle)
        ax1.grid()

        
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 1, 1)
        cs = ax1.contour(x, y, np.rad2deg(psi_2), 11)
        plt.colorbar(cs, ax=ax1)
        ax1.clabel(cs)

        circle = patches.Circle((boat.destination[0], boat.destination[1]), radius=2.0, alpha=0.5,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((obs_1.pos[0], obs_1.pos[1]), radius=obs_1.radius, alpha=0.5,
                                color='tab:red')
        ax1.add_patch(circle)
        # circle = patches.Circle((obs_2.pos[0], obs_2.pos[1]), radius=obs_2.radius, alpha=0.5,color='tab:red')
        # ax1.add_patch(circle)
        ax1.grid()

        fig = plt.figure(2)
        ax1 = fig.add_subplot(1, 1, 1)
        cs = ax1.contour(x, y, np.rad2deg(psi_3), 11)
        plt.colorbar(cs, ax=ax1)
        ax1.clabel(cs)

        circle = patches.Circle((boat.destination[0], boat.destination[1]), radius=2.0, alpha=0.5,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((obs_1.pos[0], obs_1.pos[1]), radius=obs_1.radius, alpha=0.5,
                                color='tab:red')
        ax1.add_patch(circle)
        # circle = patches.Circle((obs_2.pos[0], obs_2.pos[1]), radius=obs_2.radius, alpha=0.5,color='tab:red')
        # ax1.add_patch(circle)
        ax1.grid()
        """
        fig = plt.figure(3)
        ax1 = fig.add_subplot(1, 1, 1)
        cs = ax1.contourf(x, y, reward, 11)
        plt.colorbar(cs, ax=ax1)
        #ax1.clabel(cs,colors='white')

        circle = patches.Circle((boat.destination[0], boat.destination[1]), radius=2.0, alpha=0.5,
                                color='tab:green')
        ax1.add_patch(circle)
        circle = patches.Circle((obs_1.pos[0], obs_1.pos[1]), radius=obs_1.radius, alpha=0.5,
                                color='tab:red')
        ax1.add_patch(circle)
        # circle = patches.Circle((obs_2.pos[0], obs_2.pos[1]), radius=obs_2.radius, alpha=0.5,color='tab:red')
        # ax1.add_patch(circle)
        ax1.grid()

        plt.show()

    def test_replay_storage_initialization(self):
        """
        tests the three replay buffers strategies are set up correctly given the input files. Checks the number of
        buffers and the members for interacting with the buffers are setup correctly.

        :return:
        """

        # test initialization of a replay storage when only using one replay buffer
        file_name = 'storage_test_params_all_in_one.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)

        rs = Learn_Algorithms.ReplayStorage(h_params)

        # check the replay storage has only one replay buffer
        num_buffers = len(rs.buffers)
        self.assertEqual(num_buffers,1)

        # check the replay storage has an empty interim buffer
        interim_buffer = rs.interim_buffer
        self.assertEqual(len(interim_buffer),0)

        # check the position of the buffer is at 0
        position_rb = rs.position['only']
        self.assertEqual(position_rb,0)
        position_rb = rs.position['interim']
        self.assertEqual(position_rb, 0)

        # test initialization of a replay storage with two replay buffers. one for being close to an obstacle and one
        # being far from the obstacle
        file_name = 'storage_test_params_proximity.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)

        rs = Learn_Algorithms.ReplayStorage(h_params)

        # check the replay storage has only two replay buffers
        num_buffers = len(rs.buffers)
        self.assertEqual(num_buffers, 2)

        # check the replay storage has an empty interim buffer
        interim_buffer = rs.interim_buffer
        self.assertEqual(len(interim_buffer), 0)

        # check the position of the buffer is at 0
        position_rb = rs.position['close']
        self.assertEqual(position_rb, 0)
        position_rb = rs.position['far']
        self.assertEqual(position_rb, 0)
        position_rb = rs.position['interim']
        self.assertEqual(position_rb, 0)

        # test initialization of a replay storage with three replay buffers. one for each potential outcome of the
        # simulation
        file_name = 'storage_test_params_outcome.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)

        rs = Learn_Algorithms.ReplayStorage(h_params)

        # check the replay storage has only two replay buffers
        num_buffers = len(rs.buffers)
        self.assertEqual(num_buffers, 3)

        # check the replay storage has an empty interim buffer
        interim_buffer = rs.interim_buffer
        self.assertEqual(len(interim_buffer), 0)

        # check the position of the buffer is at 0
        position_rb = rs.position['crash']
        self.assertEqual(position_rb, 0)
        position_rb = rs.position['success']
        self.assertEqual(position_rb, 0)
        position_rb = rs.position['other']
        self.assertEqual(position_rb, 0)
        position_rb = rs.position['interim']
        self.assertEqual(position_rb, 0)

    def test_replay_storage_push_and_sort(self):


        # test initialization of a replay storage when only using one replay buffer
        file_name = 'storage_test_params_all_in_one.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(0, 1, 2, 3, True)
            else:
                rs.push(0,1,2,3,False)

            k += 1

        # check the length of the interim buffer is of length 10
        self.assertEqual(len(rs.interim_buffer),10)

        # call the sort function to sort data into the correct buffers. All should be in one buffer here. The length
        # of only should be 10
        rs.sort_data_into_buffers()

        self.assertEqual(len(rs.buffers['only'].memory), 10)

        # check that the interim buffer is empty
        self.assertEqual(len(rs.interim_buffer.memory), 0)

        # --------------------------------------------------------------------------------------------------------------
        # test initialization of a replay storage when sorting between close and far from an obstacle
        file_name = 'storage_test_params_proximity.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(0, 1, 2, 3, True, 8.0)
            else:
                rs.push(0, 1, 2, 3, False, 20.0)

            k += 1

        # check the length of the interim buffer is of length 10
        self.assertEqual(len(rs.interim_buffer), 10)

        # call the sort function to sort data into the correct buffers. There should be 9 data points in the far replay
        # buffer and 1 data point in the close buffer
        rs.sort_data_into_buffers()

        self.assertEqual(len(rs.buffers['far'].memory), 9)
        self.assertEqual(len(rs.buffers['close'].memory), 1)

        self.assertEqual(rs.position['far'], 9)
        self.assertEqual(rs.position['close'], 1)

        # --------------------------------------------------------------------------------------------------------------
        # test initialization of a replay storage when sorting between the outcome of the simulation
        file_name = 'storage_test_params_outcome.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(0, 1, 2, 3, True, 'success')
            else:
                rs.push(0, 1, 2, 3, False, 'None')

            k += 1

        # check the length of the interim buffer is of length 10
        self.assertEqual(len(rs.interim_buffer), 10)

        # call the sort function to sort data into the correct buffers. There should be 9 data points in the far replay
        # buffer and 1 data point in the close buffer
        rs.sort_data_into_buffers()

        self.assertEqual(len(rs.buffers['success'].memory), 10)
        self.assertEqual(len(rs.buffers['crash'].memory), 0)
        self.assertEqual(len(rs.buffers['other'].memory), 0)

        self.assertEqual(rs.position['success'], 10)
        self.assertEqual(rs.position['crash'], 0)
        self.assertEqual(rs.position['other'], 0)

        file_name = 'storage_test_params_outcome.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(0, 1, 2, 3, True, 'crash')
            else:
                rs.push(0, 1, 2, 3, False, 'None')

            k += 1

        # check the length of the interim buffer is of length 10
        self.assertEqual(len(rs.interim_buffer), 10)

        # call the sort function to sort data into the correct buffers. There should be 9 data points in the far replay
        # buffer and 1 data point in the close buffer
        rs.sort_data_into_buffers()

        self.assertEqual(len(rs.buffers['success'].memory), 0)
        self.assertEqual(len(rs.buffers['crash'].memory), 10)
        self.assertEqual(len(rs.buffers['other'].memory), 0)

        self.assertEqual(rs.position['success'], 0)
        self.assertEqual(rs.position['crash'], 10)
        self.assertEqual(rs.position['other'], 0)

        file_name = 'storage_test_params_outcome.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(0, 1, 2, 3, True, 'other')
            else:
                rs.push(0, 1, 2, 3, False, 'None')

            k += 1

        # check the length of the interim buffer is of length 10
        self.assertEqual(len(rs.interim_buffer), 10)

        # call the sort function to sort data into the correct buffers. There should be 9 data points in the far replay
        # buffer and 1 data point in the close buffer
        rs.sort_data_into_buffers()

        self.assertEqual(len(rs.buffers['success'].memory), 0)
        self.assertEqual(len(rs.buffers['crash'].memory), 0)
        self.assertEqual(len(rs.buffers['other'].memory), 10)

        self.assertEqual(rs.position['success'], 0)
        self.assertEqual(rs.position['crash'], 0)
        self.assertEqual(rs.position['other'], 10)

    def test_replay_storage_sample(self):

        # test initialization of a replay storage when only using one replay buffer
        file_name = 'storage_test_params_all_in_one.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True))
            else:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False))

            k += 1

        rs.sort_data_into_buffers()
        batch = rs.sample(10)

        self.assertEqual(len(batch),10)

        # test initialization of a replay storage when two buffers are used based on the proximity
        file_name = 'storage_test_params_proximity.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k >= 8:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True), 8.0)
            else:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False), 20.0)

            k += 1

        rs.sort_data_into_buffers()
        batch = rs.sample(10)

        self.assertEqual(len(batch), 10)
        n_far = 0
        n_close = 0
        for tmp_sample in batch:
            prox = tmp_sample.prox
            if prox > 10.0:
                n_far += 1
            elif prox < 10.0:
                n_close += 1
        self.assertEqual(n_close,2)
        self.assertEqual(n_far, 8)

        # add more data to the close proximity
        rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True), 8.0)
        rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True), 8.0)
        rs.sort_data_into_buffers()
        batch = rs.sample(10)
        self.assertEqual(len(batch), 10)
        n_far = 0
        n_close = 0
        for tmp_sample in batch:
            prox = tmp_sample.prox
            if prox > 10.0:
                n_far += 1
            elif prox < 10.0:
                n_close += 1
        self.assertEqual(n_close, 3)
        self.assertEqual(n_far, 7)

        # --------------------------------------------------------------------------------------------------------------
        # test initialization of a replay storage when sorting between the outcome of the simulation
        file_name = 'storage_test_params_outcome.yml'
        h_params = Environment.Environment.load_hyper_params(file_name)
        rs = Learn_Algorithms.ReplayStorage(h_params)

        # make some arbitrary data tuples and push them to the replay storage
        k = 0
        while k < 10:

            if k == 9:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True), 'success')
            else:
                rs.push(torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False), ('None'))

            k += 1

        rs.sort_data_into_buffers()
        batch = rs.sample(10)

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(rs.buffers['success']),10)
        self.assertEqual(len(rs.buffers['crash']), 0)
        self.assertEqual(len(rs.buffers['other']), 0)
        # step through each sample and check the first part is 0
        n_success = 0
        for tmp_batch in batch:
            state = tmp_batch.state
            if state == 0:
                n_success += 1
        self.assertEqual(n_success,10)

        # make some arbitrary data tuples and push them to the replay storage, but this time data is added to the other
        # replay buffer
        k = 0
        while k < 4:

            if k == 3:
                rs.push(torch.tensor(-1), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True), 'other')
            else:
                rs.push(torch.tensor(-1), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False), 'None')

            k += 1

        rs.sort_data_into_buffers()

        batch = rs.sample(10)

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(rs.buffers['success']), 10)
        self.assertEqual(len(rs.buffers['crash']), 0)
        self.assertEqual(len(rs.buffers['other']), 4)

        # step through each sample and count for the number of data points in the batch that are from the success and
        # other replay buffers
        n_success = 0
        n_other = 0
        for tmp_batch in batch:
            state = tmp_batch.state
            if state == 0:
                n_success += 1
            elif state == -1:
                n_other += 1
        self.assertEqual(n_success, 6)
        self.assertEqual(n_other, 4)

        # make some arbitrary data tuples and push them to the replay storage, but this time data is added to the crash
        # replay buffer
        k = 0
        while k < 2:

            if k == 1:
                rs.push(torch.tensor(-10), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(True),
                        'crash')
            else:
                rs.push(torch.tensor(-10), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False),
                        'None')

            k += 1

        rs.sort_data_into_buffers()

        batch = rs.sample(10)

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(rs.buffers['success']), 10)
        self.assertEqual(len(rs.buffers['crash']), 2)
        self.assertEqual(len(rs.buffers['other']), 4)

        # step through each sample and count for the number of data points in the batch that are from the success and
        # other replay buffers
        n_success = 0
        n_other = 0
        n_crash = 0
        for tmp_batch in batch:
            state = tmp_batch.state
            if state == 0:
                n_success += 1
            elif state == -1:
                n_other += 1
            elif state == -10:
                n_crash += 1
        self.assertEqual(n_success, 4)
        self.assertEqual(n_other, 4)
        self.assertEqual(n_crash, 2)

        # make some arbitrary data tuples and push them to the replay storage, but this time data is added to the crash
        # replay buffer making all of buffers long enough to pull an entire desired portion from each buffer

        rs.push(torch.tensor(-10), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False),'None')
        rs.push(torch.tensor(-10), torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(False), 'crash')

        rs.sort_data_into_buffers()

        batch = rs.sample(10)

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(rs.buffers['success']), 10)
        self.assertEqual(len(rs.buffers['crash']), 4)
        self.assertEqual(len(rs.buffers['other']), 4)

        # step through each sample and count for the number of data points in the batch that are from the success and
        # other replay buffers
        n_success = 0
        n_other = 0
        n_crash = 0
        for tmp_batch in batch:
            state = tmp_batch.state
            if state == 0:
                n_success += 1
            elif state == -1:
                n_other += 1
            elif state == -10:
                n_crash += 1
        self.assertEqual(n_success, 3)
        self.assertEqual(n_other, 3)
        self.assertEqual(n_crash, 4)


