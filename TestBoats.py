
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

import Movers


class TestBoat(TestCase):

    def test_ub_step(self):
        """
        basic test to verify that the angles and distances are calculated before and after stepping
        :return:
        """
        delta_t = 0.25
        ub = Movers.UtilityBoat.get_default(delta_t)

        c = 1
        while c < 5:
            ub.pre_step()

            mu = ub.mu
            alpha = ub.alpha
            dest_dist = ub.destination_distance
            v_x_p = ub.v_boat_x_p
            v_y_p = ub.v_boat_y_p

            ub.set_control(10000.0, np.deg2rad(c))
            ub.step()

            # some angles should not be updated yet
            self.assertEqual(mu,ub.mu)
            self.assertEqual(alpha, ub.alpha)
            self.assertEqual(dest_dist, ub.destination_distance)
            self.assertEqual(v_x_p, ub.v_boat_x_p)

            ub.post_step()

            self.assertNotEqual(mu, ub.mu)
            self.assertNotEqual(alpha, ub.alpha)
            self.assertNotEqual(dest_dist, ub.destination_distance)
            self.assertNotEqual(v_x_p, ub.v_boat_x_p)

            c += 1

    def test_ub_turn(self):

        delta = np.deg2rad(np.linspace(0,360,37))
        phi_after = np.zeros_like(delta)
        phi_before = np.zeros_like(delta)

        for i in range(len(delta)):

            delta_t = 0.25
            ub = Movers.UtilityBoat.get_default(delta_t)

            ub.set_control(10000.0, delta[i])

            phi_before[i] = ub.phi
            if phi_before[i] > np.pi:
                phi_before[i] -= 2.0*np.pi

            ub.step()

            phi_after[i] = ub.phi
            if phi_after[i] > np.pi:
                phi_after[i] -= 2.0*np.pi

        fig = plt.figure(0)
        plt.plot(np.rad2deg(delta),np.rad2deg(phi_before),label='before')
        plt.plot(np.rad2deg(delta), np.rad2deg(phi_after), label='after')
        plt.legend()
        plt.grid()
        plt.show()

        # run the calculations to get distance so the boat


    def test_gps(self):

        # x, y, radius
        #obstacles = [[4000,16000,1900],[8000,5000,1900]]
        #bounds = [[0,10000],[0,20000]]
        obstacles = [[40,160,19],[80,50,19]]
        domain = 50
        n_bins = [15,15]
        gps = Movers.GPS(obstacles, domain, n_bins)

        boat = Movers.UtilityBoat.get_default(0.5)
        boat.x = 60.0 #2000.0
        boat.y = 50.0 #2000.0
        boat.phi = np.deg2rad(70.0)

        gps.update_mask(boat)

        import matplotlib.patches as patches

        fig = plt.figure(0)
        ax = fig.add_subplot(111)

        circle = patches.Circle((obstacles[0][0], obstacles[0][1]), radius=obstacles[0][2], color='tab:red', alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[0][0], obstacles[0][1]), radius=obstacles[0][2]*1.1, color='tab:olive', alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[1][0], obstacles[1][1]), radius=obstacles[1][2],color='tab:red', alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[1][0], obstacles[1][1]), radius=obstacles[1][2] * 1.1, color='tab:olive',
                                alpha=0.3)
        ax.add_patch(circle)

        ax1_len = len(gps.data[0, :, 0])
        ax2_len = len(gps.data[0, 0, :])
        for k in range(ax1_len):
            for j in range(ax2_len):
                cen_x = gps.data[0, k, j]
                cen_y = gps.data[1, k, j]

                tmp_x = cen_x*np.cos(boat.phi)-cen_y*np.sin(boat.phi) + boat.x
                tmp_y = cen_x * np.sin(boat.phi) + cen_y * np.cos(boat.phi) + boat.y

                mask = gps.data[2, k, j]

                if mask == 0:
                    color = 'tab:blue'
                    ec = 'tab:blue'
                else:
                    color = 'tab:orange'
                    ec = 'tab:orange'

                # rotate the corner anchor
                dx = - gps.x_buffer
                dy = - gps.y_buffer
                adj_x = dx*np.cos(boat.phi) - dy*np.sin(boat.phi)
                adj_y = dx * np.sin(boat.phi) + dy * np.cos(boat.phi)

                rect = patches.Rectangle((tmp_x +adj_x, tmp_y + adj_y),
                                         2.0 * gps.x_buffer,
                                         2.0 * gps.y_buffer, angle=np.rad2deg(boat.phi), ec=ec, fc=color, alpha=0.3)
                ax.add_patch(rect)

        ax.set_xlim([0,100])
        ax.set_ylim([-50,200])
        ax.scatter(boat.x,boat.y,marker='x',color='k')
        ax.grid()

        obstacles = [[40, 160, 19], [80, 50, 19]]
        domain = 50
        n_bins = [15, 15]
        gps = Movers.GPS(obstacles, domain, n_bins)

        boat = Movers.UtilityBoat.get_default(0.5)
        boat.x = 30.0  # 2000.0
        boat.y = 130.0  # 2000.0
        boat.phi = np.deg2rad(0.0)

        gps.update_mask(boat)

        import matplotlib.patches as patches

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        circle = patches.Circle((obstacles[0][0], obstacles[0][1]), radius=obstacles[0][2], color='tab:red', alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[0][0], obstacles[0][1]), radius=obstacles[0][2] * 1.1, color='tab:olive',
                                alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[1][0], obstacles[1][1]), radius=obstacles[1][2], color='tab:red', alpha=0.3)
        ax.add_patch(circle)
        circle = patches.Circle((obstacles[1][0], obstacles[1][1]), radius=obstacles[1][2] * 1.1, color='tab:olive',
                                alpha=0.3)
        ax.add_patch(circle)

        ax1_len = len(gps.data[0, :, 0])
        ax2_len = len(gps.data[0, 0, :])
        for k in range(ax1_len):
            for j in range(ax2_len):
                cen_x = gps.data[0, k, j]
                cen_y = gps.data[1, k, j]

                tmp_x = cen_x * np.cos(boat.phi) - cen_y * np.sin(boat.phi) + boat.x
                tmp_y = cen_x * np.sin(boat.phi) + cen_y * np.cos(boat.phi) + boat.y

                mask = gps.data[2, k, j]

                if mask == 0:
                    color = 'tab:blue'
                    ec = 'tab:blue'
                else:
                    color = 'tab:orange'
                    ec = 'tab:orange'

                # rotate the corner anchor
                dx = - gps.x_buffer
                dy = - gps.y_buffer
                adj_x = dx * np.cos(boat.phi) - dy * np.sin(boat.phi)
                adj_y = dx * np.sin(boat.phi) + dy * np.cos(boat.phi)

                rect = patches.Rectangle((tmp_x + adj_x, tmp_y + adj_y),
                                         2.0 * gps.x_buffer,
                                         2.0 * gps.y_buffer, angle=np.rad2deg(boat.phi), ec=ec, fc=color, alpha=0.3)
                ax.add_patch(rect)

        ax.set_xlim([0, 100])
        ax.set_ylim([-50, 200])
        ax.scatter(boat.x, boat.y, marker='x', color='k')
        ax.grid()

        plt.show()

    def test_dist_and_time_to_turn(self):
        """
        a test that determines how fast a boat can turn itself to avoid crashing into an obstacle. THis is done
        by determining how many time steps it takes to turn the boat 90 degs traveling at maximum speed
        :return:
        """

        fig = plt.figure(0)
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2, 1, 2)

        turn_rate = [1,2,3,4,5]
        for tr in turn_rate:
            delta_t = 0.25
            boat = Movers.UtilityBoat.get_default(0.5)
            boat.power = 20000.0
            boat.v_boat_x = 4.0
            angle = [boat.phi]
            t = 0
            time = [t]

            while boat.phi < np.pi/2.0:
                prop_angle = boat.delta
                boat.set_control(20000.0,prop_angle+np.deg2rad(-tr))
                boat.step()
                angle.append(boat.phi)
                t += delta_t
                time.append(t)

            ax1.plot(time,np.rad2deg(angle),'o-',label=tr)
            ax2.plot([i for i in range(len(angle))], np.rad2deg(angle), 'o-', label=tr)

        ax1.plot([0,5],[90,90],'k--')
        ax1.legend()
        ax1.grid()
        ax2.plot([0, 20], [90, 90], 'k--')
        ax1.set_xlabel('Simulation time [s]')
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel('Number of time steps')
        plt.tight_layout()
        plt.show()



    def test_gpu_load(self):
        import torch
        print('available:\t',torch.cuda.is_available(),torch.__version__)

    def test_propogate_crash_reward(self):

        num_steps = np.linspace(1,50,50)
        crash_reward = -10.0 # -10.0
        prox_pen = -2.0 # -2.0
        prox = 10  # [m]
        gamma = [0.9,0.95,0.99]
        delta_t = 0.25  # [s]
        vel = 4  # [m/s]

        for j in range(len(gamma)):

            felt_reward = []

            for i in range(len(num_steps)):

                # add reward for closing distance to destination at optimal manner
                fr = 0
                for k in range(int(num_steps[i])):
                    fr += np.power(gamma[j],k)*(1+1+1) # velocity, destination, angle

                # add proximity award
                prox_reward = 0
                k = i
                while k >=0:

                    if k < prox/(vel*delta_t):
                        prox_reward += np.power(gamma[j],num_steps[i]-k)*prox_pen
                    k-= 1
                fr += prox_reward

                # add crashing reward
                felt_reward.append(fr+np.power(gamma[j],num_steps[i])*crash_reward)

            plt.plot(num_steps,felt_reward,label=gamma[j])

        plt.xlabel('Time steps away from crashing')
        plt.ylabel('Reward felt from a future crash')
        plt.grid()
        plt.legend()
        plt.show()

    def test_propogate_single_reward(self):

        final_reward = 1000
        gamma = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
        num_steps = np.linspace(0,100,101)
        felt_reward = np.zeros_like(num_steps)
        for j in range(len(gamma)):
            for i in range(len(num_steps)):
                felt_reward[i] = np.power(gamma[j],num_steps[i])*final_reward

            plt.plot(num_steps,felt_reward,label='gamma: '+str(gamma[j]))
        plt.legend()
        plt.grid()
        plt.xlabel('Number of steps')
        plt.ylabel('Reward felt')
        plt.show()

    def test_lidar(self):
        """
        is a unit test to verify that the lidar angles are calculated correctly
        :return:
        """

        delta_t = 0.25
        boat = Movers.UtilityBoat.get_default(delta_t)
        lidar = Movers.Lidar(1, 100)
        obstacle = np.zeros((1,3))
        obstacle[0,2] = 10 # set obstacle radius
        move_radius = 11

        # angles to move obstacle around the boat that is at 0,0
        angles = np.linspace(0,np.pi*2.0,37)
        x = np.zeros_like(angles)
        y = np.zeros_like(angles)
        d_calc = np.zeros_like(angles) # calculated distance the sensor sees
        theta_calc = np.zeros_like(angles)  # calculated angle from the boat to the center of the obstacle
        port_calc = np.zeros_like(angles)
        starboard_calc = np.zeros_like(angles)

        for i in range(len(angles)):

            x[i] = move_radius*np.cos(angles[i])
            y[i] = move_radius*np.sin(angles[i])

            obstacle[0,0] = move_radius*np.cos(angles[i])
            obstacle[0,1] = move_radius * np.sin(angles[i])

            lidar.update_measurements(boat, obstacle)

            d_calc[i] = lidar.measurments[0,0]

            theta_calc[i] = lidar.measurments[0,4]
            starboard_calc[i] = lidar.measurments[0,5]
            port_calc[i] = lidar.measurments[0, 6]

        fig = plt.figure(0,figsize=(14,8))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4,1,2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        ax1.plot(angles,x,label='x')
        ax1.plot(angles, y, label='y')
        ax1.grid()
        ax1.legend()

        #ax1.plot(angles, move_radius*np.ones_like(angles),label='truth')
        ax2.plot(np.rad2deg(angles),d_calc,label='calc')
        ax2.grid()
        ax2.legend()

        ax3.plot(np.rad2deg(angles),np.rad2deg(angles),label="truth")
        ax3.plot(np.rad2deg(angles),np.rad2deg(theta_calc),'--',label="calc")
        ax3.grid()
        ax3.legend()

        ax4.plot(np.rad2deg(angles), np.rad2deg(theta_calc), label="calc")
        ax4.plot(np.rad2deg(angles),np.rad2deg(port_calc),label='port')
        ax4.plot(np.rad2deg(angles), np.rad2deg(starboard_calc), label='starboard')
        ax4.grid()
        ax4.legend()

        plt.show()

