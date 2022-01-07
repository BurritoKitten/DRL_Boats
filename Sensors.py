
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy

import numpy as np


class Sensor:

    def __init__(self):
        pass

    @abstractmethod
    def update_measurements(self):
        raise NotImplementedError('The sensor needs to implement this method')


class Lidar:

    def __init__(self, mover_name, num_objects, max_dist):
        """
        the lidar sensor simulates processed data from a lidar sensor observing a circular object(s). The processing
        converts the pseudo ray measurement field into features specific to circular obstructions.
        :param mover_name: the string of the name of the mover it is assigned too
        :param num_objects:
        :param max_range:
        """
        self.mover_name = mover_name
        self.num_objects = num_objects
        self.max_dist = max_dist

        # these are the state variables for the lidar sensor
        self.dist = None  # 0: distance to the center of the obstacle
        self.dist_starboard = None  # 1: distance of the tangent line to the starboard of the center line from the boats view point
        self.dist_port = None  # 2: distance of the tangent line to the port of the center line form the boats view point
        self.radius = None  # 3: radius of the obstacle
        self.theta = None  # 4: angle of the line to the center of the obstacle
        self.theta_starboard = None  # 5: angle of the line of the starboard tangent line
        self.theta_port = None  # 6: angle of the line of the port tangent line

        # reduced measurements for initial use case
        self.measurment = {'dist_center':None, 'angle_center':None}
        self.measurments = OrderedDict()
        for i in range(num_objects):
            tmp = {'dist_center_'+str(i):None, 'angle_center_'+str(i):None}
            self.measurments = {**self.measurments,**tmp}
        #self.measurment = namedtuple('Lidar_State', ('d_center', 'angle_center'))
        #self.measurments = []
        #self.measurments = namedtuple('Lidar_State', ('d_center','d_starboat_tangent','d_port_tangent','radius','angle_center','angle_starboard','angle_port'))

    def update_measurements(self, mover_dict):
        """
        given a boat the sensor is on, and the state of the obstacles, update the measurements
        :param mover_dict: an ordered dictionary of the movers in the simulation
        :return:
        """

        #if len(obstacles) != self.num_objects:
        #    raise RuntimeError(
        #        'The number of obstacles given to the lidar sensor does not match the expected amount of {}'.format(
        #            self.num_objects))

        # get the boat and its current position
        x, y = None, None
        for name, mover in mover_dict.items():
            if mover.name == self.mover_name:
                x = mover.x
                y = mover.y
                break

        # reset the measurements dictrionary
        self.measurments = OrderedDict()
        k = 0  # counter for index of state variables in
        for name, mover in mover_dict.items():
            if mover.name != self.mover_name:

                radius = mover.radius

                dx = mover.pos[0] - x
                dy = mover.pos[1] - y

                # distance between center of the entity and the mover the sensor is on
                dist = np.sqrt(dx * dx + dy * dy)

                if dist > self.max_dist:
                    self.dist = 0.0
                    self.dist_starboard = 0.0
                    self.dist_port = 0.0
                    self.radius = 0.0
                    self.theta = 0.0
                    self.theta_starboard = 0.0
                    self.theta_port = 0.0
                else:
                    # measurements can take place

                    if dist <= radius:
                        # correct for crashing into the obstacle
                        dist_starboard = 0.0
                        dist_port = 0.0
                    else:
                        dist_starboard = np.sqrt(dist * dist - radius * radius)
                        dist_port = np.sqrt(dist * dist - radius * radius)

                    theta = np.arctan2(dy, dx)
                    # if np.abs(radius/dist*np.sin(np.pi/2.0)) >= 1:
                    #    check = 0
                    if radius / dist >= 1.0:
                        d_theta = 0.0
                    else:
                        d_theta = np.arcsin(radius / dist * np.sin(np.pi / 2.0))
                    theta_starboard = theta - d_theta
                    theta_port = theta + d_theta

                    # correct theta
                    if theta < 0:
                        theta += np.pi * 2.0
                    elif theta > np.pi * 2.0:
                        theta -= np.pi * 2.0

                    # correct port and theta for crossing 0-360 [deg] azimuth
                    if theta_starboard < 0:
                        theta_starboard += np.pi * 2.0
                    elif theta_starboard > np.pi * 2.0:
                        theta_starboard -= np.pi * 2.0
                    if theta_port < 0:
                        theta_port += np.pi * 2.0
                    elif theta_port > np.pi * 2.0:
                        theta_port -= np.pi * 2.0

                    # store measurements that have been calculated
                    self.dist = dist
                    self.dist_starboard = dist_starboard
                    self.dist_port = dist_port
                    self.radius = radius
                    self.theta = theta
                    self.theta_starboard = theta_starboard
                    self.theta_port = theta_port

                #self.measurments = np.concatenate((self.measurments, list(self.get_measurment().values())), axis=0)
                self.measurments = {**self.measurments,**self.get_measurment(k)}
                k += 1

    def get_measurment(self, i):
        state = {'dist_center_'+str(i):self.dist, 'angle_center_'+str(i):self.theta}
        return state

    def get_state(self):
        """
        gets the state of the sensor and returns an array of all of the measurements
        :return:
        """
        # TODO resize the measurements to be a linear list
        return self.measurments

    def get_norm_state(self):
        """
        gets the normalized state of the sensor and
        :return:
        """
        norm_state = copy.deepcopy(self.measurments)
        for key,value in norm_state.items():
            if 'dist' in key:
                norm_state[key] /= self.max_dist
            elif 'angle' in key:
                norm_state[key] = value / (2.0*np.pi)
        return norm_state