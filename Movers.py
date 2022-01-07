
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
import os

import numpy as np
import pandas as pd


class Mover(ABC):

    def __init__(self,name):

        self.name = name
        self.sensors = []
        self.history = pd.DataFrame()
        self.can_learn = False

    @abstractmethod
    def step(self):
        raise NotImplementedError('The mover '+self.name+' needs to implement the step method')

    def add_sensor(self,sensor):
        """
        add a sensor object to the mover. All sensors are stored in a list and are updated in order that they are added
        to the mover
        :param sensor: the sensor object that samples the enviornment to update its measurements
        :return:
        """
        self.sensors.append(sensor)

    def remove_sensor(self,sensor_name):
        """
        removes a sensor from the mover's sensors based on the target sensors name
        :param sensor_name:
        :return:
        """
        for i, sensor in enumerate(self.sensors):
            if sensor.name == sensor_name:
                self.sensors.pop(i)
                return True
        return False

    def update_sensors(self, mover_dict):
        """
        loops over each sensor in the mover and updates the measurements for each sensor
        :return:
        """
        for sensor in self.sensors:
            sensor.update_measurements(mover_dict)

    def get_history(self):
        """

        :return:
        """
        return self.history

    @abstractmethod
    def reset_history(self, num_steps):
        raise NotImplementedError('The mover '+self.name+' must implement the reset_history method')

    @abstractmethod
    def add_step_history(self, step_num):
        raise NotImplementedError('The mover '+self.name+' must implement the add_step_history method')

    @abstractmethod
    def normalize_state(self):
        raise NotImplementedError('The mover '+self.name+' must implement the normalize_state method')

    @abstractmethod
    def action_to_command(self):
        """
        takes an action from agent and converts it into the command to control the mover
        :return:
        """
        raise NotImplementedError('The mover ' + self.name + ' must implement the action_to_command method')


class StaticCircle(Mover,ABC):

    def __init__(self, name, max_radius):
        # instantiate mover
        super().__init__(name)

        self.pos = [0.0, 0.0]  # x, y position in meters of the obstacle
        self.norm_pos = [0.0, 0.0] # the normalized position values to be between 0 and 1
        self.radius = 1.0  # radius in meters of the obstacle
        self.norm_radius = 1.0 # the normalized radous value to be between 0 and 1
        self.max_radius = max_radius
        self.domain = None  # the domain the mover can exists in

    def set_domian(self, domain):
        """
        sets the possible domain the center of the circle can be in
        :param domain:
        :return:
        """

        # domain has information of:
        #     | x_min, x_max |
        #     | y_min, y_max |
        if domain.shape() != (2,2):
            raise ValueError('The domain must have a shape of (2,2)')
        self.domain = domain

    def update_state(self,pos, radius):
        """
        updates the position and radius of the circle.
        :param pos:
        :param radius:
        :return:
        """

        if pos[0] < self.domain[0,0] or pos[0] > self.domain[0,1]:
            raise ValueError('X dimension for mover {} is not in acceptable domain. Either adjust the domain or provide'
                             'a different x location')
        if pos[1] < self.domain[1,0] or pos[1] > self.domain[1,1]:
            raise ValueError('X dimension for mover {} is not in acceptable domain. Either adjust the domain or provide'
                             'a different x location')
        self.pos = pos
        if radius > self.max_radius:
            raise ValueError('cannot make the radius of mover {} larger than {} [m]'.format(self.name,self.max_radius)  )
        else:
            self.radius = radius

    def step(self):
        # no changes are made as this is a static circle
        pass

    def normalize_state(self):
        """
        normalizes the postion and radius of the mover to be between 0 and 1
        :return:
        """
        self.norm_pos[0] = (self.pos[0]-self.domain[0,0])/(self.domain[0,1]-self.domain[0,0])
        self.norm_pos[1] = (self.pos[1] - self.domain[1, 0]) / (self.domain[1, 1] - self.domain[1, 0])
        self.norm_radius = self.radius/self.max_radius

    def action_to_command(self):
        # there are no actions for the static circle as it does not make any decisions
        return 0

    def add_step_history(self, step_num):
        self.history.iloc[step_num] = [self.pos[0], self.pos[1], self.radius]

    def reset_history(self, num_steps):
        """
        sets all of the history to zero
        :param num_steps:
        :return:
        """
        empty_data = np.zeros((num_steps, 3))
        self.history = pd.DataFrame(data=empty_data, columns=['x_'+self.name,'y_'+self.name,'radius_'+self.name])

    def trim_history(self, step_num):
        self.history.drop(range(step_num, len(self.history)), inplace=True)


class UtilityBoat(Mover, ABC):
    """
    This is a model for a boat similiar to a fishing boat or a tug boat. It is called utility for a general name.
    """

    def __init__(self, name, delta, delta_t, fom, hull_area, hull_len, mass, phi, power, prop_diam, fuel_capacity, fuel,
                 bsfc, mode):

        # instantiate mover
        super().__init__(name)

        self.alpha = 0  # angle of incidence of the propper in radians
        self.delta = delta  # propeller angle relative to the hull
        self.delta_t = delta_t  # time step size
        self.x = 0  # x position of the boat in meters
        self.y = 0  # y position of the boat in meters
        self.fom = fom  # figure of merit for the propeller. Should be about 0.7-0.75
        self.hull_area = hull_area  # the frontal area of the hull
        self.hull_drag = 0  # the drag produced by the hull
        self.hull_len = hull_len  # the total longitudinal length of the boat hull
        self.mass = mass  # the total mass of the entire boat
        self.moi = 0  # moment of interia of the boat approximated as cylinder
        self.set_moi()  # calculate the MOI for the boat
        self.phi = phi  # angle of the hull of the boat in the global coordinates
        self.max_power = power  # the maximum power in watts the propeller has
        self.thrust = 0  # the current thrust the propeller can deliver
        self.t_x_p = 0  # the component of the thrust in the longitudinal axis of the boat
        self.t_y_p = 0  # the component of the thrust in the lateral axis of the boat
        self.power = 0  # the current amount of power the propeller has
        self.prop_diam = prop_diam  # the outer diameter of the propeller disk
        self.prop_area = np.power((prop_diam/2.0),2.0)*np.pi  # the disk area of the propeller. Assumes no hole in the propeller disk
        self.v_boat_mag = 0  # the magnitude of the boats velocity
        self.v_boat_x = 0  # the velocity of the boat in the global x direction
        self.v_boat_y = 0  # the velocity of the boat in the global y direction
        self.v_boat_theta = 0 # velocity of the boat in the theat direction
        self.v_boat_x_p = 0  # the velocity of the boat in the boats x direction (longitudinal direction)
        self.v_boat_y_p = 0  # the velocity of the boat in the boats y direction (lateral direction)
        self.a_x = 0  # acceleration of the boat in the global x direction
        self.a_y = 0  # acceleration of the boat in the global y direction
        self.a_theta = 0 # acceleration of the boat in the theta direction
        self.a_x_p = 0  # acceleration of the boat in the boats x direction (longitudinal direction)
        self.a_y_p = 0  # acceleration of the boat in the boats y direction (lateral direction)
        self.theta = 0  # angle from the boat to the destination
        self.a_boat_theta = 0  # acceleration of the boat in the vector that points from the boat to the destination
        self.mu = 0  # heading angle difference between boat heading and vector to the destination
        self.destination = [0,0]  # the x,y pair for the location of the destination in meters
        self.destination_distance = 0  # the distance in meters to the destination

        # fuel used. Values approximated from here https://en.wikipedia.org/wiki/Brake-specific_fuel_consumption .
        # note values are for engines that are used for container ships which may be more efficient then small boats.
        # the range of BSFC is [4.16 e-8, 5.555 e-8] [kg/(watt second)]. BSFC = brake specific fuel consumption
        self.fuel_capacity = fuel_capacity
        if fuel > fuel_capacity:
            raise ValueError('fuel must be less than or equal to the fuel capacity')
        self.fuel = fuel
        self.max_fuel = fuel # currently assumes that fuel is loaded to the max
        self.bsfc = bsfc

        # sets if the boat is controlled via the propeller only or if that and the power is controlled
        self.mode = mode
        # for propeller only
        self.state_prop = OrderedDict([('alpha',None), ('delta',None), ('destination_distance',None), ('mu', None),
                                       ('phi',None), ('theta',None), ('v_boat_theta',None), ('v_boat_x_p',None),
                                      ('v_boat_x',None), ('v_boat_y',None)])
        # for controlling both the propeller and the power
        self.state_full = OrderedDict([('alpha',None), ('delta',None), ('destination_distance',None), ('fuel',None),('mu', None),
                                       ('phi',None), ('power',None), ('theta',None), ('v_boat_theta',None), ('v_boat_x_p',None)])
        self.can_learn = True

    def calc_dest_metrics(self):
        """
        calculates the euclidean distance from the boat to the destination. The theta angle is the angle from the boat
        to the destination relative to the positive x axis. Theta is measured CCW from the positive x axis. Mu is the
        angle between the boats heading and the theta heading. This is the angle the boat needs to turn to point at
        the destination.
        :return:
        """

        self.destination_distance = np.sqrt((self.x - self.destination[0]) * (self.x - self.destination[0]) + (self.y - self.destination[1]) * (
                self.y - self.destination[1]))
        delta_x = self.destination[0] - self.x
        delta_y = self.destination[1] - self.y

        self.theta = np.arctan2(delta_y, delta_x)

        mu1 = self.theta - self.phi
        if mu1 >= 0:
            mu2 = np.pi*2.0 - mu1# explementary angle
        else:
            mu2 = np.pi*2.0 + mu1# explementary angle
        mu_v = [mu1,mu2]
        ind = np.argmin(np.abs(mu_v))
        self.mu = mu_v[ind]

    def bound_hull_angle(self):
        """
        bounds the hull angle to be between 0 and 2pi
        :return:
        """
        # bound the hull angle phi to be between 0 and 2 pi
        if self.phi < 0:
            self.phi += np.pi * 2.0
        elif self.phi > np.pi * 2.0:
            self.phi -= np.pi * 2.0

    def step(self):
        """
        makes one step of the boat in a simulation. Calculates the new position, angle, and velocities based on where
        the control settings applied
        :return:
        """

        # bound the hull angle phi to be between 0 and 2 pi
        #self.bound_hull_angle()

        # calculate angle of incidnce of propeller disk
        self.alpha = self.delta + np.pi / 2.0

        # convert boat velocities to boat frame
        #self.v_boat_x_p = self.v_boat_x * np.cos(self.phi) + self.v_boat_y * np.sin(self.phi)
        #self.v_boat_y_p = 0
        #self.v_boat_mag = np.sqrt(self.v_boat_x * self.v_boat_x + self.v_boat_y * self.v_boat_y)

        # calculate distance to destination
        #self.calc_dest_metrics()

        # calculate velocity and acceleration in theta direction
        #self.v_boat_theta = self.v_boat_x * np.cos(-self.theta) - self.v_boat_y * np.sin(-self.theta)
        #self.a_boat_theta = self.a_x * np.cos(-self.theta) - self.a_y * np.sin(-self.theta)

        adj = -1.0
        if self.v_boat_x_p < 0:
            adj = 1.0

        # calculate angle of incidnce of propeller disk
        self.alpha = self.delta + np.pi / 2.0

        self.calc_thrust()

        t_x_p = self.thrust * np.cos(self.delta)
        t_y_p = self.thrust * np.sin(self.delta)

        # calculate hull drag
        self.calc_hull_drag()

        f_x_p = t_x_p + adj * self.hull_drag

        if abs(f_x_p) > 1e6:
            f_x_p = 0.0

        delta_x_p = self.v_boat_x_p * self.delta_t + 0.5 * f_x_p / self.mass * self.delta_t * self.delta_t
        delta_y_p = 0

        # update rotation of boat
        delta_phi = (t_y_p) * (self.hull_len / 2.0) / self.moi * self.delta_t * self.delta_t
        self.phi = self.phi - delta_phi

        # bound the hull angle phi to be between 0 and 2 pi
        self.bound_hull_angle()

        # convert change in position to global frame
        delta_x = delta_x_p * np.cos(-self.phi) + delta_y_p * np.sin(-self.phi)
        delta_y = -delta_x_p * np.sin(-self.phi) + delta_y_p * np.cos(-self.phi)

        self.x = self.x + delta_x
        self.y = self.y + delta_y

        #self.calc_dest_metrics()

        v_boat_x_p_new = self.v_boat_x_p + f_x_p / self.mass * self.delta_t
        v_boat_y_p_new = 0
        self.v_boat_mag = np.abs(v_boat_x_p_new)  # update velocity of the boat

        # conver velocities to global frame
        self.v_boat_x = v_boat_x_p_new * np.cos(-self.phi) + v_boat_y_p_new * np.sin(-self.phi)
        self.v_boat_y = -v_boat_x_p_new * np.sin(-self.phi) + v_boat_y_p_new * np.cos(-self.phi)

        # a_x_p = (v_boat_x_p_new - self.v_boat_x_p) / self.delta_t
        a_x_p = f_x_p / self.mass
        a_y_p = 0

        # convert acceleration to global reference plane
        self.a_x = a_x_p * np.cos(-self.phi) + a_y_p * np.sin(-self.phi)
        self.a_y = -a_x_p * np.sin(-self.phi) + a_y_p * np.cos(-self.phi)

        # calculate the fuel used in the simulation
        fuel_used = self.power * self.bsfc * self.delta_t  # [kg of fuel]
        self.fuel -= fuel_used

        if self.fuel <= 0:
            self.fuel = 0.0

        # calculate angle of incidnce of propeller disk
        #self.alpha = self.delta + np.pi / 2.0

        # convert boat velocities to boat frame
        self.v_boat_x_p = self.v_boat_x * np.cos(self.phi) + self.v_boat_y * np.sin(self.phi)
        self.v_boat_y_p = 0
        self.v_boat_mag = np.sqrt(self.v_boat_x * self.v_boat_x + self.v_boat_y * self.v_boat_y)

        # calculate distance to destination
        #self.calc_dest_metrics()

        # calculate velocity and acceleration in theta direction
        self.v_boat_theta = self.v_boat_x * np.cos(-self.theta) - self.v_boat_y * np.sin(-self.theta)
        self.a_boat_theta = self.a_x * np.cos(-self.theta) - self.a_y * np.sin(-self.theta)

    def set_control(self,power,delta):
        """
        sets the power of the propeller and the angle of the propeller. Both are set in absolute terms
        :param power: The power setting in watts of the propeller
        :param delta: The angle in radians of the propeller angle relative to the boats centerline
        :return:
        """

        # limit maimum power
        if self.power > self.max_power:
            self.power = self.max_power
        elif power < 0:
            self.power = 0
        else:
            self.power = power

        # engine cannot produce power if there is no fuel
        if self.fuel <= 0.0:
            self.power = 0

        # check bounds of propeller angle
        self.delta = delta

        '''
        # make delta between 0 and 360
        if self.delta < 0:
            self.delta += np.pi * 2.0
        elif self.delta >= 2.0 * np.pi:
            self.delta -= np.pi * 2.0
        '''
        # make delta between -90 and 90 to prevent propeller going all the way around
        if self.delta < -np.pi/2.0:
            self.delta = -np.pi/2.0
        elif self.delta > np.pi/2.0:
            self.delta = np.pi/2.0


    def set_moi(self):
        """
        approximates the moment of intertia for the boat as the MOI for a cylinder.
        :return:
        """
        r = np.sqrt(self.hull_area/np.pi)
        self.moi = 0.25*self.mass*r*r + 1.0/12.0*self.mass*self.hull_len*self.hull_len

    def calc_hull_drag(self):
        """
        gets the drag produced on the hull by the boat moving through the water
        :return: drag of the boats hull
        """
        # calculate reynolds number of the boat
        if self.v_boat_mag == 0:
            # handle divide by zero case
            re = 1
        else:
            re = 997*np.abs(self.v_boat_mag)*self.hull_len/(8.90*10e-4)

        # skin friction coefficient
        cf = 0.025/np.power((np.log10(re)-2),2.0)

        # calculate the froude number
        fr = np.abs(self.v_boat_mag)/np.sqrt(9.807*self.hull_len)

        # get the profile drag
        cw = 0.0022*np.exp((fr-0.33)/0.057)

        # return the dimensional drag of the hull
        self.hull_drag = 0.5*997*np.abs(self.v_boat_mag)*np.abs(self.v_boat_mag)*self.hull_area*cf+0.5*997*np.abs(self.v_boat_mag)*np.abs(self.v_boat_mag)*self.hull_area*cw

        if self.v_boat_x_p < 0:
            self.hull_drag *= 4.0

    def thrust_eq(self,v):

        thrust = 2 * 998 * self.prop_area*np.sqrt(np.power(v, 4.0) + 2.0 * self.v_boat_x_p * np.sin(self.alpha) * np.power(v,3.0) + self.v_boat_x_p * self.v_boat_x_p * v * v )

        if np.isnan(thrust):
            thrust = 0.0

        return thrust

    def calc_thrust(self):
        """
        calculates the thrust delived to the propeller based on the controlled power
        :param alpha: angle in radians of the veloicty of the water to the propeller disk
        :param power: power given to the propeller
        :param v_0: velocity of the water passing over the propeller disk
        :return:
        """

        x0 = 0
        x1 = 5

        x2 = 0
        fx0 = self.fom*(self.v_boat_x_p*np.sin(self.alpha)+x0)*self.thrust_eq(x0) - self.power
        fx1 = self.fom*(self.v_boat_x_p*np.sin(self.alpha)+x1)*self.thrust_eq(x1) - self.power

        err = 1
        c = 0
        while err > 1e-1 and c < 50:
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

            fx2 = self.fom*(self.v_boat_x_p*np.sin(self.alpha)+x2)*self.thrust_eq(x2) - self.power
            err = np.abs(fx2)

            x0 = x1
            fx0 = fx1
            x1 = x2
            fx1 = fx2
            c += 1

        v_solve = x2
        self.thrust = self.thrust_eq(v_solve)

    @staticmethod
    def get_default(name,delta_t,mode):
        """
        a basic boat for use in training. Models roughly a 24 foot fishing boat
        :param delta_t:
        :return:
        """
        delta = 0
        fom = 0.75  # figure of merit
        hull_area = 3
        hull_len = 7
        mass = 1500
        phi = 0
        power = 20000  # [watt]
        prop_diam = 0.25  # [m]
        fuel_capacity = 2  # [kg]
        fuel = 1.5  # [kg]
        bsfc = 5.0e-8  # [kg/w-s] this is the realistic value
        #bsfc = 5.0e-7  # [kg/w-s] this is the inefficient value for use
        ub = UtilityBoat(name,delta, delta_t, fom, hull_area, hull_len, mass, phi, power, prop_diam, fuel_capacity, fuel,
                         bsfc, mode)

        return ub

    def action_to_command(self, action):
        """
        convert an action integer into what commands the vehicle will take
        :param action: integer for the action
        :return:
        """
        if self.mode == 'propeller_only':
            power_adj = 0

            delta_adj = 0
            if action == 0:
                # move propeller clockwise by 15 degrees
                delta_adj = np.deg2rad(-15)
            elif action == 1:
                # move propeller clockwise by 5 degrees
                delta_adj = np.deg2rad(-5)
            elif action == 2:
                # move propeller clockwise by 1 degrees
                delta_adj = np.deg2rad(-1)
            elif action == 4:
                # move propeller ccw by 1 degrees
                delta_adj = np.deg2rad(1)
            elif action == 5:
                # move propeller ccw by 5 degrees
                delta_adj = np.deg2rad(5)
            elif action == 6:
                # move propeller ccw by 15 degrees
                delta_adj = np.deg2rad(15)

        elif self.mode == 'propeller_and_power':
            # the boat controls both the propller angle and the power
            prop_action = int(np.floor(action / 5))
            power_action = int(action % 5)

            power_adj = 0
            if power_action == 0:
                # decrease power by 500 watts
                power_adj = -500
            elif power_action == 1:
                # decrease power by 100 watts
                power_adj = -100
            elif power_action == 3:
                # increase power by 100 watts
                power_adj = 100
            elif power_action == 4:
                # increase power by 500 watts
                power_adj = 500

            delta_adj = 0
            if prop_action == 0:
                # move propeller clockwise by 5 degres
                delta_adj = np.deg2rad(-5)
            elif prop_action == 1:
                # move propeller clockwise by 1 degres
                delta_adj = np.deg2rad(-1)
            elif prop_action == 3:
                # move propeller ccw by 1 degres
                delta_adj = np.deg2rad(1)
            elif prop_action == 4:
                # move propeller ccw by 5 degres
                delta_adj = np.deg2rad(5)

        else:
            raise ValueError('Utility boat {} has not been given a valid control mode'.format(self.name))

        return delta_adj, power_adj  # adjustment to the propeller angle and power respectivly

    def normalize_state(self, include_sensors):
        """
        normalizes the current state of the boat so that all values are between 0 and 1
        :return: a named tuple containing the normalized state
        """
        norm_state = self.get_state(False)

        # normalize the state vector to
        if self.mode == 'propeller_only':
            norm_state['alpha'] = norm_state['alpha'] / (np.pi * 2.0)
            #norm_state['delta'] = norm_state['delta'] / (np.pi * 2.0)
            norm_state['delta'] = (norm_state['delta']+np.pi/2.0) / (np.pi )
            norm_state['destination_distance'] = norm_state['destination_distance'] / 300.0
            norm_state['mu'] = (norm_state['mu'] + np.pi) / (np.pi * 2.0)
            norm_state['phi'] = norm_state['phi'] / (np.pi / 2.0)
            norm_state['theta'] = (norm_state['theta'] + np.pi) / (np.pi * 2.0)
            norm_state['v_boat_theta'] = norm_state['v_boat_theta'] / 5.0
            norm_state['v_boat_x_p'] = (norm_state['v_boat_x_p'] + 5.0) / 10.0
            norm_state['v_boat_x'] = (norm_state['v_boat_x']+5.0)/10.0
            norm_state['v_boat_y'] = (norm_state['v_boat_y'] + 5.0) / 10.0

        else:
            norm_state['alpha'] = norm_state['alpha'] / (np.pi * 2.0)
            norm_state['delta'] = norm_state['delta'] / (np.pi * 2.0)
            norm_state['destination_distance'] = norm_state['destination_distance'] / 300.0
            norm_state['fuel'] = norm_state['fuel'] / self.max_fuel
            norm_state['mu'] = (norm_state['mu'] + np.pi) / (np.pi * 2.0)
            norm_state['phi'] = norm_state['phi'] / (np.pi / 2.0)
            norm_state['power'] = norm_state['power'] / self.max_power
            norm_state['theta'] = (norm_state['theta'] + np.pi) / (np.pi * 2.0)
            norm_state['v_boat_theta'] = norm_state['v_boat_theta'] / 5.0
            norm_state['v_boat_x_p'] = (norm_state['v_boat_x_p'] + 5.0) / 10.0

        if include_sensors:
            for sensor in self.sensors:
                tmp_state = sensor.get_norm_state()
                norm_state = {**norm_state, **tmp_state}

        return norm_state

    def get_state(self, include_sensors):
        """
        get the state of the boat
        :return:
        """

        if self.mode == 'propeller_only':
            #self.state_prop = OrderedDict( [('alpha', None), ('delta', None), ('destination_distance', None), ('mu', None),('phi', None), ('theta', None), ('v_boat_theta', None), ('v_boat_x_p', None)])
            self.state_prop['alpha'] = self.alpha
            self.state_prop['delta'] = self.delta
            self.state_prop['destination_distance'] = self.destination_distance
            self.state_prop['mu'] = self.mu
            self.state_prop['phi'] = self.phi
            self.state_prop['theta'] = self.theta
            self.state_prop['v_boat_theta'] = self.v_boat_theta
            self.state_prop['v_boat_x_p'] = self.v_boat_x_p
            self.state_prop['v_boat_x'] = self.v_boat_x
            self.state_prop['v_boat_y'] = self.v_boat_y

            if include_sensors:
                # get the sensors information and add it to the state
                for sensor in self.sensors:
                    tmp_state = sensor.get_state()
                    self.state_prop = {**self.state_prop, **tmp_state}

            return self.state_prop
        else:

            self.state_full['alpha'] = self.alpha
            self.state_full['delta'] = self.delta
            self.state_full['destination_distance'] = self.destination_distance
            self.state_full['fuel'] = self.fuel
            self.state_full['mu'] = self.mu
            self.state_full['phi'] = self.phi
            self.state_full['power'] = self.power
            self.state_full['theta'] = self.theta
            self.state_full['v_boat_theta'] = self.v_boat_theta
            self.state_full['v_boat_x_p'] = self.v_boat_x_p

            return self.state_full

    def get_action_size(self, action_type):
        """
        gets the actions space size based on the type of controller (discrete or continous)
        :param action_type:
        :return:
        """
        size = None

        if self.mode == 'propeller_only':
            # use only the propeller

            if action_type == 'discrete':
                size = 7
            elif action_type == 'continous':
                size = 1
            else:
                raise ValueError('Only discrete and continous action spaces are allowed')

        else:
            # control power and propeller

            if action_type == 'discrete':
                size = 25
            elif action_type == 'continous':
                size = 2
            else:
                raise ValueError('Only discrete and continous action spaces are allowed')
        return size

    def add_step_history(self, step_num):
        telemetry = {'acc_theta': self.a_theta, 'acc_boat_theta': self.a_boat_theta, 'acc_x': self.a_x,
                     'acc_x_p': self.a_x_p, 'acc_y': self.a_y, 'acc_y_p': self.a_y_p,
                     'alpha': self.alpha, 'bsfc': self.bsfc, 'delta': self.delta,
                     'destination_x':self.destination[0],'destination_y':self.destination[1],
                     'destination_distance': self.destination_distance,
                     'fuel': self.fuel, 'mu':self.mu, 'phi': self.phi, 'power': self.power,
                     'theta': self.theta,
                     'thrust': self.thrust, 'thrust_x_p': self.t_x_p, 'thrust_y_p': self.t_y_p,
                     'v_boat_mag': self.v_boat_mag, 'v_boat_theta': self.v_boat_theta, 'v_boat_x': self.v_boat_x,
                     'v_boat_x_p': self.v_boat_x_p, 'v_boat_y': self.v_boat_y,
                     'x_pos': self.x, 'y_pos': self.y}
        # get the sensors information and add it to the state
        for sensor in self.sensors:
            tmp_state = sensor.get_state()
            telemetry = {**telemetry, **tmp_state}
        self.history.iloc[step_num] = telemetry.values()

    def reset_history(self,num_steps):
        """
        sets all of the history to zero
        :param num_steps:
        :return:
        """
        telemetry = {'acc_theta':self.a_theta,'acc_boat_theta':self.a_boat_theta,'acc_x':self.a_x,'acc_x_p':self.a_x_p,'acc_y':self.a_y,'acc_y_p':self.a_y_p,
                     'alpha':self.alpha,'bsfc':self.bsfc,'delta':self.delta,
                     'destination_x':self.destination[0],'destination_y':self.destination[1],'destination_distance':self.destination_distance,
                     'fuel':self.fuel, 'mu':self.mu,'phi':self.phi,'power':self.power,
                     'theta':self.theta,
                     'thrust':self.thrust,'thrust_x_p':self.t_x_p,'thrust_y_p':self.t_y_p,
                     'v_boat_mag':self.v_boat_mag,'v_boat_theta':self.v_boat_theta,'v_boat_x':self.v_boat_x,'v_boat_x_p':self.v_boat_x_p,'v_boat_y':self.v_boat_y,
                     'x_pos':self.x,'y_pos':self.y}
        # get the sensors information and add it to the state
        for sensor in self.sensors:
            tmp_state = sensor.get_state()
            telemetry = {**telemetry, **tmp_state}

        empty_data = np.zeros((num_steps,len(telemetry.keys())))
        self.history = pd.DataFrame(data=empty_data,columns=telemetry.keys())

    def trim_history(self, step_num):
        """
        remove the rows of data that are not used due to early termination of the simulation
        :param step_num:
        :return:
        """
        self.history.drop(range(step_num, len(self.history)), inplace=True)


class SailBoat:

    def __init__(self, delta_t, mass, area_sail, area_hull, len_hull, area_rudder):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._naca0012 = pd.read_csv(dir_path + '\\Naca0012.csv')
        self.delta_t = delta_t  # time step. 0.5 [s] reccomended
        self.mass = mass
        self.area_sail = area_sail
        self.area_hull = area_hull
        self.len_hull = len_hull
        self.area_rudder = area_rudder
        self.moi = 0
        self.set_moi()

        self.gamma = 0  # global wind angle
        self.gamma_prime = 0  # relative wind angle when accounting for boat velocity
        self.beta = 0  # angle of the sail relative to the boat
        self.phi = 0  # angle of the boat relative to the global frame
        self.delta = 0  # angle of the rudder relative to the boat
        self.alpha = 0  # initial angle of attack of the sail/wing
        self.alpha_prime = 0  # relative angle of attack of the sail/wing
        self.a_x = 0  # acceleration in global x direction
        self.a_y = 0  # acceleration in global y direction
        self.v_boat_x = 0  # velocity in the global x direction
        self.v_boat_y = 0  # velocity in the global y direction
        self.v_boat_x_p = 0  # velocity in the local x direction
        self.v_boat_y_p = 0  # velocity in the local y direction
        self.v_boat = 0  # magnitude of the boat's velocity
        self.x = 0  # x potion in global frame
        self.y = 0  # y position in global frame

        self.v_wind = 0  # magnitude of the wind
        self.v_wind_x = 0  # velocity of the wind in the x direction of the global frame
        self.v_wind_y = 0  # velocity of the wind in the y direction of the global frame
        self.v_tot_x = 0  # total velocity of the relative wind in the global x frame
        self.v_tot_y = 0  # total velocity of the relative wind in the global y frame

        self.f_x = 0  # force in the global x direction
        self.f_y = 0  # force in the global y direction
        self.l_x = 0  # sail lift force in the global x direction
        self.l_y = 0  # sail lift force in the global y direction
        self.d_x = 0  # sail drag force in the global x direction
        self.d_y = 0  # sail drag force in the global y direction
        self.l_rudder_x = 0  # rudder lift force in the global x direction
        self.l_rudder_y = 0  # rudder lift force in the global y direction
        self.d_rudder_x = 0  # rudder drag force in the global x direction
        self.d_rudder_y = 0  # rudder drag force in the global y direction
        self.hull_drag = 0  # drag of the hull

        self.theta = 0  # angle from the boat to the destination
        self.v_boat_theta = 0
        self.a_boat_theta = 0
        self.mu = 0  # heading angle difference between boat heading and vector to the destination
        self.destination = [0, 0]
        self.destination_distance = 0

    def calc_dest_metrics(self):
        """
        calculates the euclidean distance from the boat to the destination. The theta angle is the angle from the boat
        to the destination relative to the positive x axis. Theta is measured CCW from the positive x axis. Mu is the
        angle between the boats heading and the theta heading. This is the angle the boat needs to turn to point at
        the destination.
        :return:
        """

        self.destination_distance = np.sqrt(
            (self.x - self.destination[0]) * (self.x - self.destination[0]) + (self.y - self.destination[1]) * (
                    self.y - self.destination[1]))
        delta_x = self.destination[0] - self.x
        delta_y = self.destination[1] - self.y

        self.theta = np.arctan2(delta_y, delta_x)

        mu1 = self.theta - self.phi
        if mu1 >= 0:
            mu2 = np.pi * 2.0 - mu1  # explementary angle
        else:
            mu2 = np.pi * 2.0 + mu1  # explementary angle
        mu_v = [mu1, mu2]
        ind = np.argmin(np.abs(mu_v))
        self.mu = mu_v[ind]

    def step(self):
        """
        calculates a step of the sailboat moving forward in time
        :return:
        """

        self.check_angles()

        self.calc_velocity()
        self.calc_alpha()

        self.calc_wing_forces()
        self.calc_rudder_forces()

        self.calc_hull_drag()

        # convert forces to boat reference frame
        l_x_p = self.l_x * np.cos(self.phi) + self.l_y * np.sin(self.phi)
        l_y_p = -self.l_x * np.sin(self.phi) + self.l_y * np.cos(self.phi)
        l_rudder_x_p = self.l_rudder_x * np.cos(self.phi) + self.l_rudder_y * np.sin(self.phi)
        l_rudder_y_p = -self.l_rudder_x * np.sin(self.phi) + self.l_rudder_y * np.cos(self.phi)

        d_x_p = self.d_x * np.cos(self.phi) + self.d_y * np.sin(self.phi)
        d_y_p = -self.d_x * np.sin(self.phi) + self.d_y * np.cos(self.phi)
        d_rudder_x_p = self.d_rudder_x * np.cos(self.phi) + self.d_rudder_y * np.sin(self.phi)
        d_rudder_y_p = -self.d_rudder_x * np.sin(self.phi) + self.d_rudder_y * np.cos(self.phi)

        # convert boat velocities to boat frame
        self.v_boat_x_p = self.v_boat_x * np.cos(self.phi) + self.v_boat_y * np.sin(self.phi)
        self.v_boat_y_p = 0
        adj = -1.0
        if self.v_boat_x_p < 0:
            adj = 1.0

        f_x_p = l_x_p + l_rudder_x_p + d_x_p + d_rudder_x_p + adj * self.hull_drag

        delta_x_p = self.v_boat_x_p * self.delta_t + 0.5 * f_x_p / self.mass * self.delta_t * self.delta_t
        delta_y_p = 0

        # convert change in position to global frame
        delta_x = delta_x_p * np.cos(-self.phi) + delta_y_p * np.sin(-self.phi)
        delta_y = -delta_x_p * np.sin(-self.phi) + delta_y_p * np.cos(-self.phi)

        # update position
        self.x = self.x + delta_x
        self.y = self.y + delta_y

        # update rotation of boat
        delta_phi = (l_rudder_y_p + d_rudder_y_p) * (self.len_hull / 2.0) / self.moi * self.delta_t * self.delta_t
        self.phi = self.phi - delta_phi

        self.calc_dest_metrics()

        v_boat_x_p_new = self.v_boat_x_p + f_x_p / self.mass * self.delta_t
        v_boat_y_p_new = 0
        self.v_boat = np.abs(v_boat_x_p_new)  # update velocity of the boat

        # conver velocities to global frame
        self.v_boat_x = v_boat_x_p_new * np.cos(-self.phi) + v_boat_y_p_new * np.sin(-self.phi)
        self.v_boat_y = -v_boat_x_p_new * np.sin(-self.phi) + v_boat_y_p_new * np.cos(-self.phi)

        a_x_p = (v_boat_x_p_new - self.v_boat_x_p) / self.delta_t
        a_y_p = 0

        # convert acceleration to global reference plane
        self.a_x = a_x_p * np.cos(-self.phi) + a_y_p * np.sin(-self.phi)
        self.a_y = -a_x_p * np.sin(-self.phi) + a_y_p * np.cos(-self.phi)

        # calculate velocity and acceleration in theta direction
        self.v_boat_theta = self.v_boat_x * np.cos(-self.theta) - self.v_boat_y * np.sin(-self.theta)
        self.a_boat_theta = self.a_x * np.cos(-self.theta) - self.a_y * np.sin(-self.theta)

        # update the angles and velocities for the simulation to sample correctly for state prime
        self.check_angles()
        self.calc_velocity()
        self.calc_alpha()
        # convert boat velocities to boat frame
        self.v_boat_x_p = self.v_boat_x * np.cos(self.phi) + self.v_boat_y * np.sin(self.phi)
        self.v_boat_y_p = 0

    def check_angles(self):
        """
        Checks the angles of the rudder and sail and bounds them if needed
        :return:
        """

        if self.beta > np.pi * 2.0:
            self.beta -= np.pi * 2.0
        elif self.beta < 0:
            self.beta += np.pi * 2.0

        if self.gamma > np.pi * 2.0:
            self.gamma -= np.pi * 2.0
        elif self.gamma < 0:
            self.gamma += np.pi * 2.0

        if self.phi > np.pi * 2.0:
            self.phi -= np.pi * 2.0
        elif self.phi < 0:
            self.phi += np.pi * 2.0

        if self.delta > np.pi * 2.0:
            self.delta -= np.pi * 2.0
        elif self.delta < 0:
            self.delta += np.pi * 2.0

    def calc_velocity(self):
        """
        calculates the velocities relevant to the boat. Wind and boat veloicties are found
        :return:
        """
        # self.v_boat_mag = np.sqrt(self.v_x*self.v_x + self.v_y*self.v_y)
        # convert magnitude and direction of boat int
        # self.v_boat_x = self.v_boat*np.cos(self.phi)
        # self.v_boat_y = self.v_boat * np.sin(self.phi)
        self.v_boat = np.sqrt(self.v_boat_x * self.v_boat_x + self.v_boat_y * self.v_boat_y)

        self.v_wind_x = -np.cos(self.gamma) * self.v_wind
        self.v_wind_y = -np.sin(self.gamma) * self.v_wind

        self.v_tot_x = self.v_boat_x - self.v_wind_x
        self.v_tot_y = self.v_boat_y - self.v_wind_y

        # self.v_wind_mag = np.sqrt(self.v_tot_x * self.v_tot_x + self.v_tot_y * self.v_tot_y)

    def calc_alpha(self):
        """
        calculate initial and relative angle of attack
        :return:
        """
        self.alpha = self.gamma - self.beta - self.phi

        if abs(self.v_tot_x) <= 1e-12:
            if self.v_tot_y < 0:
                self.gamma_prime = np.pi / 2.0
            else:
                self.gamma_prime = 3.0 / 2.0 * np.pi
        else:
            self.gamma_prime = np.arctan(self.v_tot_y / self.v_tot_x)

        if self.gamma_prime < 0:
            self.gamma_prime += np.pi

        self.alpha_prime = self.gamma_prime - self.beta - self.phi

    def calc_wing_forces(self):
        """
        calcualtes the foces of lift and drag for the sail and converts them into the global frame
        :return:
        """
        # lift
        alpha_eff = self.alpha_prime
        while alpha_eff < 0:
            alpha_eff += 2.0 * np.pi
        cl = np.interp(np.rad2deg(alpha_eff), self._naca0012['alpha'], self._naca0012['cl'])
        # convert nondimensional lift to dimensional lift
        lift = 0.5 * 1.225 * self.v_wind * self.v_wind * self.area_sail * cl  # * adj

        adj = 1.0
        # if alpha_eff < 0:
        #    adj = -1.0

        # conver lift to global axis
        self.l_x = adj * lift * np.cos(self.gamma_prime - np.pi / 2.0)
        self.l_y = adj * lift * np.sin(self.gamma_prime - np.pi / 2.0)

        # drag
        cd = np.interp(np.rad2deg(alpha_eff), self._naca0012['alpha'], self._naca0012['cd'])
        # convert nondimensional drag to dimensional drag
        drag = 0.5 * 1.225 * self.v_wind * self.v_wind * self.area_sail * cd

        # convert drag to gloval axis
        self.d_x = drag * np.sin(self.gamma_prime - np.pi / 2.0)
        self.d_y = -drag * np.cos(self.gamma_prime - np.pi / 2.0)

    def calc_rudder_forces(self):
        """
        calculates the lift and drag of the
        :return:
        """
        # lift
        delta_eff = self.delta
        while delta_eff < 0:
            delta_eff += 2.0 * np.pi
        cl = np.interp(np.rad2deg(delta_eff), self._naca0012['alpha'], self._naca0012['cl'])
        # convert nondimensional lift to dimensional lift
        lift = 0.5 * 997 * self.v_boat * self.v_boat * self.area_rudder * cl  # * adj

        # conver lift to global axis
        self.l_rudder_x = lift * np.sin(self.phi)
        self.l_rudder_y = lift * np.cos(self.phi)

        # drag
        cd = np.interp(np.rad2deg(delta_eff), self._naca0012['alpha'], self._naca0012['cd'])
        # convert nondimensional drag to dimensional drag
        drag = 0.5 * 997 * self.v_boat * self.v_boat * self.area_rudder * cd

        # convert drag to gloval axis
        self.d_rudder_x = -drag * np.cos(self.phi)
        self.d_rudder_y = drag * np.sin(self.phi)

    def calc_hull_drag(self):
        """
        gets the drag produced on the hull by the boat moving through the water
        :return: drag of the boats hull
        """
        # calculate reynolds number of the boat
        if self.v_boat == 0:
            # handle divide by zero case
            re = 1
        else:
            re = 997 * np.abs(self.v_boat) * self.len_hull / (8.90 * 10e-4)

        # skin friction coefficient
        cf = 0.025 / np.power((np.log10(re) - 2), 2.0)

        # calculate the froude number
        fr = np.abs(self.v_boat) / np.sqrt(9.807 * self.len_hull)

        # get the profile drag
        cw = 0.0022 * np.exp((fr - 0.33) / 0.057)

        # return the dimensional drag of the hull
        self.hull_drag = 0.5 * 997 * np.abs(self.v_boat) * np.abs(
            self.v_boat) * self.area_hull * cf + 0.5 * 997 * np.abs(self.v_boat) * np.abs(
            self.v_boat) * self.area_hull * cw

    def set_moi(self):
        """
        approximates the moment of intertia for the boat as the MOI for a cylinder.
        :return:
        """
        r = np.sqrt(self.area_hull / np.pi)
        self.moi = 0.25 * self.mass * r * r + 1.0 / 12.0 * self.mass * self.len_hull * self.len_hull

    def set_control(self, beta, delta):

        # handle turning sail CCW to much or CW to much. Sail not allowed to cross 180 or -180

        '''
        if self.beta <= np.pi and beta > np.pi:
            self.beta = np.pi
        elif self.beta >= np.pi and beta < np.pi:
            self.beta = np.pi
        else:
            self.beta = beta
        '''
        self.beta = beta

        # TODO check the bounds are correct
        # maintain the rudder cannot turn more than +- 45 [deg]
        if self.delta <= np.pi / 4.0 and delta > np.pi / 4.0:
            self.delta = np.pi / 4.0
        elif self.delta >= np.pi * 7.0 / 4.0 and delta < np.pi * 7.0 / 4.0:
            self.delta = np.pi * 7.0 / 4.0
        else:
            self.delta = delta

    def set_state(self, gamma, beta, delta, phi, v_boat_x, v_boat_y, v_wind):
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.phi = phi
        self.v_boat_x = v_boat_x
        self.v_boat_y = v_boat_y
        self.v_wind = v_wind

    def get_pos(self):
        return self.x, self.y

    def get_vel(self):
        return self.v_boat_x, self.v_boat_y

    def get_state(self):
        return self.gamma, self.beta, self.delta, self.phi

    @staticmethod
    def challenger(delta_t):
        """
        default conditions for a challenger sailboat
        :param delta_t:
        :return:
        """
        sb = SailBoat(delta_t, mass=1814, area_sail=23, area_hull=4, len_hull=7, area_rudder=0.25)
        return sb


class Lidar:

    def __init__(self, num_objects, max_dist):
        """
        constructor for Lidar sensor
        :param num_objects: maximum number of objects the lidar sensor can detect
        :param max_dist: the maximum senseing distance of the lidar beams
        """
        self.num_objects = num_objects
        self.max_dist = max_dist

        # each row corresponds to one object. The columns are for the following data points
        # 0: distance to the center of the obstacle
        # 1: distance of the tangent line to the starboard of the center line from the boats view point
        # 2: distance of the tangent line to the port of the center line form the boats view point
        # 3: radius of the obstacle
        # 4: angle of the line to the center of the obstacle
        # 5: angle of the line of the starboard tangent line
        # 6: angle of the line of the port tangent line
        self.measurments = np.zeros((num_objects,7))




class GPS:
    """
    gps is a class that simulates processed GPS data for a boat. The gps keeps track of obsticles, and calculates
    relative distances, angles, and masking for the simulation domain
    """
    def __init__(self, obstacles, domain, n_bins):

        # the x layers are as follows:
        #   delta x position of the center
        #   delta y position of the center
        #   mask of the cell
        self.data = np.zeros((3,n_bins[1],n_bins[0]))

        if n_bins[0] % 2 == 0 or n_bins[1] % 2 == 0:
            raise ValueError('n_bins for the GPS must be an odd number')

        x_edges = np.linspace(-domain/2,domain/2,n_bins[0]+1)
        x_center = np.zeros(len(x_edges)-1)
        for i in range(len(x_center)):
            x_center[i] = (x_edges[i]+ x_edges[i+1])/2.0
        self.x_buffer = x_center[0] - x_edges[0]

        y_edges = np.linspace(-domain/2,domain/2, n_bins[1] + 1)
        y_center = np.zeros(len(y_edges) - 1)
        for i in range(len(y_center)):
            y_center[i] = (y_edges[i] + y_edges[i + 1]) / 2.0
        self.y_buffer = y_center[0] - y_edges[0]

        self.x_edges, self.y_edges = np.meshgrid(x_edges, y_edges)
        self.x_pos, self.y_pos = np.meshgrid(x_center,y_center)

        self.data[0,:,:] = self.x_pos
        self.data[1,:,:] = self.y_pos

        self.obstacles = obstacles

    def update_mask(self,boat):

        ax1_len = len(self.data[0, :, 0])
        ax2_len = len(self.data[0, 0, :])

        self.data[2,:,:] = 0

        tmp_x = np.add(np.subtract(np.multiply(self.data[0,:,:],np.cos(boat.phi)), np.multiply(self.data[1, :, :],np.sin(boat.phi))),boat.x)
        tmp_y = np.add(np.add(np.multiply(self.data[0,:,:],np.sin(boat.phi)), np.multiply(self.data[1,:,:],np.cos(boat.phi))),boat.y)

        data = []
        for k in range(len(self.obstacles)):
        #for k in range(1):
            tmp_dist = np.sqrt( np.add(np.multiply(np.subtract(tmp_x,self.obstacles[k][0]),np.subtract(tmp_x,self.obstacles[k][0])),np.multiply(np.subtract(tmp_y,self.obstacles[k][1]),np.subtract(tmp_y,self.obstacles[k][1])))  )

            data.append(np.where(tmp_dist < self.obstacles[k][2]*1.1, 1,0))

        mask = data[0]
        for i in range(1,len(data)):
            mask = np.add(mask,data[i])

        mask = np.where(mask>1,1,mask)

        self.data[2,:,:] = mask

        '''
        for i in range(ax1_len):
            for j in range(ax2_len):

                # take the point and calcualte x,y position and rotate it relative to the boat
                tmp_x_old = self.data[0,i,j]*np.cos(boat.phi) - self.data[1, i, j]*np.sin(boat.phi) +boat.x
                tmp_y_old = self.data[0, i, j]*np.sin(boat.phi) + self.data[1, i, j]*np.cos(boat.phi) + boat.y

                # check if center is within obstacle via distance to the center
                for k in range(len(self.obstacles)):

                    tmp_dist_old = np.sqrt((tmp_x_old-self.obstacles[k][0])*(tmp_x_old-self.obstacles[k][0]) + (tmp_y_old-self.obstacles[k][1])*(tmp_y_old-self.obstacles[k][1]))
                    if tmp_dist_old < self.obstacles[k][2]*1.1: # if distance is less than 110% of the radius mark cell as masked
                        self.data[2,i,j] = 1
        '''