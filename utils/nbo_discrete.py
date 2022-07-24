'''
The Nominal Belief Optimization (NBO) method for multisensor target tracking with discrete action space

[1]     Miller, Scott A., Zachary A. Harris, and Edwin KP Chong. "A POMDP framework for coordinated 
        guidance of autonomous UAVs for multitarget tracking." EURASIP Journal on Advances in 
        Signal Processing 2009 (2009): 1-17.
'''
import copy
from os import stat
from tracemalloc import start
import numpy as np
from pyparsing import RecursiveGrammarException
from sympy import Id
from utils.msg import Agent_basic, Single_track, Info_sense, Planning_msg
from utils.dec_agent import dec_agent
from utils.occupancy import occupancy
import utils.util as util
from scipy.optimize import differential_evolution, minimize
from joblib import Parallel, delayed
import psutil
import time
from shapely.geometry import Point
from mystic.solvers import diffev2, diffev
from pathos.pools import ProcessPool
from mystic.monitors import Monitor, VerboseMonitor, CustomMonitor, VerboseLoggingMonitor
from mystic.strategy import Rand1Bin
from scipy.optimize import linear_sum_assignment   # Hungarian alg, minimun bipartite matching
import itertools
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.utils.search.grid_search import GridSearch
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from utils.debug_video import debug_video
from collections import namedtuple
from utils.nbo import nbo_agent

class nbo_agent_discrete(nbo_agent):

    def __init__(self, horizon, sensor_para_list, agentid, dt, cdt, opt_step = -1,
                L0 = 5, v = 5, isObsdyn__=True, isRotate = False, gamma = 1.0, 
                ftol = 5e-3, gtol = 7, isParallel = True, SemanticMap=None, sigma = False, 
                distriRollout = True, OccupancyMap=None, IsStatic = False,
                factor = 5, penalty = 0, lite = False, central_kf = False, optmethod='de',
                wtp = True, info_gain = False, action_space = None, dv = 1e0):

        nbo_agent.__init__(self, horizon, sensor_para_list, agentid, dt, cdt, opt_step = opt_step,
                L0 = L0, v = v, isObsdyn__ = isObsdyn__, isRotate = isRotate, gamma = gamma, 
                isParallel = isParallel, SemanticMap=SemanticMap, sigma = sigma, 
                distriRollout = distriRollout, OccupancyMap=OccupancyMap, IsStatic = IsStatic,
                factor = factor, penalty = penalty, lite = lite, central_kf = central_kf, optmethod='discrete',
                wtp = wtp, info_gain = info_gain, action_space = action_space)
        
        
        self.State = namedtuple("State", "x, y")
        self.TimeState = namedtuple("State", "x, y, t")
        self.Action = namedtuple("Action", "dx, dy")
        self.trajectory = namedtuple("trajectory", "action_traj, reward, P_list")
        # self.dx = dx                                        # grid resolution
        self.dx = dv * self.SampleDt
        self.grid_num = int(self.v_bar / dv)
        self.action_space = []
        # for ax in range(self.grid_num):
        #     act = self.Action(int(dx * ax) * dx, 0)
        #     self.action_space.append(act)

        # for ay in range(self.grid_num):
        #     act = self.Action(0, int(dx * ay) * dx)
        #     self.action_space.append(act)
        
        for ax in range(-self.grid_num, self.grid_num + 1):
            for ay in range(-self.grid_num, self.grid_num + 1):
                if np.sqrt(ax ** 2 + ay ** 2) * self.dx <= self.v_bar * self.SampleDt:
                    act = self.Action(ax, ay)
                    self.action_space.append(act)
            
        self.value_map = {}
        self.traj_map = {}
        self.other_agent_position = {}
        self.R_pred_other_agent = {}
        self.origin_pos = None

    def rollout(self, u):

        self.sampling()
        if self.parallel:
            return self.parallelNBO()
        else:
            return self.standardNBO(u)
    
    def parallelNBO(self):
        '''
        Parallel decision making in NBO, the distributed case is PMA, 
        and centralized case is dec-POMDP
        '''
        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            
            return [[0, 0]] * self.opt_step
                    
        # PMA, agent use other agents previous decision
        self.u = self.policy_stack

        if self.distriRollout:

            action_vector, reward = self.optimize_single_discrete()
            # Perform optimization
            # print(action_vector)
            self.opt_value = reward
            self.v = action_vector[0]

            
        else:
            raise RuntimeError("not defined discrete optimization of joint action")
        self.reset()     # release the memory
        return action_vector

    def base_policy_discrete(self, agent_id: int, start_t: int):
        """ 
        find the worst track based on its P matrix in the beginning of the plan,
        agent_pos at horizon start_t - 1 and target position at time start_t - 1
        """

        worst_P, worst_track_id = -1, -1
        if start_t == 0:
            agent_pos = copy.deepcopy(self.sensor_para_list[agent_id]["position"])
        else:
            agent_pos = copy.deepcopy(self.other_agent_position[agent_id][start_t - 1])
        # assert len(agent_pos) == 3, RuntimeError("does not include theta")
        state = self.TimeState(0, 0, 0)
        # find the track inside FoV with highest P
        # print(len(self.trajInTime[start_t]))
        for i in range(len(self.trajInTime[start_t])):
            target_est_pos = self.trajInTime[start_t][i]
            
            if self.euclidan_dist(target_est_pos, agent_pos) < self.dm:
                
                P = np.trace(self.traj_map[state].P_list[i])
                if P > worst_P:
                    worst_P = P
                    worst_track_id = i
        
        if worst_track_id < 0:
            # no target around, just stay 
            remain_positions = [agent_pos] * (self.horizon - start_t)

        else:
            # once find the worst track, move towards it in the remaining time
            remain_positions = []
            for t in range(start_t, self.horizon):
                target_est_pos = self.trajInTime[t][i]

                dx = agent_pos[0] - target_est_pos[0]
                dy = agent_pos[1] - target_est_pos[1]
                theta = np.arctan2(dy, dx)
                dist = self.euclidan_dist(agent_pos, target_est_pos)
                if dist >= self.v_bar * self.SampleDt:
                    dx = self.v_bar * self.SampleDt * np.cos(theta)
                    dy = self.v_bar * self.SampleDt * np.sin(theta)

                agent_pos[0] += dx
                agent_pos[1] += dy
                agent_pos[2] = theta
                remain_positions.append(agent_pos)
        
        self.other_agent_position[agent_id] += remain_positions
        # print(self.other_agent_position[agent_id])
    
    def isInFoV(self, z: list, agent_id: int, pos: list) -> bool:
        # give z = [x, y], check if it's inside FoV of agnet_id at pos
        if self.sensor_para_list[agent_id]["shape"][0] == "circle":
            r = self.sensor_para_list[agent_id]["shape"][1]
            
            distance = self.euclidan_dist(pos, z)
            if distance > r:
                return False
            else:
                return True
        elif self.sensor_para_list[agent_id]["shape"][0] == "rectangle":
            # analysis the shape of rectangle
            width = self.sensor_para_list[agent_id]["shape"][1][0]
            height = self.sensor_para_list[agent_id]["shape"][1][1]
            angle = self.sensor_para_list[agent_id]["position"][2]
            
    
            dx = z[0] - pos[0]
            dy = z[1] - pos[1]
            dx_trans = dx * np.cos(angle) + dy * np.sin(angle)
            dy_trans = - dx * np.sin(angle) + dy * np.cos(angle)
            
            if 2 * abs(dx_trans) <= width and 2 * abs(dy_trans) <= height:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def prediction(P, F, Q):
        return np.dot(F, np.dot(P, F.T)) + Q

    def reset(self):
        self.traj_map = {}
        self.value_map = {}
        self.R_pred_other_agent = {}
        self.other_agent_position = {}

    def optimize_single_discrete(self):
        """ Body function of optimizing the ego agent plan,
        based on
        self.policy_stack: intention of other agents at last decision epoch

        """
        # 1. init all agents position, derive 
        # self.other_agent_position[time][agent_id]: all other agents future position
        # self.R_pred_other_agent[time][track_id]: info gain of other agents
        time_state = self.TimeState(0, 0, 0)
        state = self.State(0, 0)
        P_list = []
        for j in range(len(self.local_track["id"])):
            info = self.local_track["infos"][j][0]
            P0 = np.matrix(info.P).reshape(4,4)
            P_list.append(P0)
        traj = self.trajectory([], 0.0, P_list)
        self.value_map[state] = 0.0
        self.traj_map[time_state] = traj
        self.agent_position_precalculate()

        # 2. for each time, propagate to new state
        for t in range(self.horizon):
            for time_state in self.traj_map.copy():
                if time_state.t == t:
                    for action in self.action_space:
                        self.single_one_step_dynamic_cov_gain(time_state, action)

        # 3. find the best action sequence and return
        best_last_pos = max(self.value_map, key=self.value_map.get)
        best_last_timestate = self.TimeState(best_last_pos.x, best_last_pos.y, self.horizon)
        best_action = self.traj_map[best_last_timestate].action_traj
        best_value = self.value_map[best_last_pos]
        # print(best_value, self.traj_map[best_last_timestate].reward, self.value_map)
        # print(best_action)
        assert np.isclose(best_value, self.traj_map[best_last_timestate].reward), RuntimeError("suspicious result")
        # process the policy and return
        return best_action, best_value

    def cal_R_overload(self, agent_id: int, xs: list, z: list):
        # xs: sensor pos, = [x, y, theta]
        # z: measurement, [x, y] 
        dx = z[0] - xs[0]
        dy = z[1] - xs[1]
        theta = np.arctan2(dy, dx) - xs[2]
        r = max(self.sensor_para_list[agent_id]["r0"] , np.sqrt(dx**2 + dy**2))
        G = np.matrix([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        M = np.diag([0.1 * r, 0.1 * np.pi * r])
        R = self.sensor_para_list[agent_id]["quality"] * np.dot(np.dot(G, M), G.T)
        
        return R

    def single_one_step_dynamic_cov_gain(self, time_state, action):
        '''
        sequential centralized kf update, reward is cov_gain, 
        obj. max P_k - P_k+1
        '''
        t = time_state.t
        objValue = 0.0
        traj = self.traj_map[time_state]        
        z_k = self.trajInTime[t]
        new_ego_pos = [(time_state.x + action.dx) * self.dx + self.origin_pos[0], 
            (time_state.y + action.dy) * self.dx + self.origin_pos[0], np.arctan2(action.dy, action.dx)]
        new_P_list = []
        # start using a sequential way to estimate KF
        for j in range(len(z_k)):
            z = z_k[j]
            
            P_k_k_min = self.prediction(traj.P_list[j], self.SampleF, self.factor * self.tracker.Q)
            
            if not self.InsideObstacle([z[0], z[1]]) and self.isInFoV(z, self.id, new_ego_pos):
                R_i = self.cal_R_overload(self.id, new_ego_pos, z) 
                R = self.R_pred_other_agent[t][j] + np.linalg.inv(R_i)
            else:
                R = self.R_pred_other_agent[t][j]
            # update all agents track
            info_sum = np.dot(np.dot(self.tracker.H.T, R), self.tracker.H)
            P_fused = np.linalg.inv(np.linalg.inv(P_k_k_min) + info_sum)
            new_P_list.append(P_fused)
            trace =  np.trace(traj.P_list[j][0:2, 0:2]) - np.trace(P_fused[0:2, 0:2])
            objValue += trace # (self.gamma ** i) # * weight

        next_state = self.State(time_state.x + action.dx, time_state.y + action.dy) 
        next_time_state = self.TimeState(time_state.x + action.dx, time_state.y + action.dy, t+1)
        cur_reward = traj.reward + objValue
        if next_time_state in self.traj_map:
            # compare with exisiting value
            if cur_reward < self.value_map[next_state]:
                return

        # this is the current best, replace the trajectory and value        
        self.value_map[next_state] = cur_reward
        action_traj = traj.action_traj + [[action.dx * self.dx, action.dy * self.dx]]
        new_traj = self.trajectory(action_traj, cur_reward, new_P_list)
        self.traj_map[next_time_state] = new_traj
        # test passed
        # if t+1 != len(action_traj):
        #     print(new_traj.action_traj, t, len(action_traj))
            
        return objValue

    def agent_position_precalculate(self):
        """ 
        this function calculate the other agents position based on the 
        'intention' term collected. As we know there will be communication loss, 
        begin of the event or so, there is no guarantee to have all agents location
        at same length
        traj: self.trajectory, action: self.Action
        """

        # calculate all agents position in self.horizon
        for id_ in range(self.sensor_num):
            if id_ != self.id:
                self.other_agent_position[id_] = []
                x, y = self.sensor_para_list[id_]["position"][0], self.sensor_para_list[id_]["position"][1]
                for t in range(self.horizon):
                    # try to grab the policy from intention
                    try:
                        action = self.policy_stack[id_][t]
                        x += action[0]
                        y += action[1]
                        theta = np.arctan2(action[1], action[0])
                        self.other_agent_position[id_].append([x, y, theta])
                    except:
                        # use base policy to add the remaining
                        self.base_policy_discrete(id_, t)
                        # assert len(self.other_agent_position[id_]) == self.horizon, print(len(self.other_agent_position[id_]), self.horizon)
                        break
            
            # ego agent, save its current position
            else:
                self.origin_pos = (self.sensor_para_list[id_]["position"][0], 
                    self.sensor_para_list[id_]["position"][1], self.sensor_para_list[id_]["position"][2])

        # based on all other agents position, precalculate the information gain provided by all other
        # agents
        
        for t in range(self.horizon):
            R_list = []
            for i in range(len(self.trajInTime[t])):
                z = self.trajInTime[t][i]

                R = np.matrix(np.zeros((2, 2)))
                # find any observing agents
                for id_ in range(self.sensor_num):
                    if id_ != self.id:
                        agent_pos = self.other_agent_position[id_][t]
                        if self.isInFoV(z, id_, agent_pos):
                            R_i = self.cal_R_overload(id_, agent_pos, z) 
                            R += np.linalg.inv(R_i)
                
                R_list.append(R)
            self.R_pred_other_agent[t] = R_list


    def optimizer(self):
        # A DP way with pruning to select best agent
        pass