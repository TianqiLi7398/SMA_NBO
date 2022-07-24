'''
iterative best response method in determining a good solution, seems unfinished

'''
import copy
import numpy as np
from utils.msg import Agent_basic, Single_track, Info_sense
from utils.dec_agent import dec_agent
import utils.util as util
from joblib import Parallel, delayed
import psutil
import time
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment   # Hungarian alg, minimun bipartite matching
import itertools


# from filterpy.kalman import JulierSigmaPoints


class ibr_agent(dec_agent):

    def __init__(self, horizon, sensor_para_list, agentid, dt, cdt, 
                L0 = 5, v = 5, isObsdyn__=True, isRotate = False, 
                gamma = 1.0, SemanticMap=None, factor = 5, penalty = 0, 
                central_kf = False, wtp = True):
        # TODO v = 5 delete
        dec_agent.__init__(self, sensor_para_list, agentid, dt, cdt, L0 = L0, v = v, isObsdyn__=isObsdyn__, 
            isRotate = isRotate, NoiseResistant =False, SemanticMap=SemanticMap)
        self.sensor_num = len(sensor_para_list)
        self.horizon = horizon
        self.samples = {}
        self.u = [] 
        self.gamma = gamma
        self.cdt = cdt
        self.factor = factor
        self.SampleDt = dt * self.factor
        self.SampleF = np.matrix([[1, 0, self.SampleDt, 0],
                                [0, 1, 0, self.SampleDt],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self.SampleA = np.matrix([[0.5 * self.SampleDt**2, 0],
                                [0, 0.5 * self.SampleDt**2],
                                [self.SampleDt, 0],
                                [0, self.SampleDt]])
        sigma_a = self.sensor_para_list[agentid]["sigma_a"]
        self.SampleQ = np.dot(np.dot(self.SampleA, np.diag([sigma_a**2, sigma_a**2])), self.SampleA.T)
        self.SampleNum = int(self.horizon / self.SampleDt)
        
        
        self.policy_stack = []
        for i in range(self.sensor_num):
            self.policy_stack.append([])
        self.v_bar_opt = sensor_para_list[agentid]["v"] * 0.2
        self.penalty = penalty
        self.central_kf = central_kf
        self.gamma = 0.5
        self.wtp = wtp

        # needed for rough optimization
        self.dv = 0.5
        # get all possible actions ready
        gridnum = 2 * int(self.v_bar/self.dv) + 1
        v_grid = np.linspace(-self.v_bar, self.v_bar, num=gridnum)
        self.v_space = []
        for vx in v_grid:
            for vy in v_grid:                
                if vx**2 + vy**2 <= self.v_bar**2:
                    self.v_space.append([vx, vy])
        # print(self.v_space)
        # print("total action space is %s"%len(self.v_space))

    def ibr_parallel(self, u):
        self.sampling()

        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            if self.distriRollout:
                return u 
            else:
                agent_act = [0,0]
                return [agent_act] * self.sensor_num
        
        # optimization of current agent given other agents planning
        gain_list = []
        for v in self.v_space:
            gain = self.f(u, v)
            gain_list.append(gain)
        
        # pick the best one
        min_index = gain_list.index(min(gain_list))
        return self.v_space[min_index]
    
    def f(self, u, u_i):
        u_ = copy.deepcopy(u)
        u_[self.id] = u_i
        
        return self.rolloutTrailSim_centralized(u_)

    def rolloutTrailSim_centralized(self, u):
        '''
        sequential centralized kf update
        '''
        
        rate = int(self.cdt / self.dt)
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        
        neighbor_num = len(self.neighbor["id"]) + 1
        
        for i in range(len(self.sensor_para_list)):
            if i in self.neighbor["id"] or i==self.id:
                
                ego = dec_agent(copy.deepcopy(self.sensor_para_list), i, self.SampleDt, self.cdt, 
                        L0 = self.L, isObsdyn__ = self.tracker.isObsdyn, isRotate = self.isRotate, 
                        isVirtual = True, SemanticMap=self.SemanticMap)
                
                # skip data association, only have KF object
                
                for j in range(len(self.local_track["id"])):
                    info = self.local_track["infos"][j][0]
                    x0 = np.matrix(info.x).reshape(4,1)
                    P0 = np.matrix(info.P).reshape(4,4)
                    if self.isObsdyn:
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.SampleQ, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]), self.sensor_para_list[i]["r0"], 
                            quality=self.sensor_para_list[i]["quality"])
                        kf.init = False
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.SampleQ, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]))
                    ego.tracker.append(kf)
                try:
                    
                    ego.v = copy.deepcopy(u[i])
                except:
                    ego.base_policy()
                
                ego.dynamics(self.SampleDt)
                agent_list.append(ego)

        # 2. simulate the Monte Carlo 

        for i in range(self.SampleNum):
            t = self.t + (i + 1) * self.SampleDt
            z_k = self.trajInTime[i]
            
            # start using a sequential way to estimate KF
            for j in range(len(z_k)):
                z = z_k[j]
                pose = Point(z[0], z[1])
                agent_list[0].tracker[j].predict()
                if self.InsideObstacle(pose):
                    # get occluded, skip update
                    agent_list[0].tracker[j].x_k_k = agent_list[0].tracker[j].x_k_k_min
                    agent_list[0].tracker[j].P_k_k = agent_list[0].tracker[j].P_k_k_min
                    agent_list[0].tracker[j].isUpdated = False
                                        
                else:
                    # do sequential update
                    agent_list[0].tracker[j].isUpdated = False
                    for k in range(neighbor_num):
                        if agent_list[k].isInFoV(z):
                            agent_list[0].tracker[j].update(z, agent_list[k].sensor_para["position"], 
                                    agent_list[k].sensor_para["quality"])
                            agent_list[0].tracker[j].x_k_k_min = agent_list[0].tracker[j].x_k_k
                            agent_list[0].tracker[j].P_k_k_min = agent_list[0].tracker[j].P_k_k
                            agent_list[0].tracker[j].isUpdated = True        
    
            for l in range(len(z_k)):
                # SigmaMatrix = np.matrix(track.Sigma).reshape(4, 4)
                trace = np.trace(agent_list[0].tracker[l].P_k_k[0:2, 0:2])
                                
                objValue += trace # (self.gamma ** i) # * weight
            
            # 4. agent movement policy
            if np.isclose(t-tc, self.cdt) and (i <= self.SampleNum-2):
                tc = t
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    try:
                        print("here??")
                        agent.v = u[j][(i // rate)+1]
                    except:
                        agent.base_policy()
            
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        
        # multiple weighted trace penalty (MWTP) by [1]
        if self.wtp:
            wtp = 0.0
            Dj = [0] * neighbor_num

            for j in range(len(self.trajInTime[-1])):
                z = self.trajInTime[-1][j]            
                pose = Point(z[0], z[1])
                if self.InsideObstacle(pose):
                    continue
                isoutside = True
                for k in range(neighbor_num):
                    isoutside = (not agent_list[k].isInFoV(z)) and isoutside
                if isoutside:
                    # find the minimium distrance from sensor to target
                    distance = 10000
                    index = 0
                    for k in range(neighbor_num):
                        d_k = self.euclidan_dist(z, agent_list[k].sensor_para["position"][0:2])
                        if d_k < distance and Dj[k] == 0:
                            index = k
                            distance = d_k

                    wtp += self.gamma * distance * np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                    Dj[index] += 1
            objValue += wtp

        return objValue
        


    def sampling(self):
    
        '''NBO signal, just use the mean to generate 1 trajectory'''
        # sample for self.MCSnum of trails for first time step, then sample 1 for the rest time steps
        MCSample = {}
        
        for i in range(len(self.local_track["id"])):
           
            trajectory = []
            x = np.array(self.local_track["infos"][i][0].x)
            
            for k in range(self.SampleNum):
                # sample one place
                
                x = np.asarray(np.dot(self.tracker.F, x))[0]
                trajectory.append(list(x)[0:2])
                
            MCSample[self.local_track["id"][i]] = trajectory
        
        self.samples = MCSample

        trajectory = []
        for id_ in self.local_track["id"]:
            traj_i = self.samples[id_]
            trajectory.append(traj_i)
        
        self.trajInTime = []
        for j in range(self.SampleNum):
            self.trajInTime.append(self.column(trajectory, j))

    def column(self, matrix, i):
        return [row[i] for row in matrix]