'''
The Monte Carlo method of multi-agent rollout in target tracking format, based
on the description in

[1]     Li, Tianqi, Lucas W. Krakow, and Swaminathan Gopalswamy. "Optimizing Consensus-based
        Multi-target Tracking with Multiagent Rollout Control Policies." arXiv preprint 
        arXiv:2102.02919 (2021).

'''


import copy
import numpy as np
from utils.dec_agent_jpda import dec_agent_jpda
import utils.util as util
import psutil
import pyswarms as ps
from typing import List, Any, Optional, Union


class MCRollout(dec_agent_jpda):
    def __init__(
            self, 
            horizon: int, 
            sensor_para_list: dict, 
            agentid: int, 
            dt: float, 
            cdt: float, 
            opt_step: int = 1,
            L0: int = 5, 
            isObsdyn: bool=True, 
            isRotate: bool = False, 
            gamma: float = 1.0, 
            ftol: float = 5e-3, 
            gtol: float = 7, 
            MCSnum: int = 50,
            SemanticMap: Optional[dict]=None, 
            OccupancyMap: Optional[dict] = None, 
            distriRollout: bool = True, 
            factor: float = 5, 
            penalty: float = 0, 
            central_kf: bool = True, 
            optmethod: str ='pso',
            wtp: bool = False,
            info_gain: bool = False, 
            IsStatic: bool = False,
        ):
        
        dec_agent_jpda.__init__(self, sensor_para_list, agentid, dt, cdt, L0 = L0, isObsdyn=isObsdyn, 
            isRotate = isRotate, NoiseResistant =False, SemanticMap=SemanticMap, OccupancyMap=OccupancyMap)
        
        self.sensor_num = len(sensor_para_list)
        self.horizon = horizon
        self.samples = {}
        self.ftol = ftol
        self.gtol = gtol
        self.u = [] 
        self.gamma = gamma
        self.cdt = cdt
        self.factor = factor
        self.aimid = -1
        self.SampleDt = dt * self.factor
        if IsStatic:
            self.SampleF = np.matrix(np.eye(4))
            self.SampleQ = np.matrix(np.zeros((4, 4)))
        else:
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
        self.opt_step = opt_step
        self.distriRollout = distriRollout
        self.policy_stack = []
        for _ in range(self.sensor_num):
            self.policy_stack.append([])
        self.v_bar_opt = sensor_para_list[agentid]["v"] * 0.2
        self.penalty = penalty

        self.central_kf = central_kf
        self.npop = 2*self.sensor_num * self.opt_step * 10
        self.optmethod = optmethod
        self.gamma = 0.5
        self.MCSnum = MCSnum
        self.trajInTime_list = []
        self.wtp = wtp
        self.info_gain = info_gain     # indicate the optimal object if information gain
        
        self.opt_value = -1                 # saves the output of optimization
    
    def planning(self, u: List[List[float]]) -> Union[List[List[float]], List[List[List[float]]]]:
        
        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            if self.distriRollout:
                return u 
            else:
                agent_act = [[0,0]] 
                return [agent_act] * self.sensor_num
            
        else:
            # takes over some other agents updated work
            self.u = u

        self.sampling()
        
        if self.distriRollout:
            if self.optmethod == 'pso':
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                # bound on riemannian
                min_bound = [0, 0] 
                max_bound = [self.v_bar, 2 * np.pi]
                bounds = (np.array(min_bound), np.array(max_bound))
                
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
        
                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                
                self.opt_value = cost
                
                x = [pos[0], pos[1]]
                v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                # update its action
                self.u[self.id] = v
                self.v = v
                
            else:
                raise RuntimeError('optimization method undefined!')
        else:
            # centralized rollout
            if self.optmethod == 'pso':
                #  particle swarm optimization
                # https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html#Basic-Optimization-with-Arguments
                # source code
                # https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/single/global_best.html
                # setup hyperparametersa
                # strategies https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/backend/handlers.html#BoundaryHandler
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                # bound on riemannian
                min_bound = [0, 0] * (self.opt_step * self.sensor_num)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step * self.sensor_num)
                bounds = (np.array(min_bound), np.array(max_bound))

                optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, dimensions=2*self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                v_vector = []
                for ids in range(self.sensor_num):
                    v = pos[2 * (ids * self.opt_step)]
                    theta = pos[2 * ids * self.opt_step + 1]
                    action = [v*np.cos(theta), v*np.sin(theta)]
                    v_vector.append(action)
                self.u = v_vector
                self.v = self.u[self.id]
            else:
                raise RuntimeError('optimization method undefined!')
                
        return copy.deepcopy(self.u)

           
    def fobj_pso(self, u_i: np.ndarray) -> np.array:
        length = len(u_i)       
        cost_list = []
        for seed in range(length):
            cost_list.append(self.fpso(u_i[seed, :]))
              
        return np.array(cost_list)

    def fpso(self, u_i: List[float]) -> float:
        '''here u is an array'''
        if self.distriRollout:
            
            u = copy.deepcopy(self.u)
            x = [u_i[0], u_i[1]]    
            v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
            u[self.id] = v
        else:
            u = []
            
            for ids in range(self.sensor_num):
                x = [u_i[2*ids], u_i[2 * ids + 1]]
                v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                u.append(v)
        return self.f(u)

    def f(self, u: List[List[float]]) -> float:
        value = 0.0

        for trialnum in range(self.MCSnum):
            if self.central_kf:
                value += self.rolloutTrialSim_centralized_P(u, trialnum)
            else:
                # other ways for faster computation?
                pass
        # minimal value
        value /= (len(self.local_track["id"]) * self.MCSnum)
                
        return value
    

    def rolloutTrialSim_centralized_P(self, u: List[List[float]], trialnum: int) -> float:
        '''
        centralized kf update by information filter update on covariance
        '''
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        
        neighbor_num = len(self.neighbor["id"]) + 11
        
        for i in range(len(self.sensor_para_list)):
            if i in self.neighbor["id"] or i==self.id:
                
                ego = dec_agent_jpda(copy.deepcopy(self.sensor_para_list), i, self.SampleDt, self.cdt, 
                        L0 = self.L, isObsdyn__ = self.tracker.isObsdyn, isRotate = self.isRotate, 
                        isVirtual = True, SemanticMap=self.SemanticMap)
                
                # skip data association, only have KF object
                
                for j in range(len(self.local_track["id"])):
                    info = self.local_track["infos"][j][0]
                    x0 = np.matrix(info.x).reshape(4,1)
                    P0 = np.matrix(info.P).reshape(4,4)
                    if self.isObsdyn:
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]), self.sensor_para_list[i]["r0"], 
                            quality=self.sensor_para_list[i]["quality"])
                        
                    else:
                        kf = util.LinearKF(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]))
                    kf.init = False
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
            z_k = self.trajInTime_list[trialnum][i]
            
            # start using a sequential way to estimate KF
            for j in range(len(z_k)):
                z = z_k[j]
                
                agent_list[0].tracker[j].predict()
                if not self.isInsideOcclusion([z[0], z[1]]):
                    # do sequential update, use P = [P^-1 + \sum_i H^T R^-1_i H] ^ -1
                    index = -1
                    R = np.matrix(np.zeros((2, 2)))
                    # find any observing agents
                    for k in range(neighbor_num):
                        if agent_list[k].isInFoV(z):
                            R_i = agent_list[k].tracker[j].cal_R(z, agent_list[k].sensor_para["position"]) 
                            R += np.linalg.inv(R_i) * agent_list[k].sensor_para["quality"]
                            index = k

                    # update all agents track
                    if index >= 0:
                        info_sum = np.dot(np.dot(self.tracker.H.T, R), self.tracker.H)
                        P_fused = np.linalg.inv(np.linalg.inv(agent_list[0].tracker[j].P_k_k_min) + info_sum)
                        agent_list[0].tracker[j].P_k_k = copy.deepcopy(P_fused)
                        
                    else:
                        agent_list[0].tracker[j].P_k_k = copy.deepcopy(agent_list[0].tracker[j].P_k_k_min)
            
                trace = np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                                
                objValue += trace # (self.gamma ** i) # * weight
            
            # 4. agent movement policy
            if np.isclose(t-tc, self.cdt) and (i <= self.SampleNum-2):
                tc = t
                # print("at time %s" % (t - self.t))
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    agent.base_policy()
            
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        
        # multiple weighted trace penalty (MWTP) by [1]
        if self.wtp:
            # wtp = 0.0
            # Dj = [0] * neighbor_num

            # for j in range(len(self.trajInTime[-1])):
            #     z = self.trajInTime[-1][j]            
                
            #     if self.isInsideOcclusion([z[0], z[1]]):
            #         continue
            #     isoutside = True
            #     for k in range(neighbor_num):
            #         isoutside = (not agent_list[k].isInFoV(z)) and isoutside
            #     if isoutside:
            #         # find the minimium distance from sensor to target
            #         distance = 10000
            #         index = 0
            #         for k in range(neighbor_num):
            #             d_k = self.euclidan_dist(z, agent_list[k].sensor_para["position"][0:2])
            #             if d_k < distance and Dj[k] == 0:
            #                 index = k
            #                 distance = d_k

            #         wtp += self.gamma * distance * np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
            #         Dj[index] += 1
            # objValue += wtp
            
            # revised wtp, which is complete since we have ordered the traces
            wtp = 0.0
            Dj = [0] * neighbor_num
            trace_list_ordered = []
            for j in range(len(self.trajInTime_list[trialnum][-1])):
                z = self.trajInTime_list[trialnum][-1][j]            
                
                if self.isInsideOcclusion([z[0], z[1]]):
                    continue
                isoutside = True
                for k in range(neighbor_num):
                    isoutside = (not agent_list[k].isInFoV(z)) and isoutside
                if isoutside:
                    P_trace = np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                    trace_list_ordered.append([P_trace, z])
            trace_list_ordered.sort(key=lambda r: r[0], reverse=True)
            # go with the order
            for j in range(len(trace_list_ordered)):
                z = trace_list_ordered[j][1]
                # find the minimium distance from sensor to target
                distance = 10000
                index = 0
                for k in range(neighbor_num):
                    d_k = self.euclidan_dist(z, agent_list[k].sensor_para["position"][0:2])
                    # print("d_k %s, distance %s, Dj %s, k %s" % (d_k, distance, Dj, k))
                    if d_k < distance and Dj[k] == 0:
                        index = k
                        distance = d_k

                wtp += self.gamma * distance * np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                Dj[index] += 1
            
            objValue += wtp

        return objValue


    def sampling(self):
        '''Monte Carlo simulation, at the beginning of each rollout period, sampling trajectories of targets
        based on current state of tracks'''
        # sample for self.MCSnum of trails for first time step, then sample 1 for the rest time steps
        MCSample = {}
        self.trajInTime_list = []
        for i in range(len(self.local_track["id"])):
           
            trajectoryList = []
            x = np.array(self.local_track["infos"][i][0].x)
            P = np.reshape(self.local_track["infos"][i][0].P, (4, 4))
            samples = np.random.multivariate_normal(x, P, self.MCSnum).T
            
            for j in range(self.MCSnum):
                trajectory = []
                seed = samples[:, j]
                for k in range(self.SampleNum):
                    # sample one place
                    v = np.random.multivariate_normal(np.zeros(4), self.SampleQ, 1)
                    seed = np.asarray(np.dot(self.SampleF, seed) + v)[0]
                    trajectory.append(list(seed)[0:2])
                trajectoryList.append(trajectory)
            MCSample[self.local_track["id"][i]] = trajectoryList
        
        self.samples = MCSample
        for i in range(self.MCSnum):
            trajectory = []
            for id_ in self.local_track["id"]:
                traj_i = self.samples[id_][i]
                trajectory.append(traj_i)
            trajInTime = []
            for j in range(self.SampleNum):
                trajInTime.append(self.column(trajectory, j))
            self.trajInTime_list.append(trajInTime)
    
    def column(self, matrix: List[List[Any]], i: int) -> Any:
        '''return the ith columns values in a list of list structure'''
        return [row[i] for row in matrix]