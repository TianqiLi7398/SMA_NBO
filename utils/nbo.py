'''
The Nominal Belief Optimization (NBO) method for multisensor target tracking

[1]     Miller, Scott A., Zachary A. Harris, and Edwin KP Chong. "A POMDP framework for coordinated 
        guidance of autonomous UAVs for multitarget tracking." EURASIP Journal on Advances in 
        Signal Processing 2009 (2009): 1-17.
'''
import copy
import numpy as np
from utils.msg import Planning_msg
from utils.dec_agent_jpda import dec_agent_jpda
import utils.util as util
import psutil
import pyswarms as ps
from typing import Any, List


class nbo_agent(dec_agent_jpda):
    '''
    Nonimal Belief Optimization agent: makes decision based on 
    others policy and the approximation provided by nominal trajectories
    '''
    def __init__(
            self, 
            horizon: int, 
            sensor_para_list: dict, 
            agentid: int, 
            dt: float, 
            cdt: float, 
            opt_step: int = -1,
            L0: int = 5, 
            isObsdyn: bool =True, 
            isRotate: bool = False,
            gamma: float = 1.0, 
            ftol: float = 5e-3, 
            gtol: float = 7, 
            SemanticMap: Any=None, 
            IsDistriOpt: bool = True, 
            OccupancyMap: bool=None, 
            IsStatic:bool = False,
            factor: float = 5, 
            penalty: float = 0, 
            central_kf: bool = True, 
            optmethod: str='pso',
            wtp: bool = True, 
            info_gain: str = "trace_sum"
        ):
        
        dec_agent_jpda.__init__(self, sensor_para_list, agentid, dt, cdt, L0 = L0, isObsdyn=isObsdyn, 
            isRotate = isRotate, NoiseResistant =False, SemanticMap=SemanticMap, OccupancyMap=OccupancyMap, 
            IsStatic = IsStatic)
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
        
        if opt_step > 0:
            self.opt_step = opt_step   # number of steps to optimize
        else:
            self.opt_step = self.SampleNum
        
        self.IsDistriOpt = IsDistriOpt
        self.policy_stack = []
        for _ in range(self.sensor_num):
            self.policy_stack.append([])
        
        self.penalty = penalty

        self.central_kf = central_kf
        self.npop = 2*self.sensor_num * self.opt_step * 10
        self.optmethod = optmethod
        self.pso_para_record = {'c1': [],
                    'c2': [],
                    'w' : [],
                    'k' : []}
        self.gamma = 0.5
        self.wtp = wtp
        self.info_gain = info_gain     # indicate the optimal object if information gain
        self.opt_value = -1                 # saves the output of optimization
    
    
    def planning(self) -> List[List[float]]:
        '''Parallel decision making in NBO, 
        the distributed case: SMA or PMA, 
        and centralized case: central or dec-POMDP'''
        self.sampling()

        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            
            return [[0, 0]] * self.opt_step
                    
        self.u = self.policy_stack

        if self.IsDistriOpt:
            if self.optmethod == 'pso':
                # normal method for PMA-NBO or SMA-NBO
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, \
                        dimensions=2* self.opt_step, options=options, ftol=self.ftol, \
                        bounds=bounds, ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, \
                        dimensions=2* self.opt_step, options=options, ftol=self.ftol, \
                        bounds=bounds, ftol_iter = self.gtol) # , init_pos = init_pos)

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles = self.npop, 
                        dimensions=2*self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds,  
                        ftol_iter = self.gtol)

                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, \
                    n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                self.opt_value = cost
                action_vector = []
                
                for i in range(self.opt_step):
                    
                    x = [pos[2* i], pos[2* i+ 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    action_vector.append(v)
                # update its action
                self.v = action_vector[0]
                            
            else:
                raise RuntimeError('optimization method undefined!')
        else:
            if self.optmethod == 'pso':
                # method of dec-POMDP or centralized NBO
                
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step * self.sensor_num)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step * self.sensor_num)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, \
                        dimensions=2 * self.sensor_num * self.opt_step, \
                        options=options, ftol=self.ftol, bounds=bounds, \
                        ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, \
                        dimensions=2 * self.sensor_num* self.opt_step, \
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) 

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, \
                        dimensions=2 * self.sensor_num * self.opt_step, \
                        options=options, ftol=self.ftol, bounds=bounds, \
                        ftol_iter = self.gtol)
                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1,\
                    iters=5000, verbose=False)
                self.opt_value = cost
                
                action_vector = []
                
                for i in range(self.opt_step):

                    v = pos[2 * (self.id * self.opt_step + i)]
                    theta = pos[2 * (self.id * self.opt_step + i) + 1]
                    action = [v*np.cos(theta), v*np.sin(theta)]
                    action_vector.append(action)
                
                self.v = action_vector[0]
                        
            else:
                raise RuntimeError('optimization method undefined!')

        return action_vector
    
        
    def fobj_pso(self, u_i: np.ndarray) -> np.array:
        length = len(u_i)

        cost_list = []
        
        for seed in range(length):
            
            cost_list.append(self.fpso(u_i[seed, :]))
              
        return np.array(cost_list)

    def fpso(self, u_i: List[float]) -> float:
        '''here u is an array'''
        u = []
        if self.IsDistriOpt:
            
            for ids in range(self.sensor_num):
                v_list = []
                if ids == self.id:
                    for i in range(self.opt_step):
                        
                        x = [u_i[2*i], u_i[2* i + 1]]
                        v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                        v_list.append(v)
                else:
                    v_list = self.u[ids]
                u.append(v_list)
        else:
            
            for ids in range(self.sensor_num):
                v_list = []
                for i in range(self.opt_step):
                    
                    x = [u_i[2*(self.opt_step*ids + i)], u_i[2* (self.opt_step*ids + i) + 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    v_list.append(v)
                    
                u.append(v_list)
        return self.f(u)

    def f(self, u: List[List[float]]) -> float:
        if self.info_gain == "info_gain":
            value = self.rolloutTrailSim_IF(u)
        elif self.central_kf:
            value = self.rolloutTrailSim_centralized_P(u)
        else:
            raise RuntimeError("%s is not defined as objective in NBO" % self.info_gain)

        value /= (len(self.local_track["id"]))
        return value

    def update_group_decision(self, plan_msg: Planning_msg):
        '''
        input: u with samesize of the output of self.standardNBO(), this function stores the future actions 
        in self.policy_stack for the calculation of Estimation Reward To Go
        '''
        
        miss_list = plan_msg.miss()
        future_policy = []
        for i in range(self.sensor_num):
            if i in miss_list:
                # if i miss, still utlize the previous stack value if we have it
                try:
                    agent_i_policy = self.policy_stack[i][1:]
                except: 
                    agent_i_policy = [] 
            else:
                if plan_msg.u[i].time < self.t:
                    # previous intention
                    
                    agent_i_policy = plan_msg.u[i].action_vector[1:]
                else:
                    agent_i_policy = plan_msg.u[i].action_vector
            future_policy.append(agent_i_policy)
            
        self.policy_stack = future_policy
        
    
    def rolloutTrailSim_centralized_P(self, u: List[List[float]]) -> float:
        '''
        sequential centralized kf update
        '''
    
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        
        neighbor_num = len(self.neighbor["id"]) + 1
        
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
                    ego.v = copy.deepcopy(u[i][0])
                    
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
                
                agent_list[0].tracker[j].predict()
                if not self.isInsideOcclusion([z[0], z[1]]):
                    index = -1
                    R = np.matrix(np.zeros((2, 2)))
                    # find any observing agents
                    for k in range(neighbor_num):
                        if agent_list[k].isInFoV(z):
                            R_i = agent_list[k].tracker[j].cal_R(z, agent_list[k].sensor_para["position"]) 
                            R += np.linalg.inv(R_i)
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
                    try:
                        agent.v = u[j][i + 1]
                        # print("agent %s has policy!" % j)
                    except:
                        agent.base_policy()
            
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        
        # multiple weighted trace penalty (MWTP) by [1]
        if self.wtp:
            
            # revised wtp, which is complete since we have ordered the traces
            wtp = 0.0
            Dj = [0] * neighbor_num
            trace_list_ordered = []
            for j in range(len(self.trajInTime[-1])):
                z = self.trajInTime[-1][j]            
                
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
                new_dist = 100000
                index = -1
                for k in range(neighbor_num):
                    d_k = self.euclidan_dist(z, agent_list[k].sensor_para["position"][0:2])
                    # print("d_k %s, distance %s, Dj %s, k %s" % (d_k, distance, Dj, k))
                    if d_k + Dj[k] < new_dist:
                        index = k
                        distance = d_k
                        new_dist = d_k + Dj[k]
                if index >= 0 and Dj[k] == 0:
                    wtp += self.gamma * distance * np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                Dj[index] += distance
                # change sensor position to the place to cover its fov
                r = min(agent_list[index].sensor_para["shape"][1])
                angle = np.arctan2(agent_list[k].sensor_para["position"][1] - z[1], agent_list[k].sensor_para["position"][0] - z[0])
                agent_list[k].sensor_para["position"][0] += r * np.cos(angle)
                agent_list[k].sensor_para["position"][1] += r * np.sin(angle)
                
            objValue += wtp
            
        return objValue


    def rolloutTrailSim_IF(self, u):
        '''
        sequential centralized kf update based on Information gain
        '''
        
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        
        neighbor_num = len(self.neighbor["id"]) + 1
        
        # info_acc, cov_acc
        prior_trace = 0.0
        posterior_trace = 0.0
        # for j in range(len(self.local_track["id"])):
        #     info = self.local_track["infos"][j][0]
        #     # init_trace += np.trace(np.matrix(info.Sigma).reshape(4,4)[0:2, 0:2])
        #     init_trace += np.trace(np.matrix(info.P).reshape(4,4)[0:2, 0:2])


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
                    ego.v = copy.deepcopy(u[i][0])
                    
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
                
                agent_list[0].tracker[j].predict()

                prior_trace += np.trace(agent_list[0].tracker[j].P_k_k_min[0:2, 0:2])
                if self.isInsideOcclusion([z[0], z[1]]):
                    # get occluded, skip update
                    for k in range(neighbor_num):
                        agent_list[k].tracker[j].x_k_k = copy.deepcopy(agent_list[0].tracker[j].x_k_k_min)
                        agent_list[k].tracker[j].P_k_k = copy.deepcopy(agent_list[0].tracker[j].P_k_k_min)
                        agent_list[k].tracker[j].isUpdated = False
                                        
                else:
                    # do sequential update, find the best one, and update to all agents
                    p =1e5
                    index = -1

                    # find best update among all agents
                    for k in range(neighbor_num):
                        if agent_list[k].isInFoV(z):
                            agent_list[k].tracker[j].predict()
                            agent_list[k].tracker[j].update(z, agent_list[k].sensor_para["position"], 
                                agent_list[k].sensor_para["quality"])
                            # agent_list[0].tracker[j].x_k_k_min = copy.deepcopy(agent_list[0].tracker[j].x_k_k)
                            # agent_list[0].tracker[j].P_k_k_min = copy.deepcopy(agent_list[0].tracker[j].P_k_k)

                            if p > np.trace(agent_list[k].tracker[j].P_k_k):
                                index = k
                                p = np.trace(agent_list[k].tracker[j].P_k_k)

                    # update all agents track
                    if index >= 0:
                        for k in range(neighbor_num):
                            agent_list[k].tracker[j].x_k_k = copy.deepcopy(agent_list[index].tracker[j].x_k_k)
                            agent_list[k].tracker[j].P_k_k = copy.deepcopy(agent_list[index].tracker[j].P_k_k)
                            agent_list[k].tracker[j].isUpdated = True 

                    else:
                        for k in range(neighbor_num):
                            agent_list[k].tracker[j].x_k_k = copy.deepcopy(agent_list[0].tracker[j].x_k_k_min)
                            agent_list[k].tracker[j].P_k_k = copy.deepcopy(agent_list[0].tracker[j].P_k_k_min)
                            agent_list[k].tracker[j].isUpdated = False 
                posterior_trace += np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])
                # info_acc
                # Sigma = np.linalg.inv(agent_list[0].tracker[j].P_k_k)
                # trace = np.trace(Sigma[0:2, 0:2])
                # objValue += trace
                # log_det
                # det_val = np.log(np.linalg.det(agent_list[0].tracker[j].P_k_k))           
                # objValue += det_val 
            # 4. agent movement policy
            if np.isclose(t-tc, self.cdt) and (i <= self.SampleNum-2):
                tc = t
                # print("at time %s" % (t - self.t))
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    try:
                        
                        agent.v = u[j][i + 1]
                        # print("agent %s has policy!" % j)
                    except:
                        agent.base_policy()
                        
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        

        objValue = posterior_trace - prior_trace

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
                
                x = np.asarray(np.dot(self.SampleF, x))[0]
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
    
    