'''
The Nominal Belief Optimization (NBO) method for multisensor target tracking

[1]     Miller, Scott A., Zachary A. Harris, and Edwin KP Chong. "A POMDP framework for coordinated 
        guidance of autonomous UAVs for multitarget tracking." EURASIP Journal on Advances in 
        Signal Processing 2009 (2009): 1-17.
'''
import copy
import numpy as np
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

class nbo_agent(dec_agent):

    def __init__(self, horizon, sensor_para_list, agentid, dt, cdt, opt_step = -1,
                L0 = 5, v = 5, isObsdyn__=True, isRotate = False, gamma = 1.0, 
                ftol = 5e-3, gtol = 7, isParallel = True, SemanticMap=None, sigma = False, 
                distriRollout = True, OccupancyMap=None, IsStatic = False,
                factor = 5, penalty = 0, lite = False, central_kf = False, optmethod='de',
                wtp = True, info_gain = False, action_space = None):
        # TODO v = 5 delete
        dec_agent.__init__(self, sensor_para_list, agentid, dt, cdt, L0 = L0, v = v, isObsdyn__=isObsdyn__, 
            isRotate = isRotate, NoiseResistant =False, SemanticMap=SemanticMap, OccupancyMap=OccupancyMap, 
            sigma = sigma, IsStatic = IsStatic)
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
        
        self.parallel = isParallel
        self.distriRollout = distriRollout
        self.policy_stack = []
        for i in range(self.sensor_num):
            self.policy_stack.append([])
        
        self.penalty = penalty
        self.lite = lite
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
        # if self.optmethod == "discrete":
            # # discrete action space, a square grid
            # v_grid = [2.5]
            # theta_grid = np.linspace(0, 3, num=4) * np.pi / 2
            # action_space = []
            # # 1. single step action space A
            # for v in v_grid:
            #     for theta in theta_grid:
            #         action_space.append([v, theta])
            
            # # 2. over horizon A^H

            # action_space_h = [action_space] * self.opt_step
            # self.action_space_h = list(itertools.product(*action_space_h))
            
            # # 3. if this is multi-agent
            # if not self.distriRollout:
            #     action_space_h_team = [self.action_space_h] * self.sensor_num
            #     self.action_space_h_team = list(itertools.product(*action_space_h_team))
            # self.action_space_size = len(action_space)
            # # print("action size %s" % self.action_space_size)
            # if self.distriRollout:
            #     self.action_space_h = action_space
            # else:
            #     self.action_space_h_team = action_space


    def match_traj(self, traj, first_points):
        m = max(len(traj), len(first_points))

        # build a m,m fully connected bipartite graph
        edge_matrix = np.full((m, m), 10000)
        # fill the metrics value inside bipartite graph
        #                   traj
        # first_points [ matrix ]
        for i in range(len(traj)):
            x = traj[i]
            for j in range(len(first_points)):
                y = first_points[j]
                edge_matrix[j, i] = self.euclidan_dist(x, y)
                
        # after preparing the bipartite graph matrix, implement Hungarian alg
        
        row_ind, col_ind = linear_sum_assignment(edge_matrix)
        
        return row_ind, col_ind
    
    def cheat_rollout(self, u, traj):
        '''
        verified, feed ground truth to the planner. 
        Potential bug: it breaks when the track number is larger than real tarjet number
        '''
        MCSample = {}
        
        
        first_points = []
        for i in range(len(self.local_track["id"])):
            pt = self.local_track["infos"][i][0].x[0:2]
            first_points.append(pt)
        row_ind, col_ind = self.match_traj(traj[0], first_points)

        for i in range(len(self.local_track["id"])):
            index = col_ind[i]
            
            trajectory = []
            
            
            for k in range(self.SampleNum):
                # sample one place
                
                z = traj[k][index]
                trajectory.append(z)
                
            MCSample[self.local_track["id"][i]] = trajectory
        
        self.samples = MCSample
        
        trajectory = []
        for id_ in self.local_track["id"]:
            traj_i = self.samples[id_]
            trajectory.append(traj_i)
        
        self.trajInTime = []
        for j in range(self.SampleNum):
            self.trajInTime.append(self.column(trajectory, j))
        
        return self.standardNBO(u)

    def rollout(self) -> list:
        self.sampling()
        if self.parallel:
            return self.parallelNBO()
        else:
            return self.standardNBO()
    
    def parallelNBO(self):
        '''Parallel decision making in NBO, the distributed case is PMA, and centralized case is dec-POMDP'''

        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            
            return [[0, 0]] * self.opt_step
                    
        # PMA, agent use other agents previous decision
        self.u = self.policy_stack

        if self.distriRollout:
            if self.optmethod == 'de':
                raise RuntimeError('DE in PMA-NBO method not yet undefined!')

            elif self.optmethod == 'pso':
                # normal method for PMA-NBO
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) # , init_pos = init_pos)

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, dimensions=2*self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) #, init_pos = init_pos)

                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                self.opt_value = cost
                # cost, pos = optimizer.optimize(self.fobj_pso, n_processes=1, iters=5000, verbose=False)

                # print("t = %s, value = %s"%(self.t, cost))
                # plot_cost_history(cost_history=optimizer.cost_history)
                # plt.savefig('pso_time=%s.png'%self.t)
                # plt.close()
                action_vector = []
                
                for i in range(self.opt_step):
                    
                    x = [pos[2* i], pos[2* i+ 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    action_vector.append(v)
                # update its action
                self.v = action_vector[0]

            elif self.optmethod == 'discrete':
                raise RuntimeError('discrete in PMA-NBO method not yet undefined!')
                # a multiprocessing way
                self.records = multiprocessing.Manager().dict()
                p = Pool(psutil.cpu_count()-1)   # here it will utilize the maximum number of CPU
                input_index = list(range(self.action_space_size))
                p.map(self.fdis, input_index)
                index = min(self.records, key=self.records.get)
                p.close()
                # pick the lowest cost and return
                u = self.action_space_h[index]
                self.opt_value = self.records[index]
                # generate output
                action_vector = []
                for ids in range(self.sensor_num):
                    v_list = []
                    if ids == self.id:
                        for i in range(self.opt_step):
                            
                            x = [u[i][0], u[i][1]]
                            v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                            v_list.append(v)
                        # update its action
                        self.u[self.id] = v_list
                        self.v = self.u[self.id][0]
                    else:
                        v_list = self.u[ids]
                    action_vector.append(v_list)
            else:
                raise RuntimeError('optimization method undefined!')
        else:
            # method of dec-POMDP
            if self.optmethod == 'de':
                raise RuntimeError("DE is not used in dec-POMDP")

            elif self.optmethod == 'pso':
                #  particle swarm optimization
                # https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html#Basic-Optimization-with-Arguments
                # source code
                # https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/single/global_best.html
                # setup hyperparametersa
                # strategies https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/backend/handlers.html#BoundaryHandler

                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step * self.sensor_num)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step * self.sensor_num)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2 * self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    # init_pos = self.central_heuristic()
                    # init_pos = np.maximum(np.minimum(init_pos, self.v_bar), -self.v_bar)
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2 * self.sensor_num* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) # , init_pos = init_pos)

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, dimensions=2 * self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) #, init_pos = init_pos)

                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                self.opt_value = cost
                
                action_vector = []
                
                for i in range(self.opt_step):

                    v = pos[2 * (self.id * self.opt_step + i)]
                    theta = pos[2 * (self.id * self.opt_step + i) + 1]
                    action = [v*np.cos(theta), v*np.sin(theta)]
                    # action = [pos[2 * (ids * self.opt_step + i)], pos[2 * (ids * self.opt_step + i) + 1]]
                    action_vector.append(action)
                
                self.v = action_vector[0]
            
            elif self.optmethod == 'discrete':
                # pick the action in the action space with lowest utility
                # cost, index = 1000000000, -1
                # for index_, u in enumerate(self.action_space_h_team):
                    
                #     value_ = self.fdis(u)
                #     if value_ < cost:
                #         index = index_
                
                # # pick the lowest cost and return
                # u = self.action_space_h_team[index]
                # self.opt_value = cost

                # a multiprocessing way
                self.records = multiprocessing.Manager().dict()
                p = Pool(psutil.cpu_count()-1)   # here it will utilize the maximum number of CPU
                input_index = list(range(self.action_space_size))
                p.map(self.fdis, input_index)
                index = min(self.records, key=self.records.get)
                p.close()
                # pick the lowest cost and return
                u = self.action_space_h_team[index]
                self.opt_value = self.records[index]

                # generate output
                action_vector = []
                for ids in range(self.sensor_num):
                    agent_vector = []
                    for i in range(self.opt_step):

                        v = u[ids][i][0]
                        theta = u[ids][i][1]
                        action = [v*np.cos(theta), v*np.sin(theta)]
                        # action = [pos[2 * (ids * self.opt_step + i)], pos[2 * (ids * self.opt_step + i) + 1]]
                        agent_vector.append(action)
                    action_vector.append(copy.deepcopy(agent_vector))

            else:
                raise RuntimeError('optimization method undefined!')

        return action_vector
    
    def Eucildean_to_Spherical(self, u):
        spherical = []
        for ele in u:
            r = np.sqrt(ele[0]**2 + ele[1]**2)
            spherical.append(r)
            if np.isclose(r, 0.0):
                theta = 0.0
            else:
                if np.isclose(ele[0], 0.0):
                    # vx = 0
                    theta = 0.5 * np.pi + (0.5 - 0.5 * np.sign(ele[1])) * np.pi
                else:
                    theta = np.arctan(ele[1] / ele[0]) + (0.5 - 0.5 * np.sign(ele[0])) * np.pi
            spherical.append(theta)
        return spherical

    def standardNBO(self) -> list:
        
        # no track maintained   
        if len(self.local_track["id"]) < 1:
            print("no observation for agent %d"%self.id)
            
            return [[0, 0]] * self.opt_step
                    
        # PMA, agent use other agents previous decision
        self.u = self.policy_stack
        
        if self.distriRollout:
            if self.optmethod == 'de':
                bounds = [(0, self.v_bar), (0, 2 * np.pi)] * self.opt_step
                
                '''
                details about the DE with parallel 
                https://github.com/uqfoundation/mystic/blob/master/mystic/differential_evolution.py
                '''
                pool = ProcessPool(nodes=psutil.cpu_count()-1)
                # stepmon = VerboseMonitor(interval=10, xinterval=10)
                # result = diffev2(self.fobj, bounds, bounds = bounds, disp = True,itermon = stepmon,\
                #     npop=20, map=pool.map, ftol=self.ftol, gtol=self.gtol, scale=.7, strategy=Rand1Bin)
                # result = diffev2(self.f, bounds, npop=15, map=pool.map, disp = 0, itermon = stepmon)
                # print("test")
                # testv = [0.1, 0.1] * self.SampleNum
                # self.fobj(testv)
                # result = [0.1, 0.1] * self.SampleNum
                # print("successful")
                # print(self.u)

                # TODO increase npop, initial guess input, termination condition
                # if len(self.u[self.id]) > 0:
                #     spherical = self.Eucildean_to_Spherical(self.u[self.id])

                #     init_guess = spherical + spherical[-2:]
                
                #     assert len(init_guess) == 2 * self.SampleNum, "alert: " + str(self.u)
                # else:
                #     init_guess = bounds
                
                result = diffev2(self.fobj_de, bounds, bounds = bounds, disp = False, full_output = 1, \
                    npop=40 * self.opt_step, map=pool.map, ftol=self.ftol, gtol= self.gtol, scale=.7, strategy=Rand1Bin)
                    # npop=1, map=pool.map, ftol=self.ftol, gtol=self.gtol, scale=.8, strategy=Rand1Bin)
                
                if result[-1] == 1:
                    print("Warning: Maximum number of function evaluations has "\
                        "been exceeded.")
                elif result[-1] == 2:
                    print("Warning: Maximum number of iterations has been exceeded")
                v_list = []
                for i in range(len(result[0])/2):
                    x = [result[0][2*i], result[0][2*i + 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    v_list.append(v)
                self.u[self.id] = v_list
                
                self.v = self.u[self.id][0]  
                # print("opt ends===============================================")
                return copy.deepcopy(self.u)
            elif self.optmethod == 'pso':
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    # init_pos = self.central_heuristic()
                    # init_pos = np.maximum(np.minimum(init_pos, self.v_bar), -self.v_bar)
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) # , init_pos = init_pos)

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, dimensions=2*self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) #, init_pos = init_pos)

                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                self.opt_value = cost
                # cost, pos = optimizer.optimize(self.fobj_pso, n_processes=1, iters=5000, verbose=False)

                # print("t = %s, value = %s"%(self.t, cost))
                
                
                # debug_video.pso_propagate(optimizer.best_pos_history, None, self.trajInTime, self.OccupancyMap,
                #      self.sensor_para_list[self.id]["position"], self.horizon, optimizer.cost_history, self.t, self.id,
                #      self.OccupancyMap)
                # # pprint(dict(optimizer.ToHistory._asdict()))
                # plot_cost_history(cost_history=optimizer.cost_history)
                # plt.savefig('pics/test_pso/pso_time=%s_agent%s.png'% (self.t, self.id))
                # plt.close()
                action_vector = []
            
                for i in range(self.opt_step):
                    
                    x = [pos[2* i], pos[2* i+ 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    # vx, vy = u_i[ 2*(self.opt_step*ids + i)], u_i[ 2*(self.opt_step*ids + i) + 1]
                    # if vx**2 + vy**2 > 25:
                    #     return 1000
                    # v = [vx, vy]
                    action_vector.append(v)
                # update its action
                self.u[self.id] = action_vector
                self.v = self.u[self.id][0]
                
            elif self.optmethod == 'discrete':
                # TODO correct to fit new style 
                # pick the action in the action space with lowest utility
                # cost, index = 1000000000, -1
                
                # for index_, u in enumerate(self.action_space_h):
                #     value_ = self.fdis(u)
                #     if value_ < cost:
                #         index = index_
                #         cost = value_
                
                # a multiprocessing way
                self.records = multiprocessing.Manager().dict()
                p = Pool(psutil.cpu_count()-1)   # here it will utilize the maximum number of CPU
                input_index = list(range(self.action_space_size))
                p.map(self.fdis, input_index)
                index = min(self.records, key=self.records.get)
                p.close()
                # pick the lowest cost and return
                u = self.action_space_h[index]
                self.opt_value = self.records[index]
                # generate output
                action_vector = []
                for ids in range(self.sensor_num):
                    v_list = []
                    if ids == self.id:
                        for i in range(self.opt_step):
                            
                            x = [u[i][0], u[i][1]]
                            v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                            v_list.append(v)
                        # update its action
                        self.u[self.id] = v_list
                        self.v = self.u[self.id][0]
                    else:
                        v_list = self.u[ids]
                    action_vector.append(v_list)
            else:
                raise RuntimeError('optimization method undefined!')
        else:
            # print("t = %s"%self.t)
            if self.optmethod == 'de':
                bounds = [(0, self.v_bar), (0, 2 * np.pi)] * (self.sensor_num * self.opt_step)
                
                pool = ProcessPool(nodes=psutil.cpu_count()-1)                
                result = diffev2(self.fobj_de, x0=bounds, bounds = bounds, disp = False, full_output = 1,\
                    npop= self.npop, map=pool.map, ftol=self.ftol, gtol= self.gtol, scale=.5, strategy=Rand1Bin)
                
                if result[-1] == 1:
                    print("Warning: Maximum number of function evaluations has "\
                        "been exceeded.")
                elif result[-1] == 2:
                    print("Warning: Maximum number of iterations has been exceeded")
                else:
                    pass

                print("result value =%s"%result[1])
                action_vector = []
                for ids in range(self.sensor_num):
                    agent_vector = []
                    for i in range(self.opt_step):

                        v = result[0][2 * (ids * self.opt_step + i)]
                        theta = result[0][2 * ids * self.opt_step + 2*i + 1]
                        action = [v*np.cos(theta), v*np.sin(theta)]
                        # action = [result[0][2 * (ids * self.opt_step + i)], result[0][2 * (ids * self.opt_step + i) + 1]]
                        agent_vector.append(action)
                    action_vector.append(copy.deepcopy(agent_vector))

            elif self.optmethod == 'pso':
                #  particle swarm optimization
                # https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html#Basic-Optimization-with-Arguments
                # source code
                # https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/single/global_best.html
                # setup hyperparametersa
                # strategies https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/backend/handlers.html#BoundaryHandler

                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                min_bound = [0, 0] * (self.opt_step * self.sensor_num)
                max_bound = [self.v_bar, 2 * np.pi] * (self.opt_step * self.sensor_num)
                bounds = (np.array(min_bound), np.array(max_bound))
                optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2 * self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol)
                
                # bound on riemannian
                try:
                    # init_pos = self.central_heuristic()
                    # init_pos = np.maximum(np.minimum(init_pos, self.v_bar), -self.v_bar)
                    optimizer = ps.single.GlobalBestPSO(n_particles= 20 * self.opt_step, dimensions=2 * self.sensor_num* self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) # , init_pos = init_pos)

                except:
                    print("init fails at time %s"%self.t)
                    # Call instance of PSO
                    optimizer = ps.single.GlobalBestPSO(n_particles= self.npop, dimensions=2 * self.sensor_num * self.opt_step, 
                        options=options, ftol=self.ftol, bounds=bounds, ftol_iter = self.gtol) #, init_pos = init_pos)

                # Perform optimization
                cost, pos = optimizer.optimize(self.fobj_pso, n_processes=psutil.cpu_count()-1, iters=5000, verbose=False)
                self.opt_value = cost
                
                action_vector = []
                for ids in range(self.sensor_num):
                    agent_vector = []
                    for i in range(self.opt_step):

                        v = pos[2 * (ids * self.opt_step + i)]
                        theta = pos[2 * (ids * self.opt_step + i) + 1]
                        action = [v*np.cos(theta), v*np.sin(theta)]
                        # action = [pos[2 * (ids * self.opt_step + i)], pos[2 * (ids * self.opt_step + i) + 1]]
                        agent_vector.append(action)
                    action_vector.append(copy.deepcopy(agent_vector))
            
            elif self.optmethod == 'discrete':
                # pick the action in the action space with lowest utility
                # cost, index = 1000000000, -1
                # for index_, u in enumerate(self.action_space_h_team):
                    
                #     value_ = self.fdis(u)
                #     if value_ < cost:
                #         index = index_
                
                # # pick the lowest cost and return
                # u = self.action_space_h_team[index]
                # self.opt_value = cost

                # a multiprocessing way
                self.records = multiprocessing.Manager().dict()
                p = Pool(psutil.cpu_count()-1)   # here it will utilize the maximum number of CPU
                input_index = list(range(self.action_space_size))
                p.map(self.fdis, input_index)
                index = min(self.records, key=self.records.get)
                p.close()
                # pick the lowest cost and return
                u = self.action_space_h_team[index]
                self.opt_value = self.records[index]

                # generate output
                action_vector = []
                for ids in range(self.sensor_num):
                    agent_vector = []
                    for i in range(self.opt_step):

                        v = u[ids][i][0]
                        theta = u[ids][i][1]
                        action = [v*np.cos(theta), v*np.sin(theta)]
                        # action = [pos[2 * (ids * self.opt_step + i)], pos[2 * (ids * self.opt_step + i) + 1]]
                        agent_vector.append(action)
                    action_vector.append(copy.deepcopy(agent_vector))

            else:
                raise RuntimeError('optimization method undefined!')

        return action_vector
    
    def fobj_de(self, u_i):
        
        if self.distriRollout:
            u = copy.deepcopy(self.u)
            v_list = []
            for i in range(self.SampleNum):
                x = [u_i[2*i], u_i[2*i + 1]]    # why we need times 5 ????????, found Mar 1
                v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                v_list.append(v)
            u[self.id] = v_list
            
        else:
            u = []
            
            for ids in range(self.sensor_num):
                v_list = []
                for i in range(self.opt_step):
                    x = [u_i[2*(self.opt_step*ids + i)], u_i[2* (self.opt_step*ids + i) + 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    # vx, vy = u_i[2*(self.opt_step*ids + i)], u_i[2*(self.opt_step*ids + i) + 1]
                    # if vx**2 + vy**2 > 25:
                    #     return 1000
                    # v = [vx, vy]
                    
                    v_list.append(v)
                    
                u.append(copy.deepcopy(v_list))
            
            # for i in range(len(self.sensor_para_list)):
            #     self.u[i] = [u_i[2*i] * np.cos(u_i[2*i+1]), u_i[2*i] * np.sin(u_i[2*i+1])]
        
        return self.f(u)
    
    def fobj_pso(self, u_i):
        length = len(u_i)

        cost_list = []
        
        for seed in range(length):
            
            cost_list.append(self.fpso(u_i[seed, :]))
              
        return np.array(cost_list)

    def fpso(self, u_i: list) -> float:
        '''here u is an array'''
        if self.distriRollout:
            
            u = []
            # print(u_i)
            for ids in range(self.sensor_num):
                v_list = []
                if ids == self.id:
                    for i in range(self.opt_step):
                        
                        x = [u_i[2*i], u_i[2* i + 1]]
                        v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                        # vx, vy = u_i[ 2*(self.opt_step*ids + i)], u_i[ 2*(self.opt_step*ids + i) + 1]
                        # if vx**2 + vy**2 > 25:
                        #     return 1000
                        # v = [vx, vy]
                        v_list.append(v)
                else:
                    v_list = self.u[ids]
                u.append(v_list)
        else:
            u = []
            # print(u_i)
            for ids in range(self.sensor_num):
                v_list = []
                for i in range(self.opt_step):
                    
                    x = [u_i[2*(self.opt_step*ids + i)], u_i[2* (self.opt_step*ids + i) + 1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    # vx, vy = u_i[ 2*(self.opt_step*ids + i)], u_i[ 2*(self.opt_step*ids + i) + 1]
                    # if vx**2 + vy**2 > 25:
                    #     return 1000
                    # v = [vx, vy]
                    v_list.append(v)
                    
                u.append(v_list)
        return self.f(u)
    
    def fdis(self, index):
        '''here u is an array'''
        if self.distriRollout:
            u_i = self.action_space_h[index]           
            u = []
            # print(u_i)
            for ids in range(self.sensor_num):
                v_list = []
                if ids == self.id:
                    for i in range(self.opt_step):
                        
                        x = [u_i[i][0], u_i[i][1]]
                        v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                        v_list.append(v)
                else:
                    v_list = self.u[ids]
                u.append(v_list)
        else:
            u_i = self.action_space_h_team[index]
            u = []
            for ids in range(self.sensor_num):
                v_list = []
                for i in range(self.opt_step):
                    
                    x = [u_i[ids][i][0], u_i[ids][i][1]]
                    v = [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
                    v_list.append(v)
                    
                u.append(copy.deepcopy(v_list))
        self.records[index] = self.f(u)
        return

    def f(self, u: list) -> float:
        if self.info_gain:
            value = self.rolloutTrailSim_centralized_P(u)
        elif self.sigma:
            value = self.rolloutTrailSim_sigma(u)
        elif self.central_kf:
            value = self.rolloutTrailSim_centralized(u)
        else:
            value = self.rolloutTrailSim_no_jump(u)
        # minimal value
        value /= (len(self.local_track["id"]))
        # print("value of %s = %s"%(self.u[self.id][0], value))
        
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
        
        
    def rolloutTrailSim_no_jump(self, u):
        '''
        Rollout policy for muti-agent decision, in the information-driven way given MonteCarlo scenarios
        s: sensor states (position [x, y, theta]);
        u: the control vector for all agents, u = [u1, u2, ..., um] with total m agents
        each agent has ui = [ui1, ..., uiself.SampleNum] vector of control
        and the task of rollout is to generate u_id, id is the id of the agent
        output: the objective function of accumulated tr(sigma) for all tracks, all times
        '''
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        id_list = []
        neighbor_num = len(self.neighbor["id"])
        
        for i in range(self.sensor_num):
            if i in self.neighbor["id"] or i==self.id:
                id_list.append(i)
                ego = dec_agent(copy.deepcopy(self.sensor_para_list), i, self.SampleDt, self.cdt, 
                            L0 = self.L, isObsdyn__ = self.tracker.isObsdyn, 
                            isRotate = self.isRotate, isVirtual = True, SemanticMap=self.SemanticMap)
                
                # skip data association, only have KF object
                for j in range(len(self.local_track["id"])):
                    info = self.local_track["infos"][j][0]
                    x0 = np.matrix(info.x).reshape(4,1)
                    P0 = np.matrix(info.P).reshape(4,4)
                    if self.isObsdyn:
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]), 
                        self.sensor_para_list[i]["r0"], quality=self.sensor_para_list[i]["quality"])
                        kf.init = False
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]))
                        
                        kf.init = False
                    ego.tracker.append(kf)
                try:
                    ego.v = copy.deepcopy(u[i][0])
                except:
                    ego.base_policy()
                
                ego.dynamics(self.SampleDt)
                agent_list.append(ego)

        egoIndex = id_list.index(self.id)
        # 2. simulate the Monte Carlo 
        # broadcast information and recognize neighbors, we don't do it in every
        # iteration since we assume they does not lose connection
        pub_basic = []
        for agent in agent_list:
            basic_msg = Agent_basic()
            basic_msg.id = agent.id
            basic_msg.x = agent.sensor_para["position"][0]
            basic_msg.y = agent.sensor_para["position"][1]
            pub_basic.append(basic_msg)
        
        P_list = []
        P_list_i = []
        
        for kf in agent_list[egoIndex].tracker:
            P_list_i.append(np.trace(kf.P_k_k[0:2, 0:2]))     
            
        P_list.append(P_list_i)
        
        # make each agent realize their neighbors
        for agent in agent_list:
            agent.basic_info(pub_basic)

        for i in range(self.SampleNum):
            info_list = []
            P_list_i = []
            t = self.t + (i + 1) * self.SampleDt
            
            z_k = self.trajInTime[i]
            semantic_zk = []
            for z in z_k:
                semantic_zk.append(self.InsideObstacle([z[0], z[1]]))
            
            # feed info to sensors
            for agent in agent_list:
                info_0 = agent.sim_detection_callback(copy.deepcopy(z_k), t, semantic=semantic_zk)
                info_list.append(info_0)
            

            if self.lite:
                # alternative way, a min_consensus for tracks with lowest uncertianty tr(O[0:2, 0:2])
                for j in range(len(self.local_track["id"])):
                    p =1e5
                    index = 0
                    for r in range(neighbor_num):
                        track = info_list[r].tracks[j]
                        PMatrix = np.matrix(track.P).reshape(4, 4)[0:2, 0:2]
                        if p > np.trace(PMatrix):
                            index = r
                            p = np.trace(PMatrix)
                    
                    # then update the track for all agents
                    for r in range(neighbor_num):
                        agent_list[r].tracker[j].x_k_k = copy.deepcopy(agent_list[index].tracker[j].x_k_k)
                        agent_list[r].tracker[j].P_k_k = copy.deepcopy(agent_list[index].tracker[j].P_k_k)
                        agent_list[r].tracker[j].x_k_k_min = copy.deepcopy(agent_list[index].tracker[j].x_k_k_min)
                        agent_list[r].tracker[j].P_k_k_min = copy.deepcopy(agent_list[index].tracker[j].P_k_k_min)
                    # after fixed num of average consensus, accumulate self.id's agent info value
                
            else:
                # consensus starts 
                for l in range(self.L):
                    # receive all infos
                    for agent in agent_list:
                        agent.grab_info_list(info_list)
                    
                    info_list = []
                    # then do consensus for each agent, generate new info
                    for agent in agent_list:
                        info_list.append(agent.consensus())

                    
            for l in range(len(info_list[egoIndex].tracks)):
                # SigmaMatrix = np.matrix(track.Sigma).reshape(4, 4)
                track = info_list[egoIndex].tracks[l]
                PMatrix = np.matrix(track.P).reshape(4, 4)[0:2, 0:2]
                # pose = Point(track.x[0], track.x[1])
                # weight = 1 - 0.9 * self.InsideObstacle(pose)
                trace = np.trace(PMatrix)
                P_list_i.append(trace)
                
                objValue += self.interpolation(P_list[-1][l],trace) # (self.gamma ** i) # * weight
            P_list.append(P_list_i)
            
            # 4. agent movement base policy
            # if t - tc >= self.cdt:
            if t-tc >= self.cdt and (i <= self.SampleNum-2):
                
                tc = t
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    try:
                       
                        agent.v = u[j][i+1]
                    except:
                        
                        agent.base_policy()
                    # agent.v = self.u[j][i+1]
           
                    agent.dynamics(self.SampleDt)
        
        return objValue

    def interpolation(self, p_, p):
        # if p > self.tracker.common_P:
        #     p = 10000
        # if p_ > self.tracker.common_P:
        #     p_ = 10000
        
        dp = (p - p_)/self.factor
        results = 0.0
        
        for i in range(1, self.factor+1):
            p_t = p_ + i * dp
            # if p_t > self.tracker.common_P:
            #     p_t += self.penalty
            #     p_t = 10000
            results += p_t
            
        return results

    def rolloutTrailSim(self, trajInTime):
        '''
        Rollout policy for muti-agent decision, in the information-driven way given MonteCarlo scenarios
        s: sensor states (position [x, y, theta]);
        u: the control vector for all agents, u = [u1, u2, ..., um] with total m agents
        each agent has ui = [ui1, ..., uiself.SampleNum] vector of control
        and the task of rollout is to generate u_id, id is the id of the agent
        output: the objective function of accumulated tr(sigma) for all tracks, all times
        '''
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        id_list = []
        
        for i in range(len(self.sensor_para_list)):
            if i in self.neighbor["id"] or i==self.id:
                id_list.append(i)
                ego = dec_agent(copy.deepcopy(self.sensor_para_list[i]), i, self.SampleDt, self.cdt, 
                            L0 = self.L, isObsdyn__ = self.tracker.isObsdyn, 
                            isRotate = self.isRotate, isVirtual = True, SemanticMap=self.SemanticMap)
                
                # skip data association, only have KF object
                for j in range(len(self.local_track["id"])):
                    info = self.local_track["infos"][j][0]
                    x0 = np.matrix(info.x).reshape(4,1)
                    P0 = np.matrix(info.P).reshape(4,4)
                    if self.isObsdyn:
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.SampleQ, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]), 
                        self.sensor_para_list[i]["r0"], quality=self.sensor_para_list[i]["quality"])
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.SampleQ, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]), 
                        )
                    ego.tracker.append(kf)
                try:
                    ego.v = copy.deepcopy(self.u[i][0])
                except:
                    ego.base_policy()
                
                ego.dynamics(self.SampleDt)
                agent_list.append(ego)

        egoIndex = id_list.index(self.id)
        # 2. simulate the Monte Carlo 
        # broadcast information and recognize neighbors, we don't do it in every
        # iteration since we assume they does not lose connection
        pub_basic = []
        for agent in agent_list:
            basic_msg = Agent_basic()
            basic_msg.id = agent.id
            basic_msg.x = agent.sensor_para["position"][0]
            basic_msg.y = agent.sensor_para["position"][1]
            pub_basic.append(basic_msg)
        
        
        # make each agent realize their neighbors
        for agent in agent_list:
            agent.basic_info(pub_basic)

        for i in range(self.SampleNum):
            info_list = []

            t = self.t + (i + 1) * self.SampleDt
            
            z_k = trajInTime[i]
            semantic_zk = []
            for z in z_k:
                
                semantic_zk.append(self.InsideObstacle([z[0], z[1]]))
            
            # feed info to sensors
            for agent in agent_list:
                info_0 = agent.sim_detection_callback(copy.deepcopy(z_k), t, semantic=semantic_zk)
                info_list.append(info_0)
            
            # consensus starts  TODO: validate this part: consensus gives consistent value
            for l in range(self.L):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())
            # after fixed num of average consensus, accumulate self.id's agent info value
            
            # print("consensus compare tracks")
            # x, p = [], []
            # for info in info_list:
            #     agentx = []
            #     for track in info.tracks:
            #         agentx.append([track.x[0], track.x[1], np.trace(np.matrix(track.P).reshape(4, 4))])
            #     x.append(agentx)
            # print(x)

            for track in info_list[egoIndex].tracks:
                # SigmaMatrix = np.matrix(track.Sigma).reshape(4, 4)
                PMatrix = np.matrix(track.P).reshape(4, 4)

                # pose = Point(track.x[0], track.x[1])
                # weight = 1 - 0.9 * self.InsideObstacle(pose)
                objValue += np.trace(PMatrix) # (self.gamma ** i) # * weight

            # 4. agent movement base policy
            # if t - tc >= self.cdt:
            if t-tc >= self.cdt and (i <= self.SampleNum-2):
                
                tc = t
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    try:
                       
                        agent.v = copy.deepcopy(self.u[j][i+1])
                    except:
                        
                        agent.base_policy()
                    # agent.v = self.u[j][i+1]
                    agent.dynamics(self.cdt)
            
        return objValue
    
    def rolloutTrailSim_sigma(self, u):
        '''
        Rollout policy for muti-agent decision, in the information-driven way given MonteCarlo scenarios
        s: sensor states (position [x, y, theta]);
        u: the control vector for all agents, u = [u1, u2, ..., um] with total m agents
        each agent has ui = [ui1, ..., uiself.SampleNum] vector of control
        and the task of rollout is to generate u_id, id is the id of the agent
        output: the objective function of accumulated tr(sigma) for all tracks, all times
        '''
        '''
        Rollout policy for muti-agent decision, in the information-driven way given MonteCarlo scenarios
        s: sensor states (position [x, y, theta]);
        u: the control vector for all agents, u = [u1, u2, ..., um] with total m agents
        each agent has ui = [ui1, ..., uiself.SampleNum] vector of control
        and the task of rollout is to generate u_id, id is the id of the agent
        output: the objective function of accumulated tr(sigma) for all tracks, all times
        '''
        objValue = 0.0
        tc = copy.deepcopy(self.t)
        # 1. initiate all sensors
        agent_list = []
        id_list = []
        
        for i in range(self.sensor_num):
            if i in self.neighbor["id"] or i==self.id:
                id_list.append(i)
                ego = dec_agent(copy.deepcopy(self.sensor_para_list), i, self.SampleDt, self.cdt, 
                            L0 = self.L, isObsdyn__ = self.tracker.isObsdyn, sigma=self.sigma,
                            isRotate = self.isRotate, isVirtual = True, SemanticMap=self.SemanticMap)
                
                # skip data association, only have KF object
                for j in range(len(self.local_track["id"])):
                    info = self.local_track["infos"][j][0]
                    x0 = np.matrix(info.x).reshape(4,1)
                    P0 = np.matrix(info.P).reshape(4,4)
                    if self.isObsdyn:
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.tracker.Q, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]), 
                        self.sensor_para_list[i]["r0"], quality=self.sensor_para_list[i]["quality"])
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.tracker.Q, 
                        self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], self.sensor_para_list[i]["r"]]), 
                        )
                    ego.tracker.append(kf)
                try:
                    ego.v = copy.deepcopy(self.u[i][0])
                except:
                    ego.base_policy()
                
                ego.dynamics(self.SampleDt)
                agent_list.append(ego)

        egoIndex = id_list.index(self.id)
        # 2. simulate the Monte Carlo 
        # broadcast information and recognize neighbors, we don't do it in every
        # iteration since we assume they does not lose connection
        pub_basic = []
        for agent in agent_list:
            basic_msg = Agent_basic()
            basic_msg.id = agent.id
            basic_msg.x = agent.sensor_para["position"][0]
            basic_msg.y = agent.sensor_para["position"][1]
            pub_basic.append(basic_msg)
        
        P_list = []
        P_list_i = []
        
        for kf in agent_list[egoIndex].tracker:
            P_list_i.append(np.trace(kf.P_k_k[0:2, 0:2]))     
            
        P_list.append(P_list_i)
        
        # make each agent realize their neighbors
        for agent in agent_list:
            agent.basic_info(pub_basic)

        for i in range(self.SampleNum):
            info_list = []
            P_list_i = []
            t = self.t + (i + 1) * self.SampleDt
            z_k = self.trajInTime[i]
            
            # feed info to sensors
            for agent in agent_list:
                info_0 = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
            
            # consensus starts  TODO: validate this part: consensus gives consistent value
            for l in range(self.L):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())
            # after fixed num of average consensus, accumulate self.id's agent info value
                    
            for l in range(len(info_list[egoIndex].tracks)):
                # SigmaMatrix = np.matrix(track.Sigma).reshape(4, 4)
                track = info_list[egoIndex].tracks[l]
                PMatrix = np.matrix(track.P).reshape(4, 4)[0:2, 0:2]
                # pose = Point(track.x[0], track.x[1])
                # weight = 1 - 0.9 * self.InsideObstacle(pose)
                trace = np.trace(PMatrix)
                
                P_list_i.append(trace)
                
                objValue += self.interpolation(P_list[-1][l],trace) # (self.gamma ** i) # * weight
            P_list.append(P_list_i)
            
            # 4. agent movement base policy
            # if t - tc >= self.cdt:
            if t-tc >= self.cdt and (i <= self.SampleNum-2):
                
                tc = t
                for j in range(len(agent_list)):
                    agent = agent_list[j]
                    try:
                       
                        agent.v = u[j][i+1]
                    except:
                        
                        agent.base_policy()
                    # agent.v = self.u[j][i+1]
                    agent.dynamics(self.SampleDt)
        
        return objValue

    def rolloutTrailSim_centralized(self, u):
        '''
        sequential centralized kf update
        '''
        # if len(self.policy_stack[-1]) > 0:
        #     wehavepolicy = True
        # else:
        #     wehavepolicy = False
        
        # rate = int(self.cdt / self.dt)
        # print('cdt = %s, dt = %s, self.samplnum = %s' %(self.cdt, self.dt, self.SampleNum))
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
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]), self.sensor_para_list[i]["r0"], 
                            quality=self.sensor_para_list[i]["quality"])
                        
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
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
                if self.InsideObstacle([z[0], z[1]]):
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
                        # if np.isclose(1, t - self.t) and wehavepolicy:
                        #     raise RuntimeError("agent using base policy")
                        # print("agent %s use base policy policy!" % j)
            
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        
        # multiple weighted trace penalty (MWTP) by [1]
        if self.wtp:
            # wtp = 0.0
            # Dj = [0] * neighbor_num

            # for j in range(len(self.trajInTime[-1])):
            #     z = self.trajInTime[-1][j]            
                
            #     if self.InsideObstacle([z[0], z[1]]):
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
            for j in range(len(self.trajInTime[-1])):
                z = self.trajInTime[-1][j]            
                
                if self.InsideObstacle([z[0], z[1]]):
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

    def rolloutTrailSim_centralized_P(self, u):
        '''
        sequential centralized kf update
        '''
        # if len(self.policy_stack[-1]) > 0:
        #     wehavepolicy = True
        # else:
        #     wehavepolicy = False
        
        # rate = int(self.cdt / self.dt)
        # print('cdt = %s, dt = %s, self.samplnum = %s' %(self.cdt, self.dt, self.SampleNum))
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
                        kf = util.dynamic_kf(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                            self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                            self.sensor_para_list[i]["r"]]), self.sensor_para_list[i]["r0"], 
                            quality=self.sensor_para_list[i]["quality"])
                        
                    else:
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
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
                if not self.InsideObstacle([z[0], z[1]]):
                    # do sequential update, use P = [P^-1 + \sum_i H^T R^-1_i H] ^ -1
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
                        # if np.isclose(1, t - self.t) and wehavepolicy:
                        #     raise RuntimeError("agent using base policy")
                        # print("agent %s use base policy policy!" % j)
            
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
                
                if self.InsideObstacle([z[0], z[1]]):
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
        # if len(self.policy_stack[-1]) > 0:
        #     wehavepolicy = True
        # else:
        #     wehavepolicy = False
        
        # rate = int(self.cdt / self.dt)
        # print('cdt = %s, dt = %s, self.samplnum = %s' %(self.cdt, self.dt, self.SampleNum))
        
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
                
                ego = dec_agent(copy.deepcopy(self.sensor_para_list), i, self.SampleDt, self.cdt, 
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
                        kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
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
                if self.InsideObstacle([z[0], z[1]]):
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
                        # if np.isclose(1, t - self.t) and wehavepolicy:
                        #     raise RuntimeError("agent using base policy")
                        # print("agent %s use base policy policy!" % j)
            
            for j in range(len(agent_list)):
                
                agent = agent_list[j]
                agent.dynamics(self.SampleDt)
        
        # final_trace = 0.0
        # for j in range(len(z_k)):
        #     trace = np.trace(agent_list[0].tracker[j].P_k_k[0:2, 0:2])       
            # Sigma_j = np.linalg.inv(agent_list[0].tracker[j].P_k_k)
            # trace = np.trace(Sigma_j[0:2, 0:2])
            # final_trace += trace
        # multiple weighted trace penalty (MWTP) by [1]
        if self.wtp:
            
            # revised wtp, which is complete since we have ordered the traces
            wtp = 0.0
            Dj = [0] * neighbor_num
            trace_list_ordered = []
            for j in range(len(self.trajInTime[-1])):
                z = self.trajInTime[-1][j]            
                
                if self.InsideObstacle([z[0], z[1]]):
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

        objValue = posterior_trace - prior_trace

        return objValue


    def central_heuristic(self):
        vec_len = 2 * self.opt_step * self.sensor_num
        sen_list, target_list = list(range(self.sensor_num)), list(range(len(self.local_track["id"])))
        z_final = self.trajInTime[-1]
        var = 0.5
        if self.sensor_num > len(self.local_track["id"]):
            # more sensors to chase targets
            # make permutations of sensors to chase targets, and rest sensors to track the nearest target
            per = list(itertools.permutations(sen_list, len(self.local_track["id"])))
            numpermatch, resi = self.npop // len(per), self.npop % len(per)
            if numpermatch <= 1:
                print("the permutation number is %s, please increase npop accordingly"%len(per))
            init_solutions = None
            for p in per:
                # get the overall permutation 
                seed_sol = []
                for i in range(self.sensor_num):
                    ps = self.sensor_para_list[i]["position"][0:2]
                    if i in p:
                        # get action
                        j = p.index(i)
                        pt = z_final[j]
                        seed_sol_i = self.action_generator(ps, pt)
                    else:
                        # find the nearest targets
                        j = 0
                        dist = 100000
                        for l in range(len(z_final)):
                            d = self.euclidan_dist(ps, z_final[l])
                            if dist > d:
                                dist = d
                                j = l 
                        pt = z_final[j]
                        seed_sol_i = self.action_generator(ps, pt)
                    seed_sol += seed_sol_i
                noises = np.random.multivariate_normal(np.zeros(vec_len), np.diag([var] * vec_len), numpermatch - 1)
                noised_sol = seed_sol + noises
                seed_sol_array = np.asarray([seed_sol])
                noised_sol = np.concatenate((noised_sol, seed_sol_array), axis=0)
                if init_solutions is None:
                    init_solutions = noised_sol
                else:
                    init_solutions = np.concatenate((init_solutions, noised_sol), axis=0)


        else:
            # evenly assign sesnors to targets
            # pick sensor to send to targets
            per = list(itertools.permutations(target_list))
            numpermatch, resi = self.npop // len(per), self.npop % len(per)
            if numpermatch <= 1:
                print("the permutation number is %s, please increase npop accordingly"%len(per))
            
            init_solutions = None
            for p in per:
                # get the overall permutation 
                seed_sol = []
                for i in range(self.sensor_num):
                    ps = self.sensor_para_list[i]["position"][0:2]
                    pt = z_final[p[i]]
                    seed_sol_i = self.action_generator(ps, pt)
                    seed_sol += seed_sol_i
                noises = np.random.multivariate_normal(np.zeros(vec_len), np.diag([var] * vec_len), numpermatch - 1)
                noised_sol = seed_sol + noises
                seed_sol_array = np.asarray([seed_sol])
                noised_sol = np.concatenate((noised_sol, seed_sol_array), axis=0)
                if init_solutions is None:
                    init_solutions = noised_sol
                else:
                    init_solutions = np.concatenate((init_solutions, noised_sol), axis=0)

        if resi > 0:
            
            pure_rand_solutions = np.random.rand(resi, vec_len) * (self.v_bar - .5)
            return np.concatenate((init_solutions, pure_rand_solutions), axis=0)
        else:
            return init_solutions

    def action_generator(self, ps, pt):
        dx = pt[0] - ps[0]
        dy = pt[1] - ps[1]
        vx = dx / self.SampleDt
        vy = dy / self.SampleDt
        v_ = np.sqrt(vx**2 + vy**2)
        if np.isclose(v_, 0):
            return [0, 0] * self.opt_step
        else:
            return [min(self.v_bar, v_) * vx / v_, min(self.v_bar, v_) * vy / v_] * self.opt_step

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
    
    