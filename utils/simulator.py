import numpy as np
from utils.msg import Agent_basic, Single_track, Info_sense, Planning_msg
from utils.MonteCarloRollout import MCRollout
from utils.nbo import nbo_agent
from utils.ibr import ibr_agent
from utils.occupancy import occupancy
from utils.nbo_discrete import nbo_agent_discrete
import utils.util as util
import random
import copy
import json
import csv
import os, sys
import time
from pathlib import Path
import itertools

class simulator:

    @staticmethod
    def filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, deci_Schema, domain,
            lambda0 = 1e-5, r=5, MCSnum= 50, traj_type = 'normal', info_gain = False, useSemantic = True, 
            dropout_pattern = None, dropout_prob = 0.0, dv = 1e0):
        print(optmethod, deci_Schema)
        if optmethod == 'pso':
            filename = deci_Schema + '_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                    optmethod + '_seq_' + str(seq)
        elif optmethod == 'discrete':
            filename = deci_Schema + '_horizon_'+ str(horizon)+ '_' + optmethod + '_' + str(dv)
        else:
            raise RuntimeError("optmethod %s not defined!" % optmethod)
        
        if domain == 'MonteCarloRollout':
            filename += '_MC_' + str(MCSnum)
        
        if central_kf:
            filename += '_ckf'
        
        if wtp:
            filename += '_wtp'
        
        if env == 'parksim':
            filename += '_parksim'
        elif env == 'poisson':
            filename += '_poisson_' + str(r) + '_' + str(lambda0) + '_'
        elif env == 'simple1':
            filename += '_simple1'
        elif env == 'simple2':
            filename += '_simple2'
        else:
            raise RuntimeError("Unknown scenario! %s" %env)

        if not useSemantic:
            filename += 'noSemantic_'
        
        if traj_type != 'normal':
            filename += traj_type
        
        if info_gain:
            filename += info_gain   # 'info_gain', 'info_acc', 'log_det', 'cov_gain'
        
        if dropout_pattern != None:
            filename += dropout_pattern + '_p=' + str(dropout_prob)

        print(filename)
        return filename

    @staticmethod
    def init_states(traj, dt):
        traj_num = len(traj[0])
        track_init = []
        for i in range(traj_num):
            x, y = traj[0][i][0], traj[1][i][0]
            vx = (traj[0][i][1] - x) / dt
            vy = (traj[1][i][1] - y) / dt
            track_init.append([x, y, vx, vy])
        return track_init

    @staticmethod
    def map_traj_doc(env: str, traj_type: str, useSemantic: bool, r: int, lambda0: float, iteration: int):
        path = os.getcwd()
        # load sensor model
        if env == 'simple1':
            filename = os.path.join(path, "data", 'env', "simplepara.json")
        elif env == 'simple2':
            filename = os.path.join(path, 'data', 'env', 'simplepara2.json')
        else:
            filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        
        # load trajectories of targets
        if env == 'simple1':
            filename = os.path.join(path, "data", 'env', "simpletraj.json")
        elif env == 'simple2':
            filename = os.path.join(path, "data", 'env', "simpletraj2.json")
        elif traj_type == 'straight':
            filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        elif traj_type == 'static':
            filename = os.path.join(path, "data", 'env', "traj_static.json")
        else:
            filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        with open(filename) as json_file:
            traj = json.load(json_file)
        
        SemanticMap, CircleMap = None, None
        if useSemantic:
            if env == 'parksim':
                filename = os.path.join(path, 'data', 'env', 'parking_map.json')
                with open(filename) as json_file:
                    SemanticMap = json.load(json_file)
            elif env == 'poisson':
                # load semantic maps
                filename = os.path.join(path, 'data', 'env', 'poisson', 'r_' + str(r) + '_lambda_' + \
                            str(lambda0) + '_' + str(iteration) + '.json')
                
                with open(filename) as json_file:
                    aMap = json.load(json_file)
                CircleMap = {'centers': aMap['centers'], 
                                'r': r}
                        
        return data, traj, SemanticMap, CircleMap

    @staticmethod
    def generate_discrete_action_space(horizon, sensor_num):
        v_grid = [2.5]
        theta_grid = np.linspace(0, 3, num=4) * np.pi / 2
        action_space = []
        # 1. single step action space A
        for v in v_grid:
            for theta in theta_grid:
                action_space.append([v, theta])
        
        # 2. over horizon A^H
        action_space_h = [action_space] * horizon
        action_space_h_done = list(itertools.product(*action_space_h))

        # 3. if this is multi-agent
        if sensor_num == 1:
            # print("dis size is %s" % len(action_space_h_done))
            return action_space_h_done
        
        else:
            action_space_h_team = [action_space_h_done] * sensor_num
            action_space_h_team_done = list(itertools.product(*action_space_h_team))
            # print("cen size is %s" % len(action_space_h_team_done))
            return action_space_h_team_done

    @staticmethod
    def MCRollout_distributed(iteration, horizon, ftol, gtol, MCSnum, wtp, env, info_gain = False, lite=False, 
        central_kf=False, seq=0, optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal',
        repeated = -1):
        start_time = time.time()
        '''SMA-MCR'''
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'dis', 'MonteCarloRollout', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain,
            )
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'MonteCarloRollout', env, data2save)
        
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if not coverfile:
            if repeated > 0:

                filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
            else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
            my_file = Path(filename)
            if my_file.is_file():
                print("%s already exists!" % filename)
                return

        # env parameter and traj parameter
        data, traj = simulator.map_traj_doc(env, traj_type)

        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])
        
        # load semantic maps
        if env == 'parksim':
            filename = os.path.join(path, 'data', 'env', 'parking_map.json')
            with open(filename) as json_file:
                SemanticMap = json.load(json_file)
        elif env == 'poisson':
            # load semantic maps
            filename = os.path.join(path, 'data', 'env', 'poisson', 'r_' + str(r) + '_lambda_' + \
                        str(lambda0) + '_' + str(iteration) + '.json')
            with open(filename) as json_file:
                aMap = json.load(json_file)
            CircleMap = {'centers': aMap['centers'], 
                            'r': r}
        elif env in ['simple1', 'simple2']:
            SemanticMap = None

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        agent_list = []

        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                ego = MCRollout(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=False, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, opt_step=1, MCSnum = MCSnum,
                    wtp=wtp, info_gain = info_gain, IsStatic = (traj_type == 'static'))
                
            elif env == 'poisson':
                
                ego = MCRollout(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=False, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, opt_step=1, MCSnum = MCSnum,
                    wtp=wtp, info_gain = info_gain, IsStatic = (traj_type == 'static'))
            else: raise RuntimeError('No clear env defined as %s' % env)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                # decision making frequency
                # tc = t
                u = []                
                # Rollout agent by agent
                for agent in agent_list:
                    agent.base_policy()
                    u.append(agent.v)
                for agent in agent_list:
                    u = agent.rollout(u)
                
                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u))
                info_value.append(agent_list[-1].opt_value)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def MCRollout_central(iteration, horizon, ftol, gtol, MCSnum, wtp, lite=False, central_kf=False, seq=0, 
        optmethod='de', leader=0, repeated = -1):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False

        if wtp:
            data2save = 'cen_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                    optmethod + '_MC_' + str(MCSnum)
        else:
            data2save = 'cen_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                    optmethod + '_MC_' + str(MCSnum)
        
        if central_kf:
            data2save += '_ckf'
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'MonteCarloRollout', data['env'], data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'parking_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = MCRollout(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, distriRollout=False, lite=lite, 
                central_kf=central_kf, optmethod=optmethod, factor=5, opt_step=1, MCSnum = MCSnum)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0]] * len(sensor_para_list)]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)

            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                # decision making frequency
                # centralized planning by a leader
                u = [[0,0]] * len(agent_list)
                u = agent_list[leader].rollout(u)
                for ag_nb in range(len(agent_list)):
                    agent = agent_list[ag_nb]
                    agent.v = u[ag_nb]
                    
                rollout_policy.append(u)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
                
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
                
    @staticmethod
    def base_policy(iteration, data2save, MCSnum, horizon, ftol, gtol, opt_step = -1):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        # load sensor model
        filename = os.path.join(path, 'data', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        
        sensor_para_list = data["sensors"]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", "ParkingTraj2.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'parking_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, opt_step=opt_step)
            
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        # prefix = "parkingSIM_" + str(KFonly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum) + '_ftol_' + str(ftol) + '_gtol_' + str(gtol)
        prefix = "parkingSIM_base_"
        
        for i in range(step_num):
            t = i * dt
            # # if np.isclose(t%10, 0):
            # print("t = %s" %t)
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    P = track.P
                    agent_i_est_k.append([x, y, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                # decision making frequency
                # tc = t
                u = []
                for agent in agent_list:
                    agent.v = [0, 0]
                    u.append([0 ,0])
                rollout_policy.append(u)
                
                

                # broadcast this final u to all agents
                for agent in agent_list:
                    agent.update_group_decision(u)
                
                # grab just first action in the action list
                cur_u = []
                for u_i in u:
                    cur_u.append(u_i[0])

                rollout_policy.append(cur_u)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, prefix + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def NBO_distr(iteration, horizon, ftol, gtol, wtp, env, useSemantic, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal', repeated = -1):
        '''SMA-NBO'''
        action_space = simulator.generate_discrete_action_space(horizon, 1)
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'sma', 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, useSemantic = useSemantic,
            )
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if repeated > 0:

            filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
        else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
        if not coverfile:
            my_file = Path(filename)
            if my_file.is_file():
                print("%s already exists!" % filename)
                return

        # env parameter and traj parameter
        data, traj, SemanticMap, CircleMap = simulator.map_traj_doc(env, traj_type, useSemantic, r, lambda0, iteration)

        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])            
        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes

        xs = [[27.5, -1.5], [42.5, 7.5], [4.5, 31.5], [10, 60]]
        covs = [0.5, 1.5, .5, .5]
        
        # Need to redo this TODO factor = 5, opt_step = -1
        agent_list = []
        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=False, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            elif env == 'poisson':
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=False, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'), action_space=action_space)
            else: raise RuntimeError('No clear env defined as %s' % env)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)

            # initalize all tracks
            # for track_ in track_init:
            #     ego.tracker.add_track(track_, 0)
            if traj_type == 'static':
                # add init target value to all agents
                for j in range(len(xs)):
                    
                    x0 = np.matrix(xs[j] + [0, 0]).reshape(4,1)
                    P0 = np.matrix(np.diag([covs[j]**2] * 4))
                    kf = util.dynamic_kf(np.eye(4), ego.tracker.H, x0, P0, np.zeros((4, 4)), 
                            ego.sensor_para_list[ego.id]["quality"] * np.diag([ego.sensor_para_list[ego.id]["r"], 
                            ego.sensor_para_list[ego.id]["r"]]), ego.sensor_para_list[ego.id]["r0"], 
                            quality= ego.sensor_para_list[ego.id]["quality"])
                    # else:
                    #     kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                    #         self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                    #         self.sensor_para_list[i]["r"]]))
                    kf.init = False
                    
                    new_track = util.track(0, j, kf, ego.tracker.DeletionThreshold, ego.tracker.ConfirmationThreshold)
                    new_track.kf.predict()
                    new_track.bb_box_size = []
                    new_track.confirmed = True
                    ego.tracker.track_list.append(new_track)
                    ego.tracker.track_list_next_index.append(j)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        decision_time = {}   # measures each individual agents decision time
        for i in range(len(sensor_para_list)):
            decision_time[i] = []
        
        # last time control happened
        tc = 0.0
        plan_msg = Planning_msg(len(sensor_para_list))
        missing_record = []

        for i in range(step_num):
            t = i * dt
            
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, _ = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                
                # parallel agent decision making
                u_record = []
                for index, agent in enumerate(agent_list):
                    '''agent make decision in sequence, based on other agents previous decision'''
                    agent_start_time = time.time()
                    # intention passed from previous agent
                    agent.update_group_decision(plan_msg)
                    u_index = agent.rollout()
                    decision_time[index].append(time.time() - agent_start_time)
                    u_record.append(u_index)
                    
                    # # dropout pattern: agent cannot send its policy out
                    # if dropout_pattern in ['fail_publisher', 'fail_channel']:
                    #     if random.random() > dropout_prob:
                    #         # with prob of drop_out_prob to fail sending the plan to others
                    plan_msg.push(u_index,t, index)

                
                missing_record.append(plan_msg.miss())
                
                # grab just first action in the action list
                # cur_u = []
                # for u_i in u:
                #     cur_u.append(u_i[0])

                # rollout_policy.append(cur_u)

                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u_record))
                info_value.append(agent_list[-1].opt_value)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value,
                    "decision_time": decision_time,
                    "missing_record": missing_record
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # save data
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def NBO_central(iteration, horizon, ftol, gtol, wtp, env, useSemantic, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal', repeated = -1):
        action_space = simulator.generate_discrete_action_space(horizon, 3)
        start_time = time.time()
        '''cen-NBO'''
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'cen', 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, useSemantic = useSemantic,
            )
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if repeated > 0:

            filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
        else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
        if not coverfile:
            my_file = Path(filename)
            if my_file.is_file():
                print("%s already exists!" % filename)
                return

        # env parameter and traj parameter
        data, traj, SemanticMap, CircleMap = simulator.map_traj_doc(env, traj_type, useSemantic, r, lambda0, iteration)

        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])
        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes

        xs = [[27.5, -1.5], [42.5, 7.5], [4.5, 31.5], [10, 60]]
        covs = [0.5, 1.5, .5, .5]
        
        # Need to redo this TODO factor = 5, opt_step = -1
        agent_list = []
        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=False, distriRollout=False, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            elif env == 'poisson':
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=False, distriRollout=False, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'), action_space=action_space)
            else: raise RuntimeError('No clear env defined as %s' % env)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)

            # initalize all tracks
            # for track_ in track_init:
            #     ego.tracker.add_track(track_, 0)
            if traj_type == 'static':
                # add init target value to all agents
                for j in range(len(xs)):
                    
                    x0 = np.matrix(xs[j] + [0, 0]).reshape(4,1)
                    P0 = np.matrix(np.diag([covs[j]**2] * 4))
                    kf = util.dynamic_kf(np.eye(4), ego.tracker.H, x0, P0, np.zeros((4, 4)), 
                            ego.sensor_para_list[ego.id]["quality"] * np.diag([ego.sensor_para_list[ego.id]["r"], 
                            ego.sensor_para_list[ego.id]["r"]]), ego.sensor_para_list[ego.id]["r0"], 
                            quality= ego.sensor_para_list[ego.id]["quality"])
                    # else:
                    #     kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                    #         self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                    #         self.sensor_para_list[i]["r"]]))
                    kf.init = False
                    
                    new_track = util.track(0, j, kf, ego.tracker.DeletionThreshold, ego.tracker.ConfirmationThreshold)
                    new_track.kf.predict()
                    new_track.bb_box_size = []
                    new_track.confirmed = True
                    ego.tracker.track_list.append(new_track)
                    ego.tracker.track_list_next_index.append(j)
            agent_list.append(ego)
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            
            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                # decision making frequency
                # tc = t
                u = []
                # for agent in agent_list:
                #     u.append(agent.base_policy())
                    # u.append(agent.v)
                # rollout_policy.append(u)
                
                # Rollout agent by agent
                
                u = agent_list[0].rollout(u)
                # broadcast this final u to all agents
                for i in range(len(agent_list)):
                    action = u[i][0]
                    agent = agent_list[i]
                    agent.v = copy.deepcopy(action)

                # grab just first action in the action list
                # cur_u = []
                # for u_i in u:
                #     cur_u.append(u_i[0])

                # rollout_policy.append(cur_u)

                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u))
                info_value.append(agent_list[0].opt_value)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # save file
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)

    @staticmethod
    def PMA_NBO(iteration, horizon, ftol, gtol, wtp, env, useSemantic, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal', repeated = -1, dropout_pattern = None,
        dropout_prob = 0.0):

        start_time = time.time()        
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'pma', 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, useSemantic = useSemantic,
            dropout_pattern=dropout_pattern, dropout_prob=dropout_prob)
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if repeated > 0:

            filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
        else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
        if not coverfile:
            my_file = Path(filename)
            
            if my_file.is_file():
                print("%s already exists!" % filename)
                return
        
        # env parameter and traj parameter
        # env parameter and traj parameter
        data, traj, SemanticMap, CircleMap = simulator.map_traj_doc(env, traj_type, useSemantic, r, lambda0, iteration)
        

        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])            
        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes

        xs = [[27.5, -1.5], [42.5, 7.5], [4.5, 31.5], [10, 60]]
        covs = [0.5, 1.5, .5, .5]
        
        # Need to redo this TODO factor = 5, opt_step = -1
        agent_list = []
        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=True, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            elif env == 'poisson':
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=True, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            else: raise RuntimeError('No clear env defined as %s' % env)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)

            # initalize all tracks
            # for track_ in track_init:
            #     ego.tracker.add_track(track_, 0)
            if traj_type == 'static':
                # add init target value to all agents
                for j in range(len(xs)):
                    
                    x0 = np.matrix(xs[j] + [0, 0]).reshape(4,1)
                    P0 = np.matrix(np.diag([covs[j]**2] * 4))
                    kf = util.dynamic_kf(np.eye(4), ego.tracker.H, x0, P0, np.zeros((4, 4)), 
                            ego.sensor_para_list[ego.id]["quality"] * np.diag([ego.sensor_para_list[ego.id]["r"], 
                            ego.sensor_para_list[ego.id]["r"]]), ego.sensor_para_list[ego.id]["r0"], 
                            quality= ego.sensor_para_list[ego.id]["quality"])
                    # else:
                    #     kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                    #         self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                    #         self.sensor_para_list[i]["r"]]))
                    kf.init = False
                    
                    new_track = util.track(0, j, kf, ego.tracker.DeletionThreshold, ego.tracker.ConfirmationThreshold)
                    new_track.kf.predict()
                    new_track.bb_box_size = []
                    new_track.confirmed = True
                    ego.tracker.track_list.append(new_track)
                    ego.tracker.track_list_next_index.append(j)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []       
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        decision_time = {}   # measures each individual agents decision time
        for i in range(len(sensor_para_list)):
            decision_time[i] = []
        
        # last time control happened
        tc = 0.0
        plan_msg = Planning_msg(len(sensor_para_list))
        missing_record = []

        for i in range(step_num):
            t = i * dt
            
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, _ = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)    
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                # parallel agent decision making
                u_record = []
                for index, agent in enumerate(agent_list):
                    '''agent make decision in parallel, based on other agents previous decision'''
                    agent_start_time = time.time()
                    u_index = agent.rollout()
                    decision_time[index].append(time.time() - agent_start_time)
                    u_record.append(u_index)
                    
                    # dropout pattern: agent cannot send its policy out
                    if dropout_pattern in ['fail_publisher', 'fail_channel']:
                        if random.random() > dropout_prob:
                            # with prob of drop_out_prob to fail sending the plan to others
                            plan_msg.push(u_index, t, index)
                            

                # broadcast this final u to all agents
                for agent in agent_list:
                    agent.update_group_decision(plan_msg)
                
                missing_record.append(plan_msg.miss())
                plan_msg.clean()
                
                # grab just first action in the action list
                # cur_u = []
                # for u_i in u:
                #     cur_u.append(u_i[0])

                # rollout_policy.append(cur_u)

                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u_record))
                info_value.append(agent_list[-1].opt_value)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value,
                    "decision_time": decision_time,
                    "missing_record": missing_record
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Save
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)

    
    @staticmethod
    def decPOMDP_NBO(iteration, horizon, ftol, gtol, wtp, env, useSemantic, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal', repeated=-1):

        start_time = time.time()        
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'decPOMDP', 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, useSemantic = useSemantic,
            )
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if repeated > 0:

            filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
        else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
        if not coverfile:
            my_file = Path(filename)
            if my_file.is_file():
                print("%s already exists!" % filename)
                return

        # env parameter and traj parameter
        # env parameter and traj parameter
        data, traj, SemanticMap, CircleMap = simulator.map_traj_doc(env, traj_type, useSemantic, r, lambda0, iteration)
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])            
        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes

        xs = [[27.5, -1.5], [42.5, 7.5], [4.5, 31.5], [10, 60]]
        covs = [0.5, 1.5, .5, .5]
        
        # Need to redo this TODO factor = 5, opt_step = -1
        agent_list = []
        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=True, distriRollout=False, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            elif env == 'poisson':
                
                ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=True, distriRollout=False, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'))
            else: raise RuntimeError('No clear env defined as %s' % env)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)

            # initalize all tracks
            # for track_ in track_init:
            #     ego.tracker.add_track(track_, 0)
            if traj_type == 'static':
                # add init target value to all agents
                for j in range(len(xs)):
                    
                    x0 = np.matrix(xs[j] + [0, 0]).reshape(4,1)
                    P0 = np.matrix(np.diag([covs[j]**2] * 4))
                    kf = util.dynamic_kf(np.eye(4), ego.tracker.H, x0, P0, np.zeros((4, 4)), 
                            ego.sensor_para_list[ego.id]["quality"] * np.diag([ego.sensor_para_list[ego.id]["r"], 
                            ego.sensor_para_list[ego.id]["r"]]), ego.sensor_para_list[ego.id]["r0"], 
                            quality= ego.sensor_para_list[ego.id]["quality"])
                    # else:
                    #     kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                    #         self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                    #         self.sensor_para_list[i]["r"]]))
                    kf.init = False
                    
                    new_track = util.track(0, j, kf, ego.tracker.DeletionThreshold, ego.tracker.ConfirmationThreshold)
                    new_track.kf.predict()
                    new_track.bb_box_size = []
                    new_track.confirmed = True
                    ego.tracker.track_list.append(new_track)
                    ego.tracker.track_list_next_index.append(j)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                # decision making frequency
                # tc = t
                u = []
                decision_time = {}   # measures each individual agents decision time
                decision_values = []
                for index, agent in enumerate(agent_list):
                    '''agent make decision in parallel in the manner of dec-POMDP'''
                    agent_start_time = time.time()
                    u_index = agent.rollout()
                    decision_time[index] = time.time() - agent_start_time
                    u.append(u_index)
                    decision_values.append(agent.opt_value)

                # NO broadcast this final u to all agents
                # for agent in agent_list:
                #     agent.update_group_decision(u)
                
                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u))
                info_value.append(decision_values)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value,
                    "decision_time": decision_time
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Save
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def PMA_NBO_discrete(iteration, horizon, ftol, gtol, wtp, env, useSemantic, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='de', lambda0 = 1e-3, r = 5, coverfile = False, traj_type = 'normal', repeated = -1, dropout_pattern = None,
        dropout_prob = 0.0, dv = 1e0):

        start_time = time.time()        
        isObsDyn = True
        isRotate = False
        
        data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            'pma', 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, useSemantic = useSemantic,
            dropout_pattern=dropout_pattern, dropout_prob=dropout_prob, dv=dv)
        
        path = os.getcwd()
        dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # if file exists, means we don't need to do it
        if repeated > 0:

            filename = os.path.join(dataPath, data2save + "_" + str(iteration)+ "_" + str(repeated) + ".json")
        else:  filename = os.path.join(dataPath, data2save + "_" + str(iteration) + ".json")
        if not coverfile:
            my_file = Path(filename)
            if my_file.is_file():
                print("%s already exists!" % filename)
                return
        
        # env parameter and traj parameter
        # env parameter and traj parameter
        data, traj, SemanticMap, CircleMap = simulator.map_traj_doc(env, traj_type, useSemantic, r, lambda0, iteration)

        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps
        # control parameters:
        cdt = data["control_dt"]
        step_num = len(traj[0][0])            
        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes

        xs = [[27.5, -1.5], [42.5, 7.5], [4.5, 31.5], [10, 60]]
        covs = [0.5, 1.5, .5, .5]
        
        # Need to redo this TODO factor = 5, opt_step = -1
        agent_list = []
        for i in range(len(sensor_para_list)):
            if env in ['parksim', 'simple1', 'simple2']:
                
                ego = nbo_agent_discrete(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    SemanticMap=SemanticMap, isParallel=True, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'), dv = dv)
            elif env == 'poisson':
                
                ego = nbo_agent_discrete(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                    L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                    OccupancyMap=CircleMap, isParallel=True, distriRollout=True, lite=lite, 
                    central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp, info_gain = info_gain,
                    IsStatic = (traj_type == 'static'), dv = dv)
            else: raise RuntimeError('No clear env defined as %s' % env)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)

            # initalize all tracks
            # for track_ in track_init:
            #     ego.tracker.add_track(track_, 0)
            if traj_type == 'static':
                # add init target value to all agents
                for j in range(len(xs)):
                    
                    x0 = np.matrix(xs[j] + [0, 0]).reshape(4,1)
                    P0 = np.matrix(np.diag([covs[j]**2] * 4))
                    kf = util.dynamic_kf(np.eye(4), ego.tracker.H, x0, P0, np.zeros((4, 4)), 
                            ego.sensor_para_list[ego.id]["quality"] * np.diag([ego.sensor_para_list[ego.id]["r"], 
                            ego.sensor_para_list[ego.id]["r"]]), ego.sensor_para_list[ego.id]["r0"], 
                            quality= ego.sensor_para_list[ego.id]["quality"])
                    # else:
                    #     kf = util.EKFcontrol(self.SampleF, self.tracker.H, x0, P0, self.factor * self.tracker.Q, 
                    #         self.sensor_para_list[i]["quality"] * np.diag([self.sensor_para_list[i]["r"], 
                    #         self.sensor_para_list[i]["r"]]))
                    kf.init = False
                    
                    new_track = util.track(0, j, kf, ego.tracker.DeletionThreshold, ego.tracker.ConfirmationThreshold)
                    new_track.kf.predict()
                    new_track.bb_box_size = []
                    new_track.confirmed = True
                    ego.tracker.track_list.append(new_track)
                    ego.tracker.track_list_next_index.append(j)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []       
        agent_est = []
        rollout_policy = [[[0,0], [0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        info_value = []
        decision_time = {}   # measures each individual agents decision time
        for i in range(len(sensor_para_list)):
            decision_time[i] = []
        
        # last time control happened
        tc = 0.0
        plan_msg = Planning_msg(len(sensor_para_list))
        missing_record = []

        for i in range(step_num):
            t = i * dt
            
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)    
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                # parallel agent decision making
                u_record = []
                for index, agent in enumerate(agent_list):
                    '''agent make decision in parallel, based on other agents previous decision'''
                    agent_start_time = time.time()
                    u_index = agent.rollout([])
                    decision_time[index].append(time.time() - agent_start_time)
                    u_record.append(u_index)
                    
                    # dropout pattern: agent cannot send its policy out
                    if dropout_pattern in ['fail_publisher', 'fail_channel']:
                        if random.random() > dropout_prob:
                            # with prob of drop_out_prob to fail sending the plan to others
                            plan_msg.u[index] = u_index

                # broadcast this final u to all agents
                for agent in agent_list:
                    agent.update_group_decision(plan_msg)
                
                missing_record.append(plan_msg.miss())
                plan_msg.clean()
                
                # grab just first action in the action list
                # cur_u = []
                # for u_i in u:
                #     cur_u.append(u_i[0])

                # rollout_policy.append(cur_u)

                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u_record))
                info_value.append(agent_list[-1].opt_value)
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "info_value": info_value,
                    "decision_time": decision_time,
                    "missing_record": missing_record
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Save
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)


    @staticmethod
    def NBO_para_opt_central(iteration, horizon, ftol, gtol, lite=False, central_kf=False, seq=0, 
        optmethod='de', repeated = -1):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        data2save = 'cen_select_para_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + optmethod
        if central_kf:
            data2save += '_ckf'
        
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'nbo', data['env'], data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'parking_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, distriRollout=False, lite=lite, 
                central_kf=central_kf, optmethod=optmethod, factor=1, opt_step=5)
            ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
        # broadcast information and recognize neighbors
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)

            
            t = i * dt
            # # if np.isclose(t%10, 0):
            # print("t = %s" %t)
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                u = []
                u = agent_list[0].rollout(u)
                
                # broadcast this final u to all agents
                for i in range(len(agent_list)):
                    action = u[i][0]
                    agent = agent_list[i]
                    agent.v = copy.deepcopy(action)
                    
                
                rollout_policy.append(copy.deepcopy(u))
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
        
        filename = os.path.join(dataPath, "para_" + data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(agent_list[0].pso_para_record, outfiles, indent=4)

    @staticmethod
    def ibr_parallel(iteration, horizon, wtp, ibr_num, central_kf=False):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        if wtp:
            data2save = 'parallel_wtp_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
        else:
            data2save = 'parallel_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
        if central_kf:
            data2save += '_ckf'
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'ibr', data['env'], data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        
        sensor_para_list = data["sensors"]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'parking_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = ibr_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, 
                SemanticMap=SemanticMap, 
                central_kf=central_kf, factor=5, 
                wtp=wtp)
            agent_list.append(ego)
                        
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0]] * len(sensor_para_list)]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        is_pne = []
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)

            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                print("start optimization")
                init_time = time.time()
                pne = False
                # decision making frequency
                # tc = t
                u = []
                for agent in agent_list:
                    agent.base_policy()
                    u.append(agent.v)
                
                ibr_list = {"policy": [], "value": []}
                # iteration happens now
                for ibrs in range(ibr_num):
                    # Rollout agent by agent
                    new_u = []
                    for agent in agent_list:
                        action = agent.ibr_parallel(u)
                        new_u.append(action)
                        
                    
                    if new_u == u:
                        # PNE found
                        pne = True
                        break
                    # evaluation
                    value = agent_list[0].rolloutTrailSim_centralized(new_u)
                    ibr_list["policy"].append(new_u)
                    ibr_list["value"].append(value)
                    u = copy.deepcopy(new_u)
                if not pne:
                    min_index = ibr_list["value"].index(min(ibr_list["value"]))
                    u = ibr_list["policy"][min_index]
                
                # execute the policy
                for agentid in range(len(agent_list)):
                    agent_list[agentid].v = u[agentid]

                    
                rollout_policy.append(u)
                is_pne.append([pne, ibrs])
            
                print("ibr ends with time cost %s and pne = %s" % (time.time() - init_time, pne))
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy,
                    "PNE": is_pne
                }

        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    

    @staticmethod
    def deterministic_optimization(iteration, horizon, ftol, gtol, lite=False, central_kf=False, seq=0, 
        optmethod='de'):
        
        isObsDyn = True
        isRotate = False
        
        data2save = 'cen_cheat_parking_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + optmethod
        if central_kf:
            data2save += '_ckf'
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'nbo', data['env'], data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'parking_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, distriRollout=False, lite=lite, 
                central_kf=central_kf, optmethod=optmethod, factor=1, opt_step=5)
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num - agent_list[0].SampleNum):
            t = i * dt
            # # if np.isclose(t%10, 0):
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for j in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[j].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, P])
                    
                agent_est[j].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                # decision making frequency
                # tc = t
                u = []
                future_traj = []
                for j in range(1, agent_list[0].SampleNum + 1):
                    z_j = []
                    for ii in range(len(traj[0])):
                        z_j.append([traj[0][ii][i + j], traj[1][ii][i + j]])
                    future_traj.append(z_j)
                
                u = agent_list[0].cheat_rollout(u, future_traj)
                for i in range(len(agent_list)):
                    action = u[i][0]
                    agent = agent_list[i]
                    agent.v = copy.deepcopy(action)
                    
                rollout_policy.append(copy.deepcopy(u))
            # update position given policy u
            for j in range(len(agent_list)):
                
                agent = agent_list[j]                
                agent.dynamics(dt)
                sensor_para_list[j]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, prefix + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def NBO_simple_central(iteration, horizon, ftol, gtol, lite=False, central_kf=False, seq=0, 
        optmethod='de'):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        data2save = 'cen_simple1_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + optmethod
        if central_kf:
            data2save += '_ckf'
        
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'simplepara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'nbo', data['env'], data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "simpletraj2.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'simple_map.json')
        with open(filename) as json_file:
            SemanticMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, distriRollout=False, lite=lite, 
                central_kf=central_kf, optmethod=optmethod)
            ego.tracker.DeletionThreshold = [100, 100]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
        # broadcast information and recognize neighbors
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)

            
            t = i * dt
            # # if np.isclose(t%10, 0):
            # print("t = %s" %t)
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)
                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                u = []
                u = agent_list[0].rollout(u)
                
                # broadcast this final u to all agents
                for i in range(len(agent_list)):
                    action = u[i][0]
                    agent = agent_list[i]
                    agent.v = copy.deepcopy(action)
                    
                
                rollout_policy.append(copy.deepcopy(u))
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)

    @staticmethod
    def NBO_simple_distr(iteration, horizon, ftol, gtol, seq, useSemantic=False, sigma=False, lite=False, 
        central_kf=False, optmethod='de'):
        
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        data2save = 'distr_simple_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + optmethod
        if central_kf:
            data2save += '_ckf'
        
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'simplepara2.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'nbo', data['env'], data2save)


        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", "simpletraj2.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        if useSemantic:
            filename = os.path.join(path, 'data', 'simple_map.json')
            with open(filename) as json_file:
                SemanticMap = json.load(json_file)
        else:
            SemanticMap = None

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                SemanticMap=SemanticMap, isParallel=False, sigma=sigma, lite=lite, central_kf=central_kf, 
                optmethod=optmethod)
            ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0
        
        for i in range(step_num):
            t = i * dt
            # # if np.isclose(t%10, 0):
            # print("t = %s" %t)
            z_k = []
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                
                u = []
                
                for agent in agent_list:
                    u = agent.rollout(u)

                # broadcast this final u to all agents
                for agent in agent_list:
                    agent.update_group_decision(u)
                
                rollout_policy.append(copy.deepcopy(u))
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)
    
    @staticmethod
    def occupancy_nbo_distr(iteration, horizon, ftol, gtol, wtp, lambda0, r, lite=False, central_kf=False, seq=0, 
        optmethod='de'):
        start_time = time.time()
        
        isObsDyn = True
        isRotate = False
        
        if wtp:
            data2save = 'dis_wtp_poisson_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                    optmethod + '_seq_' + str(seq)
        else:
            data2save = 'dis_poisson_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                    optmethod + '_seq_' + str(seq)

        if central_kf:
            data2save += '_ckf'
        
        
        path = os.getcwd()
        # load sensor model
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)

        dataPath = os.path.join(path, 'data', 'result', 'nbo', 'poisson', data2save)

        try:
            os.mkdir(dataPath)
        except OSError:
            pass
        
        
        agent_seq = data["seq"][str(seq)]
        sensor_para_list = [data["sensors"][i-1] for i in agent_seq]
        dt = data["dt"]
        # consensus parameters:
        N = 5   #consensus steps

        # control parameters:
        cdt = data["control_dt"]
        
        # load trajectories of targets
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
        # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
        with open(filename) as json_file:
            traj = json.load(json_file)

        step_num = len(traj[0][0])
        
        # load semantic maps
        filename = os.path.join(path, 'data', 'env', 'poisson', 'r_' + str(r) + '_lambda_' + str(lambda0) + '_' + str(iteration) + '.json')
        with open(filename) as json_file:
            OccupancyMap = json.load(json_file)

        time_set = np.linspace(dt, dt * step_num, step_num)
        # 1. initialization of nodes
        
        
        agent_list = []
        for i in range(len(sensor_para_list)):
            ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
                L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
                OccupancyMap=OccupancyMap, isParallel=False, distriRollout=True, lite=lite, 
                central_kf=central_kf, optmethod=optmethod, factor=5, wtp=wtp)
            # ego.tracker.DeletionThreshold = [30, 40]
            # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
            agent_list.append(ego)
            
            
        # 2. consensus IF
        true_target_set = []    
        
        agent_est = []
        rollout_policy = [[[0,0], [0,0]]]
        for i in range(len(sensor_para_list)):
            agent_est.append([])
            
        obs_set = []
        con_agent_pos = []
        
        # last time control happened
        tc = 0.0

        for i in range(step_num):
            t = i * dt
            # broadcast information and recognize neighbors
            pub_basic = []
            for agent in agent_list:
                basic_msg = Agent_basic()
                basic_msg.id = agent.id
                basic_msg.x = agent.sensor_para["position"][0]
                basic_msg.y = agent.sensor_para["position"][1]
                basic_msg.theta = agent.sensor_para["position"][2]
                pub_basic.append(basic_msg)
            
            # make each agent realize their neighbors
            for agent in agent_list:
                agent.basic_info(pub_basic)
            
            z_k = []
            for ii in range(len(traj[0])):
                z_k.append([traj[0][ii][i], traj[1][ii][i]])
            obs_k = []
            info_list = []
            
            # 1. feed info to sensors
            for agent in agent_list:
                info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
                info_list.append(info_0)                
                obs_k += z_k_out
                
            obs_set.append(obs_k)
            

            # 2. consensus starts
            for l in range(N):
                # receive all infos
                for agent in agent_list:
                    agent.grab_info_list(info_list)
                
                info_list = []
                # then do consensus for each agent, generate new info
                for agent in agent_list:
                    info_list.append(agent.consensus())

            # 3. after fixed num of average consensus, save result in menory
            for i in range(len(sensor_para_list)):

                agent_i_est_k = []
                
                for track in info_list[i].tracks:
                    x = track.x[0]
                    y = track.x[1]
                    vx = track.x[2]
                    vy = track.x[3]
                    P = track.P
                    agent_i_est_k.append([x, y, vx, vy, P])
                    
                agent_est[i].append(agent_i_est_k)
                
            # 4. agent movement 
            
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                # decision making frequency
                # tc = t
                u = []
                # for agent in agent_list:
                #     u.append(agent.base_policy())
                    # u.append(agent.v)
                # rollout_policy.append(u)
                
                # Rollout agent by agent
                for agent in agent_list:
                    u = agent.rollout(u)

                # broadcast this final u to all agents
                for agent in agent_list:
                    agent.update_group_decision(u)
                
                # grab just first action in the action list
                # cur_u = []
                # for u_i in u:
                #     cur_u.append(u_i[0])

                # rollout_policy.append(cur_u)

                # record all actions to generate the plan in near future
                rollout_policy.append(copy.deepcopy(u))
            # update position given policy u
            for i in range(len(agent_list)):
                
                agent = agent_list[i]                
                agent.dynamics(dt)
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
            
            con_agent_pos.append(copy.deepcopy(agent_pos_k))

            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                tc = t
                output_data = {
                    "true_target": true_target_set,
                    "agent_est": agent_est,
                    "agent_pos": con_agent_pos,
                    "agent_obs": obs_set,
                    "policy": rollout_policy
                }
        
        end_time = time.time()
        output_data["time"] = end_time - start_time
        # Recurrsive Save

        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename, 'w') as outfiles:
            json.dump(output_data, outfiles, indent=4)