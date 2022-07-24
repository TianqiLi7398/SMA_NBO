import numpy as np
from utils.msg import Agent_basic, Single_track, Info_sense
from utils.MonteCarloRollout import MCRollout
from utils.nbo import nbo_agent
from utils.ibr import ibr_agent
from utils.occupancy import occupancy
import random
import copy
import json
import csv
import os, sys
import time
import matplotlib.pyplot as plt

def euclidan_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def replay(iteration, horizon, ftol, gtol, wtp, lite=False, central_kf=False, seq=0, 
    optmethod='de'):
    

    isObsDyn = True
    isRotate = False

    if wtp:
        date2run = 'dis_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                optmethod + '_seq_' + str(seq)
    else:
        date2run = 'dis_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + \
                optmethod + '_seq_' + str(seq)

    if central_kf:
        date2run += '_ckf'


    path = os.getcwd()
    # load sensor model
    filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    dataPath = os.path.join(path, 'data', 'result', 'nbo', data['env'], date2run)

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

    policyfile = os.path.join(dataPath, date2run + "_"+str(iteration)+".json")
    
    with open(policyfile) as json_file:
        result = json.load(json_file)
    
    
    # 1. initialization of nodes

    # Need to redo this TODO factor = 5, opt_step = -1
    agent_list = []
    for i in range(len(sensor_para_list)):
        ego = nbo_agent(horizon, copy.deepcopy(sensor_para_list), i, dt, cdt,
            L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, ftol=ftol, gtol=gtol, 
            SemanticMap=SemanticMap, isParallel=False, distriRollout=True, lite=lite, 
            central_kf=central_kf, optmethod=optmethod, factor=1, opt_step=5, wtp=wtp)
        # ego.tracker.DeletionThreshold = [30, 40]
        # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
        agent_list.append(ego)
        
    
    consensus_obs = {}

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
        info_list_report = []
        
        # 1. feed info to sensors
        for agent in agent_list:
            info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
            info_list.append(info_0)                
            info_report = agent.report_tracks()
            info_list_report.append(info_report)
            obs_k += z_k_out
            
        obs_set.append(obs_k)
        
        consensus_obs[i] = []
        consensus_obs[i].append(info_list_report)
        # 2. consensus starts
        for l in range(N):
            # receive all infos
            for agent in agent_list:
                agent.grab_info_list(info_list)
            info_list = []
            info_list_report = []
            # then do consensus for each agent, generate new info
            for agent in agent_list:
                info_list.append(agent.consensus())
                info_list_report.append(agent.report_tracks())
            consensus_obs[i].append(info_list_report)

        # 3. after fixed num of average consensus, save result in menory
        for ii in range(len(sensor_para_list)):

            agent_i_est_k = []
            
            for track in info_list[ii].tracks:
                x = track.x[0]
                y = track.x[1]
                vx = track.x[2]
                vy = track.x[3]
                P = track.P
                agent_i_est_k.append([x, y, vx, vy, P])
                
            agent_est[ii].append(agent_i_est_k)
            
        # 4. agent movement 
        
        
        for ii in range(len(agent_list)):
            
            agent = agent_list[ii]                
            # agent.dynamics(dt)
            agent.sensor_para["position"][0] = result["agent_pos"][i][ii][0]
            agent.sensor_para["position"][1] = result["agent_pos"][i][ii][1]
            sensor_para_list[ii]["position"] = result["agent_pos"][i][ii]

    return consensus_obs, step_num

def analysis_consensus(consensus_obs, step_num, track_num, t = -1, L = 5):

    path = os.getcwd()
    filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
    # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    agent_num = 3


    if t > 0:

        process = {}
        for i in range(agent_num):
            process[i] = []
        
        ii = 0
        z_k = [traj[0][ii][t], traj[1][ii][t]]

        for l in range(L):
            info_l = consensus_obs[t][l]
            for i in range(agent_num):
                agent_infos = info_l[i]
                process[i].append(find_match(agent_infos, z_k))
    
        for i in range(agent_num):
            plt.plot(process[i])
        plt.show()
                
    else: 
        process = {}
        for i in range(agent_num):
            process[i] = []
        for t in range(step_num):
            
            ii = track_num
            z_k = [traj[0][ii][t], traj[1][ii][t]]
            error_list = {}
            for i in range(agent_num):
                error_list[i] = []

            for l in range(L):
                info_l = consensus_obs[t][l]
                
                for i in range(agent_num):
                    agent_infos = info_l[i]
                    error_list[i].append(find_match(agent_infos, z_k))
            for i in range(agent_num): process[i].append(error_list[i])
        
        
        color_bar = ['r', 'b', 'g', 'yellow']
        mean_list = []
        conf_inter = []
        for i in range(agent_num):
            dist_ = np.matrix(process[i][2:])
            
            mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            mean_list.append(mean)
            
            
            
            conf_inter.append(1.96 * std / np.sqrt(step_num))  # 97.5% percentile point of the standard normal distribution
            
        for i in range(agent_num):
            plt.plot(mean_list[i], color=color_bar[i])
            plt.fill_between(range(L), mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
        plt.title("Average Error in Consensus of Information")
        plt.xticks(np.arange(0, 5, 1))
        plt.xlabel("Consensus step")
        plt.ylabel("Euclidean error/m")
        plt.grid()
        plt.show()

def find_match(info_list, z):
    error = 10000
    index = 0
    # print("z: %s"%z)
    for i in range(len(info_list.tracks)):
        info = info_list.tracks[i]
        # print("est: %s"%info.x[0:2])
        error_new = euclidan_dist(info.x[0:2], z)
        if error_new < error:
            error = error_new
            index = i 
    
    return error

if __name__ == "__main__":
    sim_num = 50
    MCSnum, horizon = 50, 1 # 3.0, 5.0, 8.0
    isCentralized = False
    domain = 'nbo'  # or 'MonteCarloRollout'
    ftol = 5e-4   #1e-3   #5e-3
    gtol = 50   #10
    opt_step = -1
    useSemantic = True
    sigma = False
    lite = True   # use max consensus in NBO
    ckf = True   # what sensor fusion to use in NBO, centralized kf or consensus
    method = 'pso'   # 'pso' or 'de'
    case = 'parking'
    wtp = True
    ibr_num = 4
    lambda0 = 0.005  # tree density
    r = 5    # radius of tree
    consensus_obs, step_num = replay(0, horizon, ftol, gtol, wtp, lite=lite, central_kf=ckf, seq=0, 
    optmethod=method)
    analysis_consensus(consensus_obs, step_num, 3, t=0)