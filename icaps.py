from utils.simulator import simulator
from utils.effects import effects
import numpy as np
import os, sys
import json
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
from matplotlib.text import Text


def plot_trajs(iteration, horizon, ftol, gtol, wtp, env, info_gain = False, lite=False, central_kf=False, seq=0, 
        optmethod='pso', lambda0 = 1e-3, r = 5, traj_type = 'normal', ddt=25, time_length=200, dt=25):

    '''
    plots the trajectory of agents in the simulation
    '''

    # filename of a certain file
    data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            False, 'nbo', lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain)
        
    path = os.getcwd()
    dataPath = os.path.join(path, 'data', 'result', 'nbo', env, data2save)
    filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    
    filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'env', 'poisson', 'r_' + str(r) + '_lambda_' + \
                        str(lambda0) + '_' + str(iteration) + '.json')
    with open(filename) as json_file:
        OccupancyMap = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    
    color_list_gradient = [
        # ['#800000', '#8B0000', '#A52A2A', '#B22222', '#DC143C', '#FF0000'],
        # ['#800000', '#FA8072', '#FFA500', '#EEE8AA', '#9ACD32', '#00FFFF', '#8A2BE2']
        ['#731603', '#bf2708', '#f03b16', '#ed5d40', '#ed7961', '#e89280', '#e8a99b'],
        ['#052f73', '#0545ad', "#075fed", '#2a73e8', '#4c87e6', '#6e9ce6', '#8daee3'],
        ['#4a0225', '#8f0448', '#cc0667', '#e84896', '#e86faa', '#f08dbd', '#f0b9d3'],
        ['#000000', '#232323', "#494949", "#696969", '#808080', '#A9A9A9', '#B9B9B9', '#C0C0C0', '#D0D0D0']
    ]

    color_list = ['red', "blue", 'purple']

    gradient_list = ['Reds', 'Blues', 'PuRd', 'Greys']
    h = 2

    # pic 1, initial position and setup of trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1]), 
                            ylim=(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1]))
    ax.set_xticks(range(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1], 5) ,minor=True)
    ax.set_yticks(range(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1], 5) ,minor=True)
    ax.set_xticks(range(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1], 10))
    ax.set_yticks(range(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1], 10))
    ax.grid(which="major",alpha=0.6)
    ax.grid(which="minor",alpha=0.3)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # add occlusions
    for j in range(len(OccupancyMap["centers"][0])):
        
        e = Circle((OccupancyMap['centers'][0][j], OccupancyMap['centers'][1][j]), 
            radius = r,
            fc ='none',  
            ec ='b', 
            lw = 1,
            linestyle = '-')
        ax.add_artist(e)

    # position of agents
    
    for i in range(len(data["sensors"])):
        rec_cen = record["agent_pos"][0][i]
        
        e = effects.make_rectangle(rec_cen[0], rec_cen[1], rec_cen[2], data["sensors"][i], color=color_list_gradient[i][3])
        ax.add_artist(e)
        
    
    for t in range(int(round(time_length/ddt))+1):
        x, y = [], []
        h = t 
        for i in range(len(traj[0])):
            x.append(traj[0][i][t*ddt])
            y.append(traj[1][i][t*ddt])
        plt.scatter(x, y, c=color_list_gradient[-1][h])

        
    # positions of targets in the record
    for i in range(len(traj[0])):
        plt.plot(traj[0][i][0:time_length], traj[1][i][0:time_length], color='grey')
    
    plt.tight_layout()
    
    filename = os.path.join(path, 'pics', 'Parking_slowmotion1.png')
    plt.savefig(filename, dpi=800)
    plt.close()

if __name__ == "__main__":
    
    sim_num = 50
    MCSnum, horizons = 50, [1, 3, 5] # 3.0, 5.0, 8.0
    horizon = 3
    isCentralized = False
    domain = 'MonteCarloRollout'  # 'nbo' or 'MonteCarloRollout'
    ftol = 5e-4   #1e-3   #5e-3
    gtol = 50   #10
    opt_step = -1
    useSemantic = True  # False for poisson?
    sigma = False
    lite = True   # use max consensus in NBO
    ckf = True   # what sensor fusion to use in NBO, centralized kf or consensus
    method = 'pso'   # 'pso' or 'de'
    case = 'poisson' # 'poisson' or 'parking' or 'simple1' or 'simple2'
    wtp = False
    ibr_num = 4
    lambda0 = 5e-3
    lambda0_list = [1e-3, 3e-3, 5e-3]  # tree density
    r = 5    # radius of tree
    traj_type = 'normal' # 'straight' or 'normal' or 'static'
    info_gain = 'trace_sum' 
    plot_trajs(0, horizon, ftol, gtol, wtp, case, info_gain=info_gain, central_kf=ckf, r=r, lambda0=lambda0)