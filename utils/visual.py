import numpy as np
import json
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
from scipy.stats.distributions import chi2
from utils.effects import effects
from utils.simulator import simulator
import argparse


def replay_poisson(horizon, lambda0, r, iter_num, env, domain, wtp, central_kf=False, seq=0, 
        optmethod='de', ftol=1e-3, gtol=7, traj_type = 'normal', info_gain = False, deci_Schema='dis', 
        repeated = -1,):

    path = os.getcwd()

    data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            deci_Schema, domain, lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain)

    dataPath = os.path.join(path, 'data', 'result', domain, env, data2save)

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
    with open(filename) as json_file:
        traj = json.load(json_file)

    videopath = os.path.join(path, 'video', domain, env, data2save)
    try:
        os.mkdir(videopath)
        print("path %s made"%videopath)
    except OSError:
        print("path %s not made"%videopath)
        
    if repeated > 0:   iter_num = repeated
    
    x_samples, y_samples = traj

    for iteration in range(iter_num):
        if env == 'parksim':
            filename = os.path.join(path, 'data', 'env', 'parking_map.json')
            with open(filename) as json_file:
                OccupancyMap = json.load(json_file)
            OccupancyMap["centers"] = []
        elif env == 'poisson':
            # load semantic maps
            i = iteration
            if repeated > 0:
                i = 0
            
            filename = os.path.join(path, 'data', 'env', 'poisson', 'r_' + str(r) + '_lambda_' + \
                        str(lambda0) + '_' + str(i) + '.json')
            with open(filename) as json_file:
                aMap = json.load(json_file)
            OccupancyMap = {'centers': aMap['centers'], 
                            'r': r, 
                            'canvas': aMap['canvas']}
        elif env == 'simple1' or 'simple2':
            filename = os.path.join(path, 'data', 'env', 'simple_map.json')
            with open(filename) as json_file:
                OccupancyMap = json.load(json_file)
            OccupancyMap["centers"] = []
        
        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        if repeated > 0:
            filename = os.path.join(dataPath, data2save + "_0_" + str(iteration) + ".json")



        filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
        with open(filename) as json_file:
            record = json.load(json_file)
        record["policy"].pop(0)
        step_num = len(record["agent_est"][0])
        # assert np.isclose(step_num, len(traj[0][0]))
        sensor_para_list = data["sensors"]
        dt = data["dt"]
        # consensus parameters:
        
        obs_set = record["agent_obs"]    
        con_agent_pos = record["agent_pos"]
        agent_est = []
        for i in range(len(sensor_para_list)):
            agent_est.append([])
        

        for i in range(step_num):
            for agentid in range(len(sensor_para_list)):
                est_k_i = []
                
                for track in record["agent_est"][agentid][i]:
                    est_k_i.append([track[0], track[1], track[2], track[3], np.array(track[-1]).reshape(4,4)])
                agent_est[agentid].append(est_k_i)
        
        P_G = 0.9
        gating_size = chi2.ppf(P_G, df = 2)
        
        global ax, fig
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1]), 
                            ylim=(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1]))

        trj1, = ax.plot([], [], 'ko', ms=2)
        trj2, = ax.plot([], [], 'ro', ms=2)
        trj3, = ax.plot([], [], 'go', ms=2)
        trj4, = ax.plot([], [], 'bo', ms=2)
        
        trjs = [trj1, trj2, trj3, trj4]
        consensus_est, = ax.plot([], [], 'r*', ms=2)

        obs_pot, = ax.plot([], [], 'b*', ms=2)
        time_display = ax.text(0, 50, 't = 0', fontsize = 10)
        ax.set_xticks(range(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1], 5) ,minor=True)
        ax.set_yticks(range(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1], 5) ,minor=True)
        ax.set_xticks(range(OccupancyMap['canvas'][0][0], OccupancyMap['canvas'][0][1], 10))
        ax.set_yticks(range(OccupancyMap['canvas'][1][0], OccupancyMap['canvas'][1][1], 10))
        ax.grid(which="major",alpha=0.6)
        ax.grid(which="minor",alpha=0.3)
        
        def init():

            """initialize animation"""
            # add FoVs
            trj1.set_data([], [])

            
            return trj1,

        def animate(i):
            """perform animation step"""
        
            for j in range(len(traj[0])):
                trj = trjs[j]
                x, y = [], []
                x.append(x_samples[j][i])
                y.append(y_samples[j][i])
                
                trj.set_data(x, y)

            for obj in ax.findobj(match = Ellipse):
                obj.remove()
            
            for obj in ax.findobj(match = FancyArrowPatch):
                obj.remove()

            # for obj in ax.findobj(match = Text):
                
            #     obj.set_visible(False)
            
            for obj in ax.findobj(match = Circle):
                obj.remove()
            
            for obj in ax.findobj(match = Rectangle):
            
                if obj._width > 1:
                    obj.remove()
        
            # add centers
            try:
                for j in range(len(OccupancyMap["centers"][0])):
                    
                    e = Circle((OccupancyMap['centers'][0][j], OccupancyMap['centers'][1][j]), 
                        radius = r,
                        fc ='none',  
                        ec ='b', 
                        lw = 1,
                        linestyle = '-')
                    ax.add_artist(e)
            except:  pass
            x, y = [], []
            for track in agent_est[0][i]:
                mu = [track[0], track[1]]
                x.append(mu[0])
                y.append(mu[1])
                P = track[-1]
                # trackID = track[3]
                S = np.matrix(P).reshape(4,4)[0:2, 0:2]
                e = effects.make_ellipse(mu, S, gating_size, 'r')
                ax.add_artist(e)
                
            consensus_est.set_data(x, y)
            
            time_display.set_text("t = %s" % round(dt*i + dt, 1))
            
            for j in range(len(con_agent_pos[i])):
                rec_cen = [con_agent_pos[i][j][0], con_agent_pos[i][j][1]]
                e = effects.make_rectangle(rec_cen[0], rec_cen[1], con_agent_pos[i][j][2], sensor_para_list[j])
                ax.add_artist(e)
            
            for point in obs_set[i]:
                
                x.append(point[0])
                y.append(point[1])
            obs_pot.set_data(x, y)
            

            return trj1, trj2, trj3, trj4

        ani = animation.FuncAnimation(fig, animate, frames=step_num,
                                    interval=10, blit=True, init_func=init, repeat = False)
        path = os.getcwd()

        videopath = os.path.join(path, 'video', domain, env, data2save)
        try:
            os.mkdir(videopath)
        except OSError:
            pass
        
        filename = os.path.join(videopath, data2save + "_"+str(iteration)+'.mp4')
        if not os.path.exists(videopath):
            os.makedirs(videopath)
        if repeated > 0:
            filename = os.path.join(videopath, data2save + "_0_"+str(iteration)+'.mp4')
        ani.save(filename, fps=5)
        plt.close()
        print("Video saved in ", filename)



def visualize(args: argparse.Namespace, seq = 0):
    
    replay_poisson(args.horizon, args.lambda0, args.r, args.iteration, args.case, args.domain, args.wtp, 
        central_kf=args.ckf, seq=seq, 
        optmethod=args.optmethod, ftol=args.ftol, gtol=args.gtol, traj_type = args.traj_type, 
        info_gain = args.info_gain, deci_Schema=args.deci_Schema,
        repeated=args.repeated_times)
