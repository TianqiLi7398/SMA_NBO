import numpy as np
from utils.msg import Agent_basic, Single_track, Info_sense
from utils.MonteCarloRollout import MCRollout
from utils.effects import effects
import random
import copy
import json
import csv
import os, sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
from matplotlib.text import Text
import os, sys
from scipy.stats.distributions import chi2
from utils.metrics import ospa
from utils.simulator import simulator

def replay(dataPath, data2save, domain, wantVideo, isbatch=False, iteration=0, isCentral=False, 
            ftol=1e-3, gtol=7, agentid=0, horizon=5, seq=0, video=False, lite=False, central_kf=False,
            case='parking'):

    # isMoving = True
    # isObsDyn = True
    # isRotate = False
    # isFalseAlarm = False
    # isKFOnly = True
    
    path = os.getcwd()
    
    if case == 'parking':
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    else:
        filename = os.path.join(path, 'data', 'env', 'simplepara.json')

    with open(filename) as json_file:
        data = json.load(json_file)
    
    if case == 'parking':
        filename = os.path.join(path, 'data','env',  'parking_map.json')
    else:
        filename = os.path.join(path, 'data', 'env', 'simple_map.json')

    with open(filename) as json_file:
        SemanticMap = json.load(json_file)
    
    # load trajectories of targets
    if case == 'parking':
        filename = os.path.join(path, "data",'env', "ParkingTraj2.json")
    else:
        filename = os.path.join(path, 'data','env', 'simpletraj2.json')

    with open(filename) as json_file:
        traj = json.load(json_file)

    # step_num = 7
    x_samples, y_samples = traj
    filename = os.path.join(dataPath, data2save + "_"+str(iteration)+".json")
    with open(filename) as json_file:
        record = json.load(json_file)
    record["policy"].pop(0)
    step_num = len(record["agent_est"][agentid])
    sensor_para_list = data["sensors"]
    dt = data["dt"]
    T = step_num * dt
    cdt = data["control_dt"]
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
    if video:
        VideoGenerator_horizon(x_samples, y_samples, agent_est, agentid, con_agent_pos, data2save, domain, data['env'], SemanticMap,
            dt, cdt, sensor_para_list, obs_set, step_num, isbatch, iteration, record["policy"], horizon, T)
        # Vel_polt(x_samples, y_samples, agent_est, agentid, dt, prefix, iteration)

    # else:
    #     for i in range(step_num):
    #         for agentid in range(len(sensor_para_list)):
    #             est_k_i = []
                
    #             for track in record["agent_est"][agentid][i]:
    #                 est_k_i.append([track[0], track[1], np.array(track[2]).reshape(4,4)])
    #             agent_est[agentid].append(est_k_i)

    VideoGenerator(x_samples, y_samples, agent_est, agentid, con_agent_pos, data2save, domain, data['env'], SemanticMap,
            dt, sensor_para_list, obs_set, step_num, isbatch, iteration, isReplay=True)

    
def VideoGenerator(x_samples, y_samples, agent_est, agentid, con_agent_pos, data2save, domain, env, SemanticMap,
            dt, sensor_para_list, obs_set, step_num, isbatch, iteration, isReplay = False):
    P_G = 0.9
    gating_size = chi2.ppf(P_G, df = 2)
    l = agentid
    global ax, fig
    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-10 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-20 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))

    trj1, = ax.plot([], [], 'ko', ms=2)
    trj2, = ax.plot([], [], 'ro', ms=2)
    trj3, = ax.plot([], [], 'go', ms=2)
    trj4, = ax.plot([], [], 'bo', ms=2)
    
    trjs = [trj1, trj2, trj3, trj4]
    consensus_est, = ax.plot([], [], 'r*', ms=2)

    obs_pot, = ax.plot([], [], 'b*', ms=2)
    
    def init():

        """initialize animation"""
        # add FoVs
        trj1.set_data([], [])
        e = Rectangle((0, 0), 
                    100, 60,
                    angle = 0,
                    fc ='none',  
                    ec ='b', 
                    lw = 1,
                    linestyle = '-')
        ax.add_artist(e)

        
        return trj1,

    def animate(i):
        """perform animation step"""
    
        for j in range(len(trjs)):
            trj = trjs[j]
            x, y = [], []
            x.append(x_samples[j][i])
            y.append(y_samples[j][i])
            
            trj.set_data(x, y)

        for obj in ax.findobj(match = Ellipse):
            obj.remove()
        
        for obj in ax.findobj(match = FancyArrowPatch):
            obj.remove()
        
        for obj in ax.findobj(match = Rectangle):
            
            if obj._width > 1:
                obj.remove()

        for obj in ax.findobj(match = Text):
            
            obj.set_visible(False)
    
        for block in SemanticMap["Nodes"]:
            e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                    max(block['feature']['x']) - min(block['feature']['x']), 
                    max(block['feature']['y']) - min(block['feature']['y']),
                    angle = 0,
                    fc ='none',  
                    ec ='g', 
                    lw = 2,
                    linestyle = '-')
            ax.add_artist(e)

        x, y = [], []
        for track in agent_est[l][i]:
            mu = [track[0], track[1]]
            x.append(mu[0])
            y.append(mu[1])
            P = track[-1]
            # trackID = track[3]
            S = np.matrix(P).reshape(4,4)[0:2, 0:2]
            e = effects.make_ellipse(mu, S, gating_size, 'r')
            ax.add_artist(e)
            
            # ax.add_artist(Text(track[0] + 2, track[1] + 2, 'ID: ' + str(trackID), fontsize = 10))
        consensus_est.set_data(x, y)
        ax.add_artist(Text(0, 0, "t = %s" % round(dt*i + dt, 1), fontsize = 10))
        
        for j in range(len(con_agent_pos[i])):
            rec_cen = [con_agent_pos[i][j][0], con_agent_pos[i][j][1]]
            # TODO theta will be derived from csv too
            # h = sensor_para_list[j]["shape"][1][1]
            # w = sensor_para_list[j]["shape"][1][0]
            # x, y = left_bot_point(seq_agent_pos[i][j][0], seq_agent_pos[i][j][1], theta, h, w)

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
    
    ani.save(filename, fps=5)
    plt.close()
    print("Video saved in ", filename)


def VideoGenerator_horizon(x_samples, y_samples, agent_est, agentid, con_agent_pos, data2save, domain, env, SemanticMap,
            dt, cdt, sensor_para_list, obs_set, step_num, isbatch, iteration, policy, horizon, T):
    color_bar = ['r', 'b', "y", 'g', 'orange', '#6e9ce6', '#8daee3']
    P_G = 0.9
    gating_size = chi2.ppf(P_G, df = 2)
    l = agentid
    global ax, fig
    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-10 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-20 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))

    trj1, = ax.plot([], [], 'k', ms=2)
    trj2, = ax.plot([], [], 'r', ms=2)
    trj3, = ax.plot([], [], 'g', ms=2)
    trj4, = ax.plot([], [], 'b', ms=2)
    # trjs = [trj1, trj2]
    trjs = [trj1, trj2, trj3, trj4]
    consensus_est, = ax.plot([], [], 'r*', ms=2)
    obs_pot, = ax.plot([], [], 'b*', ms=2)
    nbo_pt, = ax.plot([], [], 'g*', ms=2)
    trajectory_fov = [[], [], []]
    true_data, = ax.plot([], [], 'k*', ms=5)
    
    
    def init():

        """initialize animation"""
        # add FoVs
        trj1.set_data([], [])
        e = Rectangle((0, 0), 
                    100, 60,
                    angle = 0,
                    fc ='none',  
                    ec ='b', 
                    lw = 1,
                    linestyle = '-')
        ax.add_artist(e)

        
        return trj1,

    def animate(i):
        """perform animation step"""
        
        t = i*dt + dt
    
        for j in range(len(trjs)):
            trj = trjs[j]
            
            trj.set_data(x_samples[j], y_samples[j])
        
        x, y = [], []
        for j in range(len(trjs)):
            x.append(x_samples[j][i])
            y.append(y_samples[j][i])
        true_data.set_data(x, y)

        for obj in ax.findobj(match = Ellipse):
            obj.remove()
        
        for obj in ax.findobj(match = FancyArrowPatch):
            obj.remove()
        
        for obj in ax.findobj(match = Rectangle):
            
            if obj._width > 1:
                obj.remove()

        for obj in ax.findobj(match = Text):
            
            obj.set_visible(False)
    
        for block in SemanticMap["Nodes"]:
            e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                    max(block['feature']['x']) - min(block['feature']['x']), 
                    max(block['feature']['y']) - min(block['feature']['y']),
                    angle = 0,
                    fc ='none',  
                    ec ='g', 
                    lw = 2,
                    linestyle = '-')
            ax.add_artist(e)


        xpt, ypt = [], []
        x, y = [], []
        for track in agent_est[l][i]:
            mu = [track[0], track[1]]
            x.append(mu[0])
            y.append(mu[1])
            P = track[-1]
            # trackID = track[3]
            S = np.matrix(P).reshape(4,4)[0:2, 0:2]
            e = effects.make_ellipse(mu, S, gating_size, 'r')
            ax.add_artist(e)

            if np.isclose(t % cdt, 0) and not np.isclose(t, T):
                trajectory = nbo_traj(horizon, track[0:4], cdt)
                for pt in trajectory:
                    xpt.append(pt[0])
                    ypt.append(pt[1])
                # draw dashed lines
                for k in range(len(trajectory) - 1):
                    pt1 = trajectory[k]
                    pt2 = trajectory[k+1]
                    e = FancyArrowPatch((pt1[0], pt1[1]), 
                            (pt2[0], pt2[1]),
                            arrowstyle='->',
                            linewidth=2,
                            color='green')
                    ax.add_artist(e)
                
            # ax.add_artist(Text(track[0] + 2, track[1] + 2, 'ID: ' + str(trackID), fontsize = 10))
        nbo_pt.set_data(xpt, ypt)
        consensus_est.set_data(x, y)
        ax.add_artist(Text(0, 50, "t = %s" % (dt*i + dt), fontsize = 10))
        
        for j in range(len(con_agent_pos[i])):
            rec_cen = [con_agent_pos[i][j][0], con_agent_pos[i][j][1]]
            # TODO theta will be derived from csv too
            # h = sensor_para_list[j]["shape"][1][1]
            # w = sensor_para_list[j]["shape"][1][0]
            # x, y = left_bot_point(seq_agent_pos[i][j][0], seq_agent_pos[i][j][1], theta, h, w)

            e = effects.make_rectangle(rec_cen[0], rec_cen[1], con_agent_pos[i][j][2], sensor_para_list[j])
            ax.add_artist(e)
            
            # p2 = effects.vector(con_agent_pos[i][j], arrow_dia)

            # e = FancyArrowPatch((con_agent_pos[i][j][0], con_agent_pos[i][j][1]), 
            #             (p2[0], p2[1]),
            #             arrowstyle='->',
            #             linewidth=2,
            #             color=sensor_para_list[j]["color"])
            # ax.add_artist(e)
            if np.isclose(t % cdt, 0) and not np.isclose(t, T - dt):
                index = int(t//cdt)-1
                # print(t, cdt, index, T)
                try:
                    trajectory_fov[j] = fov_plan(horizon, rec_cen, cdt, policy[index][j])
                except: pass
                
            # draw dashed lines
            for k in range(len(trajectory_fov[j])):
                
                pt = trajectory_fov[j][k]
                e = effects.make_rectangle(pt[0], pt[1], con_agent_pos[i][j][2], sensor_para_list[j], 
                    linestyle='--', color=color_bar[k])
                ax.add_artist(e)
            
        x, y = [], []
        for point in obs_set[i]:
            
            x.append(point[0])
            y.append(point[1])
        obs_pot.set_data(x, y)
        

        return trj1, trj2, trj3, trj4, true_data

    ani = animation.FuncAnimation(fig, animate, frames=step_num,
                                interval=10, blit=True, init_func=init, repeat = False)
    path = os.getcwd()  
    videopath = os.path.join(path, 'video', domain, env, data2save)   
    filename = os.path.join(videopath, data2save + "_"+str(iteration)+'.mp4')
    ani.save(filename, fps=20)
    plt.close()
    print("Video saved in ", filename)

def replay_poisson(horizon, lambda0, r, iter_num, env, domain, wtp, lite=False, central_kf=False, seq=0, 
        optmethod='de', ftol=1e-3, gtol=7, traj_type = 'normal', info_gain = False, deci_Schema='dis', repeated = -1, 
        dropout_pattern = None, dropout_prob = 0.0, dv = 1e0):

    path = os.getcwd()

    data2save = simulator.filename_generator(horizon, ftol, gtol, wtp, env, seq, central_kf, optmethod, 
            deci_Schema, domain, lambda0 = lambda0, r=r, traj_type = traj_type, info_gain = info_gain, 
            dropout_pattern=dropout_pattern, dropout_prob=dropout_prob, dv=dv)

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
    # step_num = 7
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
        assert np.isclose(step_num, len(traj[0][0]))
        sensor_para_list = data["sensors"]
        dt = data["dt"]
        T = step_num * dt
        cdt = data["control_dt"]
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
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
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
        major_ticks_top = np.linspace(0,20,6)
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
                
                # ax.add_artist(Text(track[0] + 2, track[1] + 2, 'ID: ' + str(trackID), fontsize = 10))
            consensus_est.set_data(x, y)
            # ax.add_artist(Text(0, 50, "t = %s" % (dt*i + dt), fontsize = 10))
            time_display.set_text("t = %s" % round(dt*i + dt, 1))
            
            for j in range(len(con_agent_pos[i])):
                rec_cen = [con_agent_pos[i][j][0], con_agent_pos[i][j][1]]
                # TODO theta will be derived from csv too
                # h = sensor_para_list[j]["shape"][1][1]
                # w = sensor_para_list[j]["shape"][1][0]
                # x, y = left_bot_point(seq_agent_pos[i][j][0], seq_agent_pos[i][j][1], theta, h, w)

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
        if repeated > 0:
            filename = os.path.join(videopath, data2save + "_0_"+str(iteration)+'.mp4')
        ani.save(filename, fps=5)
        plt.close()
        print("Video saved in ", filename)



def Vel_polt(x_samples, y_samples, agent_est, agentid, dt, prefix, iteration, c=40):
    track_num = len(x_samples)
    step_num = len(x_samples[0])
    traj_vel = {}
    for i in range(track_num):
        traj_vel[i] = {"real": {"vx": [], "vy": [], "v": [], "t": []}, 
                        "est": {"vx": [], "vy": [], "v": [], "t": []}}
    for i in range(step_num - 1):
        t = dt * (i + 1)
        traj_k = []
        for j in range(track_num):
            pt = [x_samples[j][i+1], y_samples[j][i+1]]
            traj_k.append(pt)
            vx = (x_samples[j][i+1] - x_samples[j][i])/dt
            vy = (y_samples[j][i+1] - y_samples[j][i])/dt
            v = np.sqrt(vx**2 + vy**2)
            traj_vel[j]["real"]["t"].append(t)
            traj_vel[j]["real"]["vx"].append(vx)
            traj_vel[j]["real"]["vy"].append(vy)
            traj_vel[j]["real"]["v"].append(v)
        
        est_k = []
        for track in agent_est[agentid][i]:
            est_k.append([track[0], track[1]])

        # doing the matching
        X, Y, result = ospa.pairing(traj_k, est_k)
        row_ind = result[1]
        missing_num = 0
        m, n = len(X), len(Y)
        # np.testing.assert_array_almost_equal(m, len(row_ind))
        
        if len(traj_k) <= len(est_k):
            # x is traj, y is est
        
            for j in range(m):
                x = X[j]
                y = Y[row_ind[j]]
                error = ospa.dist(x, y)
                if error <= c:
                    vx_est = agent_est[agentid][i][row_ind[j]][2]
                    vy_est = agent_est[agentid][i][row_ind[j]][3]
                    v_est = np.sqrt(vx_est**2 + vy_est**2)
                    traj_vel[j]["est"]["t"].append(t)
                    traj_vel[j]["est"]["vx"].append(vx_est)
                    traj_vel[j]["est"]["vy"].append(vy_est)
                    traj_vel[j]["est"]["v"].append(v_est)
        
        else:
            # y is traj_k
            for j in range(m):
                x = X[j]
                y = Y[row_ind[j]]
                error = ospa.dist(x, y)
                if error <= c:
                    vx_est = agent_est[agentid][i][j][2]
                    vy_est = agent_est[agentid][i][j][3]
                    v_est = np.sqrt(vx_est**2 + vy_est**2)
                    traj_vel[row_ind[j]]["est"]["t"].append(t)
                    traj_vel[row_ind[j]]["est"]["vx"].append(vx_est)
                    traj_vel[row_ind[j]]["est"]["vy"].append(vy_est)
                    traj_vel[row_ind[j]]["est"]["v"].append(v_est)

    for i in range(track_num):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
        
        axs[0].plot(traj_vel[i]["real"]["t"], traj_vel[i]["real"]["vx"], color='r')
        axs[0].plot(traj_vel[i]["est"]["t"], traj_vel[i]["est"]["vx"], color='b')
        axs[0].set_title("vx comparing")
        axs[0].legend(['real', 'estimation'])

        axs[1].plot(traj_vel[i]["real"]["t"], traj_vel[i]["real"]["vy"], color='r')
        axs[1].plot(traj_vel[i]["est"]["t"], traj_vel[i]["est"]["vy"], color='b')
        axs[1].set_title("vy comparing")
        axs[1].legend(['real', 'estimation'])

        axs[2].plot(traj_vel[i]["real"]["t"], traj_vel[i]["real"]["v"], color='r')
        axs[2].plot(traj_vel[i]["est"]["t"], traj_vel[i]["est"]["v"], color='b')
        axs[2].set_title("v comparing")
        axs[2].legend(['real', 'estimation'])
            
        
        # filename = os.path.join(path, 'pics', 'Parking_slowmotion_combined.png')
        filename = os.path.join(os.getcwd(), "video", "batch", prefix + "_"+str(iteration)+"_trj_"+str(i)+'_vel.png')
        plt.savefig(filename, dpi=400)
        plt.close()


def nbo_traj(horizon, x, cdt):
    SampleDt = cdt
    SampleNum = int(horizon // cdt)
    SampleF = np.matrix([[1, 0, SampleDt, 0],
                            [0, 1, 0, SampleDt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    trajectory = []
    x = np.array(x)
    
    for k in range(SampleNum):
        # sample one place
        
        x = np.asarray(np.dot(SampleF, x))[0]
        trajectory.append(list(x)[0:2])
    return trajectory

def fov_plan(horizon, x, cdt, policy):
    
    SampleNum = int(horizon // cdt)
    
    trajectory = []

    for k in range(SampleNum):
        action = policy[k]
        x[0] += action[0] * cdt
        x[1] += action[1] * cdt
        
        trajectory.append(copy.deepcopy(x))
    
    return trajectory


def main():
    sim_num = 1
    MCSnum, horizons = 50, [3, 5]      # 3.0, 5.0, 8.0
    horizon = 3
    deci_Schema = 'sma'             # 'cen' or 'dis' or 'decPOMDP' or 'pma'
    domain = 'nbo'                  # 'nbo' or 'MonteCarloRollout'
    ftol = 5e-4                     # 1e-3   #5e-3  # 5e-4 for decentralized, 2.5e-4 for centralized
    gtol = 50                       # 50 or 75 for centralized
    opt_step = -1
    useSemantic = True              # False for poisson?
    sigma = False
    lite = True                     # use max consensus in NBO
    ckf = True                      # what sensor fusion to use in NBO, centralized kf or consensus
    method = 'pso'                  # 'pso' or 'de' or 'discrete'
    case = 'poisson'                # 'poisson' or 'parksim' or 'simple1' or 'simple2'
    wtp = False
    ibr_num = 4
    lambda0 = 1e-3
    lambda0_list = [1e-3, 3e-3, 5e-3]     # tree density   [1e-3, 3e-3, 5e-3] 
    radius = 5                           # radius of tree
    traj_type = 'normal'            # 'straight' or 'normal' or 'static'
    info_gain = 'trace_sum'         # 'trace_sum', 'info_gain', 'info_acc', 'log_det', 'cov_gain'
    repeat_time = 20                 # repeat on same map, will do repeat when >1
    dropout_pattern = None  
    dropout_prob = 0.1
    dv = 5e-1
    seq = 0
    # if case in ['poisson', 'simple1', 'simple2']:
    replay_poisson(horizon, lambda0, radius, sim_num, case, domain, wtp, lite=lite, central_kf=ckf, seq=seq, 
        optmethod=method, ftol=ftol, gtol=gtol, traj_type = traj_type, info_gain = info_gain, deci_Schema=deci_Schema,
         dropout_pattern=dropout_pattern, dropout_prob=dropout_prob, dv = dv, repeated=repeat_time)
    return

    if domain == 'nbo':
        if isCentralized:
            if wtp:
                data2save = 'cen_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
            else:
                data2save = 'cen_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
        else:
            if wtp:
                data2save = 'dis_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq)
            else:
                data2save = 'dis_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq)
    elif domain == 'MonteCarloRollout':
        if isCentralized:
            if wtp:
                data2save = 'cen_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
            else:
                data2save = 'cen_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
        else:
            if wtp:
                data2save = 'dis_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq) + '_wider'
            else:
                data2save = 'dis_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq)
        data2save += '_MC_' + str(MCSnum)
    elif domain == 'ibr':
        if wtp:
            data2save = 'parallel_wtp_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
        else:
            data2save = 'parallel_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
    if ckf:
        data2save += '_ckf'
    
    path = os.getcwd()
    # load sensor model
    if case == 'parking':
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    else:
        filename = os.path.join(path, 'data', 'env', 'simplepara.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    dataPath = os.path.join(path, 'data', 'result', domain, data['env'], data2save)

    videopath = os.path.join(path, 'video', domain, data['env'], data2save)
    try:
        os.mkdir(videopath)
        print("path %s made"%videopath)
    except OSError:
        print("path %s not made"%videopath)
        pass

    for i in range(0, sim_num):
            
        replay(dataPath, data2save, domain, True, isbatch=True, ftol=ftol, gtol=ftol,
            iteration=i, isCentral=isCentralized, agentid=0, horizon=horizon, seq=0, video=False, lite=lite, 
            central_kf=ckf, case=case)
            


if __name__ == "__main__":
    main()