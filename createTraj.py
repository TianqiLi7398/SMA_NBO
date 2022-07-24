from utils.sample import sampling
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
import json
from matplotlib.patches import Rectangle
from utils.effects import effects
import os, sys
import utils.curves

def gen_curves():

    path = os.getcwd()
    # y = -3768.515 + 498.364x - 26.62405x**2 + 0.71544x**3 - 0.00958x**4 + 0.00005x**5
    coeff1 = [-3768.515, 498.364, - 26.62405, 0.7154399, - 0.009576507, + 0.00005087052]

    # y = 4.68718 + 0.36366x - 0.01099x**2 + 0.00009x**3
    coeff2 = [4.68718, 0.36366, - 0.01099, 0.00009]

    
    # y = 26.84071 + 0.1502x - 0.00128x**2
    coeff3 = [26.84071, 0.1502, - 0.00128]

    # y = 46.86611 + 0.40161x - 0.00596x**2 + 0.00002x**3
    coeff4 = [46.86611, 0.40161, - 0.00596, 0.00002]

    # y = 0.5883768 + 3.481813*x - 0.1304584*x^2 + 0.00187602*x^3 - 0.000008747492*x^4
    coeff3 = [0.5883768, 3.481813, - 0.1304584, 0.00187602, - 0.000008747492]

    # y = 71.28861 - 2.342178*x + 0.09136746*x^2 - 0.001396769*x^3 + 0.00000681985*x^4
    coeff4 = [71.28861, - 2.342178, 0.09136746, - 0.001396769, 0.00000681985]

    directions = [1, -1, -1, 1]
    x0s = [27, 42, 55, 0]
    directions = [1, -1, 1, 1]
    x0s = [27, 42, 23, 10]

    coeff_list = [coeff1, coeff2, coeff3, coeff4]

    # now let's make the trajectories based on the polynomial functions, and discretize
    # the walk step based on levy walk feature
    minstep, maxstep = 0.2, 0.6
    sampleNum = 200

    filename = os.path.join(path, 'data', 'env', 'parking_map.json')
    with open(filename) as json_file:
        SemanticMap = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    x_samples, y_samples = [], []
    for i in range(len(coeff_list)):
        coeff = coeff_list[i]
        y0 = sampling.poly(x0s[i], coeff)
        x_, y_, r_ = sampling.discretize_levy(coeff, x0s[i], y0, sampleNum, minstep, maxstep, directions[i])
        x_samples.append(x_)
        y_samples.append(y_)
    
    # display them in a plot
    # for i in range(len(x_samples)):
    #     plt.plot(x_samples[i], y_samples[i])

    # filename = os.path.join(path, 'pics', 'ParkingTraj3.png')
    # plt.savefig(filename)
    # plt.close()
    x_samples, y_samples = [], []
    dt = .2
    # x_, y_ = [], []
    # for i in range(sampleNum):
    #     t = i * dt
    #     x_.append(2 + t)
    #     y_.append(2 + t)
    # x_samples.append(x_)
    # y_samples.append(y_)

    x_, y_ = [], []
    for i in range(sampleNum):
        
        x_.append(27 + 0.1185 * i)
        y_.append(0.97 + 0.325 * i)
    x_samples.append(x_)
    y_samples.append(y_)

    x_, y_ = [], []
    for i in range(sampleNum):
        t = i * dt
        x_.append(42 - 0.325 * i)
        y_.append(7.2 + 0.093 * i)
    x_samples.append(x_)
    y_samples.append(y_)

    x_, y_ = [], []
    for i in range(sampleNum):
        t = i * dt
        x_.append(23 + 0.339 * i)
        y_.append(32 )
    x_samples.append(x_)
    y_samples.append(y_)

    x_, y_ = [], []
    for i in range(sampleNum):
        t = i * dt
        x_.append(10 + 0.33 * i)
        y_.append(55.6 )
    x_samples.append(x_)
    y_samples.append(y_)

    # x_, y_ = [], []
    # for i in range(sampleNum):
    #     t = i * dt
    #     x_.append(2 + 1.414 * t)
    #     y_.append(-2)
    # x_samples.append(x_)
    # y_samples.append(y_)
    dx = 0.5
    zigzager = utils.curves.zigzag(dx)
    for i in range(len(x_samples)):
        x_samples[i], y_samples[i] = zigzager.process(x_samples[i], y_samples[i])
        
    ParkingTraj = [x_samples, y_samples]
    filename = os.path.join(path, "data", 'env', "traj_zigzag.json")
    with open(filename, 'w') as outfiles:
        json.dump(ParkingTraj, outfiles, indent=4)

    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-2 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-2 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))
    
    # add obstacles

    for block in SemanticMap["Nodes"]:
        e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                max(block['feature']['x']) - min(block['feature']['x']), 
                max(block['feature']['y']) - min(block['feature']['y']),
                angle = 0,
                fc ='g',  
                ec ='g', 
                lw = 2,
                linestyle = '-')
        ax.add_artist(e)
    
    for i in range(len(data["sensors"])):
        rec_cen = [data["sensors"][i]["position"][0], data["sensors"][i]["position"][1]]
        # TODO theta will be derived from csv too
        # h = sensor_para_list[j]["shape"][1][1]
        # w = sensor_para_list[j]["shape"][1][0]
        # x, y = left_bot_point(seq_agent_pos[i][j][0], seq_agent_pos[i][j][1], theta, h, w)

        e = effects.make_rectangle(rec_cen[0], rec_cen[1], data["sensors"][i]["position"][2], data["sensors"][i])
        ax.add_artist(e)
    
    for i in range(len(x_samples)):
        plt.plot(x_samples[i], y_samples[i])
        
    plt.tight_layout()
    
    filename = os.path.join(path, 'pics', 'traj_zigzag.png')
    plt.savefig(filename)
    plt.close()

    # animation of these trajectories.
    # make video of the result
    # global ax, fig
    # fig = plt.figure()
    # #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                     xlim=(-2 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
    #                     ylim=(-2 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))

    # trj1, = ax.plot([], [], 'ko', ms=2)
    # trj2, = ax.plot([], [], 'ro', ms=2)
    # trj3, = ax.plot([], [], 'go', ms=2)
    # trj4, = ax.plot([], [], 'bo', ms=2)
    
    # trjs = [trj1, trj2, trj3, trj4]
    
    


    # def init():

    #     """initialize animation"""
    #     # add FoVs
    #     trj1.set_data([], [])
    #     e = Rectangle((0, 0), 
	# 				100, 60,
	# 				angle = 0,
	# 				fc ='none',  
	# 				ec ='b', 
	# 				lw = 1,
	# 				linestyle = '-')
    #     ax.add_artist(e)

    #     for block in SemanticMap["Nodes"]:
    #         e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
	# 				max(block['feature']['x']) - min(block['feature']['x']), 
    #                 max(block['feature']['y']) - min(block['feature']['y']),
	# 				angle = 0,
	# 				fc ='none',  
	# 				ec ='g', 
	# 				lw = 2,
	# 				linestyle = '-')
    #         ax.add_artist(e)
        
        
    #     return trj1,

    # def animate(i):
    #     """perform animation step"""
       
    #     for j in range(len(trjs)):
    #         trj = trjs[j]
    #         x, y = [], []
    #         x.append(x_samples[j][i])
    #         y.append(y_samples[j][i])
            
    #         trj.set_data(x, y)

        

    #     return trj1, trj2, trj3, trj4

    # ani = animation.FuncAnimation(fig, animate, frames=sampleNum,
    #                             interval=10, blit=True, init_func=init, repeat = False)
    
    # filename = os.path.join(path, "video", 'ParkingTraj3.mp4')
    # ani.save(filename, fps=20)
    
    # plt.close()

def gen_static():

    path = os.getcwd()
    
    x0s = [27, 42, 5, 10]
    y0s = [-1, 7, 32, 60]

    sampleNum = 100

    filename = os.path.join(path, 'data', 'env', 'parking_map.json')
    with open(filename) as json_file:
        SemanticMap = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    x_samples, y_samples = [], []
    for i in range(len(x0s)):
        
        x_ = [x0s[i]] * sampleNum
        y_ = [y0s[i]] * sampleNum
        x_samples.append(x_)
        y_samples.append(y_)
    
    # display them in a plot
    # for i in range(len(x_samples)):
    #     plt.plot(x_samples[i], y_samples[i])

    # filename = os.path.join(path, 'pics', 'ParkingTraj3.png')
    # plt.savefig(filename)
    # plt.close()

    ParkingTraj = [x_samples, y_samples]
    filename = os.path.join(path, "data", 'env', "traj_static.json")
    with open(filename, 'w') as outfiles:
        json.dump(ParkingTraj, outfiles, indent=4)

    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-2 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-2 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))
    
    # add obstacles

    for block in SemanticMap["Nodes"]:
        e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                max(block['feature']['x']) - min(block['feature']['x']), 
                max(block['feature']['y']) - min(block['feature']['y']),
                angle = 0,
                fc ='g',  
                ec ='g', 
                lw = 2,
                linestyle = '-')
        ax.add_artist(e)
    
    for i in range(len(data["sensors"])):
        rec_cen = [data["sensors"][i]["position"][0], data["sensors"][i]["position"][1]]
        # TODO theta will be derived from csv too
        # h = sensor_para_list[j]["shape"][1][1]
        # w = sensor_para_list[j]["shape"][1][0]
        # x, y = left_bot_point(seq_agent_pos[i][j][0], seq_agent_pos[i][j][1], theta, h, w)

        e = effects.make_rectangle(rec_cen[0], rec_cen[1], data["sensors"][i]["position"][2], data["sensors"][i])
        ax.add_artist(e)
    
    for i in range(len(x_samples)):
        plt.plot(x_samples[i], y_samples[i], '*')
        
    plt.tight_layout()
    
    filename = os.path.join(path, 'pics', 'traj_static.png')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    gen_static()

