import numpy as np
import os, sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.effects import effects
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
from matplotlib.text import Text
import copy

class debug_video:

    @staticmethod
    def pso_propagate(best_pos_list, fov, nominal_trajs, SemanticMap, 
        cur_position, horizon, value_history, opt_t, id_, OccupancyMaps):
        agent_position_list = []
        
        SemanticMap["canvas"] = [
            [
                -40,
                100
            ],
            [
                -20,
                80
            ]
        ]
        for action in best_pos_list:
            a_traj = {"x":[cur_position[0]], "y": [cur_position[1]]}
            
            cur_pos = copy.deepcopy(cur_position)
            for t in range(horizon):
                cur_pos[0] += action[2 * t] * np.cos(action[2 * t +1])
                cur_pos[1] += action[2 * t] * np.sin(action[2 * t +1])
                
                a_traj["x"].append(cur_pos[0])
                a_traj["y"].append(cur_pos[1])
            agent_position_list.append(a_traj)
        
        # generate video    
        # debug_video.generate_video()
        nbo_traj = []
        num_traj = len(nominal_trajs[0])
        for i in range(num_traj):
            a_nbo_traj = {"x":[], "y": []}
            nbo_traj.append(a_nbo_traj)
        for t in range(horizon):
            trajs = nominal_trajs[t]
            for i in range(num_traj):
                nbo_traj[i]["x"].append(trajs[i][0])
                nbo_traj[i]["y"].append(trajs[i][0])
        
        debug_video.generate_video(agent_position_list, nbo_traj, SemanticMap, 
            value_history, opt_t, id_, OccupancyMaps)
        return
                

    
    @staticmethod
    def generate_video(agent_position_list, nominal_trajs, SemanticMap, value_history, 
        opt_t, id_, OccupancyMaps):
        color_bar = ['r', 'b', "y", 'g', 'orange', '#6e9ce6', '#8daee3']
        
        global ax, fig
        fig = plt.figure()
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-10 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                            ylim=(-20 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))

        trj1, = ax.plot([], [], 'k', ms=2)
        trj2, = ax.plot([], [], 'r', ms=2)
        # trj3, = ax.plot([], [], 'g', ms=2)
        # trj4, = ax.plot([], [], 'b', ms=2)
        # targets = []
        # for i in range(len(nominal_trajs)):
        #     tr, = ax.plot([], [], 'r', ms=2)
        #     targets.append(tr)
        targets_scatter = ax.scatter([], [])
        
        # trjs = [trj1, trj2, trj3, trj4]
        consensus_est, = ax.plot([], [], 'r*', ms=2)
        obs_pot, = ax.plot([], [], 'b*', ms=2)
        nbo_pt, = ax.plot([], [], 'g*', ms=2)
        trajectory_fov = [[], [], []]
        
        
        def init():

            """initialize animation"""
            # add FoVs
            trj1.set_data([], [])

            
            return trj1,

        def animate(i):
            """perform animation step"""
            
            # planned agent position
            trj1.set_data(agent_position_list[i]['x'], agent_position_list[i]['y'])

            # targets position
            X, Y = [], []
            
            for l in range(len(nominal_trajs)):
                # targets[l].set_data(nominal_trajs[l]["x"], nominal_trajs[l]["y"])
                X += nominal_trajs[l]["x"]
                Y += nominal_trajs[l]["y"]
            
            ax.scatter(X, Y, s=1)
            # for j in range(len(trjs)):
            #     trj = trjs[j]
                
            #     trj.set_data(x_samples[j], y_samples[j])

            # for obj in ax.findobj(match = Ellipse):
            #     obj.remove()
            
            # for obj in ax.findobj(match = FancyArrowPatch):
            #     obj.remove()
            
            # for obj in ax.findobj(match = Rectangle):
                
            #     if obj._width > 1:
            #         obj.remove()

            for obj in ax.findobj(match = Text):
                
                obj.set_visible(False)
            
            # add centers
            try:
                for j in range(len(OccupancyMap["centers"][0])):
                    
                    e = Circle((OccupancyMap['centers'][0][j], OccupancyMap['centers'][1][j]), 
                        radius = radius,
                        fc ='none',  
                        ec ='b', 
                        lw = 1,
                        linestyle = '-')
                    ax.add_artist(e)
            except:  pass
        
            

            ax.add_artist(Text(0, 0, "val = %s" % value_history[i], fontsize = 10))
            

            return trj1, trj2#, trj3, trj4

        ani = animation.FuncAnimation(fig, animate, frames= len(agent_position_list),
                                    interval=10, blit=True, init_func=init, repeat = False)
        path = os.getcwd()
        
        videopath = os.path.join(path, 'video', 'pso_test')
        try:
            os.mkdir(videopath)
        except OSError:
            pass
        
        filename = os.path.join(videopath, 'pso_time=%s_agent%s.mp4'% (opt_t, id_))
        # mywriter = animation.FFMpegWriter()
        
        plt.show()
        ani.save(filename, fps=5)
        # ??????
        plt.close()