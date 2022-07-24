'''
Author: Tianqi Li
Institution: Mechanical Engineering, TAMU
Date: 2020/09/18

This file defines the class agent for decentralized JPDA algorithm, which is 
based on Consensus of Information (CI) for sensor fusion,
Based on the decentralized agent, it makes the base policy of moving to the worst tracks.

Properities:
    1. track-to-track assign function: self.match, which is based on Hungarian algorithm;
    2. TODO exchange information procotol: each agent generates the local message regardless of 
    whether there is an observation associated with the track, which will be enhanced later;
    3. TODO some noisy tracks are confirmed after JDPA, which is supposed to be enhanced later;
'''

from utils.msg import Agent_basic, Single_track, Info_sense
import numpy as np
from utils.jpda_consensus import dec_jpda


class dec_agent(dec_jpda):

    def __init__(self, sensor_para_list, agentid, dt, cdt, L0 = 5, v = 5, 
        isObsdyn__=True, isRotate = False, NoiseResistant = True, IsStatic = False,
        isVirtual = False,t0 = 0, SemanticMap=None, OccupancyMap=None, sigma = False):
        dec_jpda.__init__(self, sensor_para_list, agentid, dt, L = L0, 
                    isObsdyn_=isObsdyn__, NoiseResistant =NoiseResistant, 
                    isVirtual=isVirtual, t0 = t0, SemanticMap=SemanticMap,
                    OccupancyMap=OccupancyMap, sigma=sigma, IsStatic = IsStatic)
        self.dm = sensor_para_list[agentid]["dm"]      # the diameter limit for sensor's local base policy
        self.dm = 40
        self.v = [0,0]
        self.v_bar = sensor_para_list[agentid]["v"]  
        self.isRotate = isRotate
        self.cdt = cdt
    
    def predictPos(self, x):
        return [x[0] + self.cdt * x[2], x[1] + self.cdt * x[3]]

    def base_policy(self):
        
        # pick up the base policy for 
        
        worst_x = self.sensor_para["position"][0:2]
        worst_P = -1
        # find the track inside FoV with highest P
        if self.isVirtual:
            for i in range(len(self.tracker)):
                x = np.squeeze(np.asarray(self.tracker[i].x_k_k))
                tar_pos = self.predictPos(x)
                
                # another certia is inside the fov
                # if self.tracker.isInFoV(tar_pos):
                if self.euclidan_dist(tar_pos, self.sensor_para["position"][0:2]) < self.dm:
                    P = np.trace(self.tracker[i].P_k_k)
                    if P > worst_P:
                        worst_P = P
                        worst_x = x
        else:
            for i in range(len(self.local_track["id"])):
                
                tar_pos = self.predictPos(self.local_track["infos"][i][0].x)
                
                # another certia is inside the fov
                # if self.tracker.isInFoV(tar_pos):
                if self.euclidan_dist(tar_pos, self.sensor_para["position"][0:2]) < self.dm:

                    P_k_k = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)
                    P = np.trace(P_k_k)
                    if P > worst_P:
                        worst_P = P
                        worst_x = self.local_track["infos"][i][0].x[0:2]
            
        # once find the worst track, move towards it
        dx = worst_x[0] - self.sensor_para["position"][0]
        dy = worst_x[1] - self.sensor_para["position"][1]
        d = np.sqrt(dx**2 + dy **2)
        
        
        if d > 0:
            self.v = [round(self.v_bar* dx / d, 2), round(self.v_bar * dy / d, 2)]
        else:
            self.v = [0, 0]
        
        # print(self.id, "'s target is ", worst_x)
        return
    
    def updateAngle(self):
        if abs(self.v[0]) + abs(self.v[1]) > 0:
            if np.isclose(self.v[0], 0):
                theta = np.sign(self.v[1]) * np.pi/2
            else:
                theta = np.arctan(self.v[1]/self.v[0])
                if theta > 0:
                    if self.v[0] < 0:
                        theta += np.pi
                else:
                    if self.v[0] < 0:
                        theta += np.pi
            self.sensor_para["position"][2] = theta


    def dynamics(self, dt):
        # update the position of sensors
        
        self.sensor_para["position"][0] += self.v[0] * dt
        self.sensor_para["position"][1] += self.v[1] * dt

        # update angle
        if self.isRotate:
            self.updateAngle()
        
    