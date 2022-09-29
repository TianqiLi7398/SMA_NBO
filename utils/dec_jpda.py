'''
Author: Tianqi Li
Institution: Mechanical Engineering, TAMU
Date: 2020/09/18

This file defines the class agent for decentralized JPDA algorithm, which is 
based on Consensus of Information (CI) for sensor fusion,


Properities:
    1. track-to-track assign function: self.match, which is based on Hungarian algorithm;
    2. exchange information procotol: each agent generates the local message only if there is observation
    on the track, to avoid spurious tracking/false alarm;
    
'''

from utils.msg import Agent_basic, Single_track, Info_sense
import numpy as np
from utils.jpda_node import jpda_single
from utils.util import LinearKF, dynamic_kf, track, vertex
from scipy.optimize import linear_sum_assignment   # Hungarian alg, minimun bipartite matching
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import Any, List, Tuple

class dec_jpda:
    def __init__(
            self, 
            sensor_para_list: dict, 
            agentid: int, 
            dt: float, 
            L: int = 5, 
            isObsdyn_: bool = False, 
            NoiseResistant: bool = True, 
            isVirtual: bool = False,
            t0: float = 0.0, 
            SemanticMap: Any = None, 
            OccupancyMap: Any = None, 
            IsStatic: bool = False
        ):
        self.id = agentid
        self.sensor_para = sensor_para_list[agentid]
        self.sensor_para_list = sensor_para_list
        self.neighbor = {"id": [], "pos": []}
        self.dc = self.sensor_para["dc"]
        self.dt = dt
        self.pre_time = -1
        self.L = L  # consensus steps
        self.l = 0  # current conensus iteration
        self.comm_neighbor = []
        self.infos_received = []
        # self.info_msg contains all tracks with observations
        self.info_msg = Info_sense()
        # self.local_track maintains all active tracks
        self.local_track = { "infos": [], "id": []}
        
        self.isObsdyn = isObsdyn_
        self.isVirtual = isVirtual
        if self.isVirtual:
            # the tracker is a list of KF objects
            self.tracker = []
        else:
            if self.isObsdyn:
                self.tracker = jpda_single(self.dt, self.sensor_para,
                    isSimulation = True, 
                    isObsdyn=isObsdyn_,
                    ConfirmationThreshold = self.sensor_para["ConfirmationThreshold"],
                    DeletionThreshold = self.sensor_para["DeletionThreshold"], 
                    t0=t0, IsStatic = IsStatic)
                    
            else:
                self.tracker = jpda_single(self.dt, self.sensor_para,
                    isSimulation = True, 
                    isObsdyn=isObsdyn_, 
                    ConfirmationThreshold = self.sensor_para["ConfirmationThreshold"],
                    DeletionThreshold = self.sensor_para["DeletionThreshold"], 
                    t0=t0, IsStatic = IsStatic)
        self.default_bbsize = [1,1]
        # a threshold d0 for metrics
        self.d0 = self.sensor_para["d0"]
        self.t = t0
        self.NoiseResistant = NoiseResistant
        self.SemanticMap = SemanticMap
        self.OccupancyMap = OccupancyMap
        self.useSemantic = (self.SemanticMap is not None) 
        self.useOccupancy = (self.OccupancyMap is not None)
        self.Obstacles = []
        if self.useSemantic:
            for obstacle in self.SemanticMap["Nodes"]:
                corners = []
                for i in range(len(obstacle['feature']['x'])):
                    corners.append((obstacle['feature']['x'][i], obstacle['feature']['y'][i]))
                self.Obstacles.append(Polygon(corners))
        
        
        self.w0 = .5

    def basic_info(self, message_list: List[Agent_basic]):
        self.neighbor =  {"id": [], "pos": []}

        for message in message_list:

            # update agents position
            self.sensor_para_list[message.id]["position"][0] = message.x
            self.sensor_para_list[message.id]["position"][1] = message.y
            self.sensor_para_list[message.id]["position"][2] = message.theta

            # here message info type is Agent_basic
            if message.id in self.neighbor["id"] or message.id == self.id:
                continue
            distance = (message.x - self.sensor_para["position"][0])**2 + (message.y - self.sensor_para["position"][1])**2
            
            if distance < self.dc**2:
                self.neighbor["id"].append(message.id)
                self.neighbor["pos"].append([message.x, message.y])
        
        return

    def init_vertex(self, track: track):
        # when jpda returns a certain track, we need to utlize the 
        # track in track-to-track matching func, thus we need to activate
        # the track 
        track.agent_id = self.id 

    def pre_msg(self, track_result) -> Info_sense:
        # prepare the new message, after 1 sensor update and JPDA
        self.local_track = { "infos": [], "id": [], "incomes": []}
        self.info_msg = Info_sense()
        self.info_msg.id = self.id 
        self.info_msg.l = 0
        for track in track_result:
            if track.id < 0:
                # initialize the track to agent
                self.init_vertex(track)

            newtrack = Single_track()
            newtrack.track_id = track.id
            x = track.kf.x_k_k
            P = track.kf.P_k_k            
            Sigma = np.linalg.inv(P)
            q = np.dot(Sigma, x)
            newtrack.x = x.flatten().tolist()[0]
            newtrack.q = q.flatten().tolist()[0]
            newtrack.P = P.flatten().tolist()[0]
            newtrack.Sigma = Sigma.flatten().tolist()[0]
            newtrack.isObs = track.history[-1]   # critical step to represent local obs
            # save it to the initial structure, in preparation of updating
            if track.history[-1] == 1:
                # only send tracks which has observations
                self.info_msg.tracks.append(newtrack)
            
            self.local_track["infos"].append([newtrack])
            self.local_track["id"].append(newtrack.track_id)
            #  self.local_track["incomes"].append(0)
        
        self.l += 1

        return copy.deepcopy(self.info_msg)

    def cal_R(self, z: List[float], xs: List[float]) -> np.ndarray:
        dx = z[0] - xs[0]
        dy = z[1] - xs[1]
        if np.isclose(dx, 0):
            theta = np.sign(dy) * 0.5 * np.pi - xs[2]
        else:
            theta = np.arctan(dy / dx) - xs[2]
        r = max(self.sensor_para["r0"], np.sqrt(dx**2 + dy**2))
        G = np.matrix([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        M = np.diag([0.1 * r, 0.1 * np.pi * r])
        R = self.sensor_para["quality"] * np.dot(np.dot(G, M), G.T)
        
        return R

    def isInFoV(self, z: List[float]) -> bool:
        # give z = [x, y], check if it's inside FoV
        return self.tracker.isInFoV(z)

    def VirtualUpdate(self, 
            z_k: List[float], 
            semantic: List[bool]
        ) -> Info_sense:
        # Do KF update in the order of z_k
        if self.isObsdyn:
            for i in range(len(z_k)):
                self.tracker[i].predict()
                if semantic[i]:
                    # target in occlusion, skip kf update
                    self.tracker[i].isUpdated = False
                    self.tracker[i].x_k_k = copy.deepcopy(self.tracker[i].x_k_k_min)
                    self.tracker[i].P_k_k = copy.deepcopy(self.tracker[i].P_k_k_min)
                else:
                    z = z_k[i]
                    # if z is inside the fov, just update it
                    if self.isInFoV(z):
                        self.tracker[i].update(z, self.sensor_para["position"], self.sensor_para["quality"])
                    else:
                        self.tracker[i].isUpdated = False
                        self.tracker[i].x_k_k = copy.deepcopy(self.tracker[i].x_k_k_min)
                        self.tracker[i].P_k_k = copy.deepcopy(self.tracker[i].P_k_k_min)

        else:
            for i in range(len(z_k)):
                self.tracker[i].predict()
                if semantic[i]:
                    self.tracker[i].isUpdated = False
                    self.tracker[i].x_k_k = copy.deepcopy(self.tracker[i].x_k_k_min)
                    self.tracker[i].P_k_k = copy.deepcopy(self.tracker[i].P_k_k_min)
                else:
                    z = z_k[i]
                    if self.isInFoV(z):
                        self.tracker[i].update(z)
                    else:
                        self.tracker[i].isUpdated = False
                        self.tracker[i].x_k_k = copy.deepcopy(self.tracker[i].x_k_k_min)
                        self.tracker[i].P_k_k = copy.deepcopy(self.tracker[i].P_k_k_min)
        
        self.local_track = { "infos": [], "id": [], "incomes": []}
        self.info_msg = Info_sense()
        self.info_msg.id = self.id 
        self.info_msg.l = 0
        for i in range(len(self.tracker)):
            track = self.tracker[i]
            newtrack = Single_track()
            newtrack.isObs = track.isUpdated
            newtrack.track_id = i
            x = track.x_k_k
            P = track.P_k_k
            Sigma = np.linalg.inv(P)
            q = np.dot(Sigma, x)
            newtrack.x = x.flatten().tolist()[0]
            newtrack.q = q.flatten().tolist()[0]
            newtrack.P = P.flatten().tolist()[0]
            newtrack.Sigma = Sigma.flatten().tolist()[0]
            # save it to the initial structure, in preparation of updating
            self.info_msg.tracks.append(newtrack)
            
            self.local_track["infos"].append([newtrack])
            self.local_track["id"].append(i)
            #  self.local_track["incomes"].append(0)
        
        self.l = 1

        return copy.deepcopy(self.info_msg)

    def sigma_obs(
            self, 
            z: np.ndarray, 
            P: np.ndarray, 
            n: int=2
        ) -> Tuple[list, list]:
        '''
        generate sigma points on the eigen basis direction, from 
        https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
        '''
        
        z = [z[0,0], z[1, 0]]
        z_list = [z]
        w_list = [self.w0]
        wj = (1 - self.w0)/(2*n)
        L, V = np.linalg.eigh(P)
        u = V[0] / np.linalg.norm(V[0])
        if np.isclose(u[0,0], 0):
            angle = np.sign(u[0,1]) * 0.5 * np.pi
        else:
            angle = np.arctan(u[0, 1]/u[0,0])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        S = np.sqrt(L)
        for dim in range(n):
            origin_vector = np.zeros(n)
            origin_vector[dim] = 1
            origin_vector *= np.sqrt((1-self.w0) * S[dim]/2)
            p2 = np.dot(R, origin_vector)
            new_z = [z[0] + p2[0], z[1] + p2[1]]
            z_list.append(new_z)
            w_list.append(wj)
            new_z = [z[0] - p2[0], z[1] - p2[1]]
            z_list.append(new_z)
            w_list.append(wj)
        
        return z_list, w_list


    def euclidan_dist(self, p1: List[float], p2: List[float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def sim_detection_callback(self, 
            obs_raw: List[float], 
            t: List[float], 
            semantic: Any = None
        ) -> Info_sense:
        
        z_k = copy.deepcopy(obs_raw)
        self.t = t
        # simulated callback after receiving sensor's info
        size_k = []
        if self.isVirtual:
            # only KF updates without data association, happens in planning
            return self.VirtualUpdate(z_k, semantic)

        if self.isObsdyn:
            for i in range(len(z_k)):
                R = self.cal_R(z_k[i], self.sensor_para["position"]) * 0.5
                
                vk = np.random.multivariate_normal(mean=np.zeros(2), cov=R, size=1)
                
                z_k[i][0] += vk[0, 0]
                z_k[i][1] += vk[0, 1]
                size_k.append([1, 1, 2, 0.1])
            z_k, size_k = self.tracker.obs_fov(z_k, size_k, self.sensor_para["position"])
            z_k = self.SimpleSemantic(z_k)
            track_result = self.tracker.track_update(t, self.dt, z_k, size_k, self.sensor_para["position"]) # list of [id, positon]
                    
        else:
            for i in range(len(z_k)):
                vk = np.random.multivariate_normal(mean=np.zeros(2), cov=self.tracker.R, size=1)
                z_k[i][0] += vk[0, 0]
                z_k[i][1] += vk[0, 1]
                size_k.append([1, 1, 2, 0.1])
            z_k, size_k = self.tracker.obs_fov(z_k, size_k, self.sensor_para["position"])
            z_k = self.SimpleSemantic(z_k)
            track_result = self.tracker.track_update(t, self.dt, z_k, size_k, self.sensor_para["position"]) # list of [id, positon]

        return self.pre_msg(track_result), z_k, size_k

    def isInsideOcclusion(self, pt: List[float]) -> bool:
        '''
        check if the pt is inside the Occlusion
        '''
        IsInside = False
        if self.useSemantic:
            pt = Point(pt[0], pt[1])
            for obstacle in self.Obstacles:
                if obstacle.contains(pt): return True
        elif self.useOccupancy:
            for i in range(len(self.OccupancyMap['centers'][0])):
                x, y = self.OccupancyMap['centers'][0][i], self.OccupancyMap['centers'][1][i]
                if self.euclidan_dist([x, y], pt) < self.OccupancyMap['r']: return True
                
        return IsInside

    def SimpleSemantic(self, z_k: List[List[float]]) -> List[List[float]]:
        # filter targets inside the area of obstacles (like tree shadows)
        output = []
        for z in z_k:
            if not self.isInsideOcclusion([z[0], z[1]]):
                output.append(z)
        return output

    def grab_info_list(self, msg_list: List[Info_sense]):
        for msg in msg_list:

            self.grab_info(msg)

    def grab_info(self, msg: Info_sense):
        # msg type: Info_sense
        # in callback function when receive info from other nodes,
        # append the info
        if msg.id not in self.neighbor["id"]:
            return
        if msg.l != self.info_msg.l:
            print("iteration does not match!!")
            print(self.id, msg.l, self.l)
            # not current iteration
            return
        if msg.id in self.comm_neighbor:
            # already has the info
            return
        new_msg = copy.deepcopy(msg)
        self.infos_received.append(new_msg)
        self.comm_neighbor.append(new_msg.id)
        return
        
    def Gaussian_error(
            self, 
            x0: np.ndarray, 
            Q0: np.ndarray, 
            xi: np.ndarray, 
            Qi: np.ndarray
        ) -> float:
        # calculate 2 Gaussian distribution's divergence
        # x_error = x0 - xi
        # error = np.dot(np.dot(x_error.T, np.linalg.inv(Q0 + Qi)), x_error) + np.log(np.linalg.det(Q0 + Qi))
        # return error[0,0]

        # easy way, just calculate their mean difference
        error = x0.flatten() - xi.flatten()
        return np.sqrt(np.dot(error, error.T)[0,0])
    
    def assign_track(
            self, 
            info: Single_track, 
            track_id: int, 
            local: bool,
            agent_id: int =-1
        ):
        ''' assign one track to another track, 
        INPUTS
        info: Single_track(), is one track 
        track_id: the track_id for the obj's tracking id 
        local: bool value, check if this is obj-sub or sub-sub 
        agent_id: if this is obj-sub, this is sub's agent id
        '''
        # if local:
        index = self.local_track["id"].index(track_id)
        self.local_track["infos"][index].append(info)
        #  self.local_track["incomes"][index] = 1 
        # add this to memory
        self.tracker.track_list[track_id].track_memory(agent_id, info.track_id)
        self.tracker.track_list[track_id].neighbor = []

        # for debug usage
        # if abs(info.x[0] - self.tracker.track_list[track_id].kf.x_k_k[0,0]) > 5:
        #     raise RuntimeError('wrong assignment')

        # else:
        #     self.neighbor_track["infos"][track_id].append(info)
        
        return
    
    def set_index(
            self, 
            a_list: list, 
            element: Any
        ) -> Tuple[int, bool]:
        # pretend the list as a set, return index in a_list given element
        try:
            index = a_list.index(element)
            changed = False
        except:
            a_list.append(element)
            index = len(a_list) - 1
            changed = True
        return index, changed

    def matching_history(self, sub_msg: Info_sense) -> List[int]:
        '''
        ego-other agents info matching
        sub_msg: msg received from another agent
        '''

        X = {"track_id": [], "track": []}
        Y = {"track_id": [], "vertex": [], "index": []}
        
        # put all sub_msg inside Y:
        for j in range(len(sub_msg.tracks)):
            id1 = sub_msg.tracks[j].track_id
            v_y = vertex(id1)
            Y["vertex"].append(v_y)
            Y["index"].append(j)
            Y["track_id"].append(id1)
        
        # preX saves IDs for self.local_track["id"] which is not assigned with historial
        # memeory
        # highest priority: tracking history
        preX = []
        for i in range(len(self.local_track["infos"])):
            m = 0
            isHistory = False
            id0 = self.local_track["id"][i]
            track = self.tracker.track_list[id0]
            for j in range(len(Y["track_id"])):
                j = j - m
                sub_track_id = Y["track_id"][j]
                try:
                    if sub_track_id in track.memory[sub_msg.id]:
                        index_sub_msg = Y["index"][j]
                        self.assign_track(sub_msg.tracks[index_sub_msg], id0, True, agent_id=sub_msg.id)
                        del Y["vertex"][j]
                        del Y["track_id"][j]
                        del Y["index"][j]
                        isHistory = True
                        m += 1
                        break
                except:
                    pass
            if not isHistory:
                preX.append(i)

        # 0.  use threshold to filter out irrelevant tracks
        # for i in range(len(self.local_track["infos"])):
        for i in preX:
            
            id0 = self.local_track["id"][i]
            
            x0 = np.matrix(self.local_track["infos"][i][0].x).reshape(4,1)
            P0 = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)

            for j in Y["index"]:
                id1 = sub_msg.tracks[j].track_id
                x1 = np.matrix(sub_msg.tracks[j].x).reshape(4,1)
                P1 = np.matrix(sub_msg.tracks[j].P).reshape(4,4)
                error = self.Gaussian_error(x0, P0, x1, P1)

                # if self.id == 0:
                #     print(x0.flatten().tolist(), x1.flatten().tolist())
                if error < self.d0:
                    # if self.id == 0:
                    #     print("matched")
                    index_y, changed_y = self.set_index(Y["track_id"], id1)
                    index_x, changed_x = self.set_index(X["track_id"], id0)

                    if changed_x:
                        
                        # TODO: IndexError: list index out of range
                        X["track"].append(self.tracker.track_list[id0])

                    # change the info of neighbor
                    # in neighbor list, should save what remains unchanged, which is
                    # the track id actually, so we should directly save X["index"], Y["index"] as 
                    # track ids
                    X["track"][index_x].neighbor.append([id1, error])
                    try:
                        Y["vertex"][index_y].neighbor.append([id0, error])
                    except IndexError:
                        print(Y, index_y, changed_y, id1, id0)
                        raise RuntimeError("-_-")
    
                    
        ############## change ["index"] and neighbor all below
        # 1.  for all 1-to-1 tracks, just assign them
        # temporal variable to calculate the time of del in list
        m = 0    

        if len(X["track_id"]) > 1 and len(Y["track_id"]) > 1:
            for i in range(len(X["track"])):
               
                i -= m
                track = X["track"][i]
                track_id = X["track_id"][i]
                if len(track.neighbor) == 1:
                    # check if this is the only neighbor for the opposite
                    
                    if track.neighbor[0][0] in Y["track_id"]:
                        index_y = Y["track_id"].index(track.neighbor[0][0])
                        vert = Y["vertex"][index_y]
                        if len(vert.neighbor) == 1:
                            # assign these 2 tracks
                            index_sub_msg = Y["index"][index_y]
                            
                        
                            self.assign_track(sub_msg.tracks[index_sub_msg], track_id, True, agent_id=sub_msg.id)
                            
                            # delete these 2 tracks from set
                            del X["track"][i]
                            del X["track_id"][i]
                            del Y["vertex"][index_y]
                            del Y["track_id"][index_y]
                            del Y["index"][index_y]
                            m += 1
                            continue

        # 2. still left some tracks unassigned, we need to consider it as a minimum 
        # bipartite matching problem, which is Hungarian algorithm, descriped in 
        # http://theory.stanford.edu/~tim/w16/l/l5.pdf
        # The other options are in 
        # https://cyberlab.engr.uconn.edu/wp-content/uploads/sites/2576/2018/09/Lecture_8.pdf

        if len(X["track_id"]) < 1 or len(Y["track_id"]) < 1:
            return Y["index"]
        
        # 3. pick the minimum to assign, then end, no need to go through Hungarian 
        if len(X["track_id"]) < 2:

            error = self.d0
            sub_track_id = -1
            track_id = X["track_id"][0]
            if len(X["track"][0].neighbor) > 0:
                for y in X["track"][0].neighbor:
                    tem_sub_track_id = y[0]
                    if tem_sub_track_id in Y["track_id"]:
                        error_y = y[1]
                        if error_y < error:
                            error = error_y
                            sub_track_id = tem_sub_track_id
                
                # ValueError: 0 is not in list TODO
                if sub_track_id != -1:
                    index_y = Y["track_id"].index(sub_track_id)
                    sub_index = Y["index"][index_y]
                    # assign these two
                    # ValueError: 205 is not in list TODO
                    self.assign_track(sub_msg.tracks[sub_index], track_id, True, agent_id=sub_msg.id)
                    del Y["index"][index_y]
                    return Y["index"]
            else:
                return Y["index"]

        elif len(Y["index"]) < 2:
            # only 1 from y, we need to pick best x
            error = self.d0
            track_id_x = -1
            sub_index = Y["index"][0]
            if len(Y["vertex"][0].neighbor) > 0:
                for x in Y["vertex"][0].neighbor:
                    tem_track_id_x = x[0]
                    if tem_track_id_x in X["track_id"]:
                        error_x = x[1]
                        if error_x < error:
                            error = error_x
                            track_id_x = tem_track_id_x
                
                if track_id_x != -1:
                    # assign these two
                    self.assign_track(sub_msg.tracks[sub_index], track_id_x, True, agent_id=sub_msg.id)

                    del Y["index"][0]
                    return Y["index"]
            else:
                return Y["index"]
        
        else:
            # more than 1 from sub and more than 1 from obj
            # Minimum Bipartite Matching in a connected graph
            m = max(len(Y["track_id"]), len(X["track_id"]))

            # build a m,m fully connected bipartite graph
            edge_matrix = np.full((m, m), 10000)
            # fill the metrics value inside bipartite graph
            for i in range(len(X["track"])):
                track = X["track"][i]
                for y in track.neighbor:
                    subtrack = y[0]
                    try: 
                        index_y = Y["track_id"].index(subtrack)
                        edge_matrix[i, index_y] = y[1]
                    except:
                        pass

            # after preparing the bipartite graph matrix, implement Hungarian alg
            _, col_ind = linear_sum_assignment(edge_matrix)
            # make assignments
            indexes_to_delete = []
            for i in range(len(X["track_id"])):
                index_y = col_ind[i]
                if index_y < len(Y["track_id"]):
                    sub_index = Y["index"][index_y]
                    self.assign_track(sub_msg.tracks[sub_index], X["track_id"][i], True, agent_id=sub_msg.id)
                    indexes_to_delete.append(index_y)
            
            # how to delete a bunch of indexes from a list in a cool way
            # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
            for sub_index in sorted(indexes_to_delete, reverse=True):    
                del Y["index"][sub_index]

        # 4. release all neighbor memory in local tracks
        for i in range(len(X["track_id"])):
            track_id = X["track_id"][i]
            self.tracker.track_list[track_id].neighbor = []
        return Y["index"]

    def matching(self, sub_msg: Info_sense):
        

        # only for host-customer matching

        X = {"track_id": [], "track": []}
        Y = {"track_id": [], "vertex": [], "index": []}

        
        
        # put all sub_msg inside Y:
        for j in range(len(sub_msg.tracks)):
            id1 = sub_msg.tracks[j].track_id
            v_y = vertex(id1)
            Y["vertex"].append(v_y)
            Y["index"].append(j)
            Y["track_id"].append(id1)
        

        # 0.  use threshold to filter out irrelevant tracks
        for i in range(len(self.local_track["infos"])):
            id0 = self.local_track["id"][i]
            x0 = np.matrix(self.local_track["infos"][i][0].x).reshape(4,1)
            P0 = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)

            for j in range(len(sub_msg.tracks)):
                id1 = sub_msg.tracks[j].track_id
                x1 = np.matrix(sub_msg.tracks[j].x).reshape(4,1)
                P1 = np.matrix(sub_msg.tracks[j].P).reshape(4,4)
                error = self.Gaussian_error(x0, P0, x1, P1)

                # if self.id == 0:
                #     print(x0.flatten().tolist(), x1.flatten().tolist())
                if error < self.d0:
                    # if self.id == 0:
                    #     print("matched")
                    index_y, changed_y = self.set_index(Y["track_id"], id1)
                    index_x, changed_x = self.set_index(X["track_id"], id0)

                    if changed_x:
                        
                        # TODO: IndexError: list index out of range
                        X["track"].append(self.tracker.track_list[id0])

                    # change the info of neighbor
                    # in neighbor list, should save what remains unchanged, which is
                    # the track id actually, so we should directly save X["index"], Y["index"] as 
                    # track ids
                    X["track"][index_x].neighbor.append([id1, error])
                    Y["vertex"][index_y].neighbor.append([id0, error])
    
                    
        ############## change ["index"] and neighbor all below
        # 0.5.  for all 1-to-1 tracks, just assign them
        # temporal variable to calculate the time of del in list
        m = 0    

        if len(X["track_id"]) > 1 and len(Y["track_id"]) > 1:
            for i in range(len(X["track"])):
               
                i -= m
                track = X["track"][i]
                track_id = X["track_id"][i]
                if len(track.neighbor) == 1:
                    # check if this is the only neighbor for the opposite
                    
                    if track.neighbor[0][0] in Y["track_id"]:
                        index_y = Y["track_id"].index(track.neighbor[0][0])
                        vert = Y["vertex"][index_y]
                        if len(vert.neighbor) == 1:
                            # assign these 2 tracks
                            index_sub_msg = Y["index"][index_y]
                            
                        
                            self.assign_track(sub_msg.tracks[index_sub_msg], track_id, True, agent_id=sub_msg.id)
                            
                            # delete these 2 tracks from set
                            del X["track"][i]
                            del X["track_id"][i]
                            del Y["vertex"][index_y]
                            del Y["track_id"][index_y]
                            del Y["index"][index_y]
                            m += 1
                            continue

                # 1.  use memory to do assignment, check if there exits a previous assigned
                # track from this subject
                for y in track.neighbor:
                    sub_track_id = y[0]
                    
                    try:
                        # subtrack_id = Y["vertex"][index_y].track_id
                        if sub_track_id in track.memory[sub_msg.id]:
                            index_y = Y["track_id"].index(sub_track_id)
                            sub_index = Y["index"][index_y]
                            # if any track appears in the memory before, assign these two
                            self.assign_track(sub_msg.tracks[sub_index], track_id, True, agent_id=sub_msg.id)
                            
                            # delete these 2 tracks from set
                            del X["track"][i]
                            del X["track_id"][i]
                            del Y["vertex"][index_y]
                            del Y["index"][index_y]
                            del Y["track_id"][index_y]
                            m += 1
                            break
                    except:
                        pass
                    
                    

        # 2. still left some tracks unassigned, we need to consider it as a minimum 
        # bipartite matching problem, which is Hungarian algorithm, descriped in 
        # http://theory.stanford.edu/~tim/w16/l/l5.pdf
        # The other options are in 
        # https://cyberlab.engr.uconn.edu/wp-content/uploads/sites/2576/2018/09/Lecture_8.pdf

        if len(X["track_id"]) < 1 or len(Y["track_id"]) < 1:
            return Y["index"]
        
        # pick the minimum to assign, then end, no need to go through Hungarian 
        if len(X["track_id"]) < 2:

            error = self.d0
            sub_track_id = -1
            track_id = X["track_id"][0]
            if len(X["track"][0].neighbor) > 0:
                for y in X["track"][0].neighbor:
                    tem_sub_track_id = y[0]
                    if tem_sub_track_id in Y["track_id"]:
                        error_y = y[1]
                        if error_y < error:
                            error = error_y
                            sub_track_id = tem_sub_track_id
                
                # ValueError: 0 is not in list TODO
                if sub_track_id != -1:
                    index_y = Y["track_id"].index(sub_track_id)
                    sub_index = Y["index"][index_y]
                    # assign these two
                    # ValueError: 205 is not in list TODO
                    self.assign_track(sub_msg.tracks[sub_index], track_id, True, agent_id=sub_msg.id)
                    del Y["index"][index_y]
                    return Y["index"]
            else:
                return Y["index"]

        elif len(Y["index"]) < 2:
            # only 1 from y, we need to pick best x
            error = self.d0
            track_id_x = -1
            sub_index = Y["index"][0]
            if len(Y["vertex"][0].neighbor) > 0:
                for x in Y["vertex"][0].neighbor:
                    tem_track_id_x = x[0]
                    if tem_track_id_x in X["track_id"]:
                        error_x = x[1]
                        if error_x < error:
                            error = error_x
                            track_id_x = tem_track_id_x
                
                if track_id_x != -1:
                    # assign these two
                    self.assign_track(sub_msg.tracks[sub_index], track_id_x, True, agent_id=sub_msg.id)

                    del Y["index"][0]
                    return Y["index"]
            else:
                return Y["index"]
        
        else:
            # more than 1 from sub and more than 1 from obj
            # Minimum Bipartite Matching in a connected graph
            m = max(len(Y["track_id"]), len(X["track_id"]))

            # build a m,m fully connected bipartite graph
            edge_matrix = np.full((m, m), 10000)
            # fill the metrics value inside bipartite graph
            for i in range(len(X["track"])):
                track = X["track"][i]
                for y in track.neighbor:
                    subtrack = y[0]
                    try: 
                        index_y = Y["track_id"].index(subtrack)
                        edge_matrix[i, index_y] = y[1]
                    except:
                        pass

            # after preparing the bipartite graph matrix, implement Hungarian alg
            _, col_ind = linear_sum_assignment(edge_matrix)
            # make assignments
            indexes_to_delete = []
            for i in range(len(X["track_id"])):
                index_y = col_ind[i]
                if index_y < len(Y["track_id"]):
                    sub_index = Y["index"][index_y]
                    self.assign_track(sub_msg.tracks[sub_index], X["track_id"][i], True, agent_id=sub_msg.id)
                    indexes_to_delete.append(index_y)
            
            # how to delete a bunch of indexes from a list in a cool way
            # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time/41079803
            for sub_index in sorted(indexes_to_delete, reverse=True):    
                del Y["index"][sub_index]

        # 4. release all neighbor memory in local tracks
        for i in range(len(X["track_id"])):
            track_id = X["track_id"][i]
            self.tracker.track_list[track_id].neighbor = []
        return Y["index"]

    def track_to_track(self):
        if self.isObsdyn:
            self.track_to_track_dyn()
        else:
            self.track_to_track_sta()
       
    def track_to_track_sta(self):
        '''
        One to one message matching, obj to sub
        match tracks with dynamic R matrices (sensor model)
        '''
        for neighbor in self.infos_received:
        
            left_tracks_index = self.matching_history(neighbor)
            # add the rest info to local track set in self.tracker, thus it is convinent 
            # for track-to-track assignment of tracks outside FoV
            
            for i in left_tracks_index:
                info = neighbor.tracks[i]
                x0 = np.matrix(info.x).reshape(4,1)
                P0 = np.matrix(info.P).reshape(4,4)
                kf = LinearKF(self.tracker.F, self.tracker.H, x0, P0, self.tracker.Q, self.tracker.R)
                id_ = len(self.tracker.track_list)
                new_track = track(self.t, id_, kf, self.tracker.DeletionThreshold, self.tracker.ConfirmationThreshold, isForeign=True)
                new_track.kf.predict()
                # make this track a confirmed one
                new_track.confirmed = True

                # need to take care of this step, we should not give confident history record
                new_track.history = [1] * self.tracker.DeletionThreshold[1]
                new_track.bb_box_size = self.default_bbsize
                self.tracker.track_list.append(new_track)
                self.tracker.track_list_next_index.append(id_)
                # after adding it to the jpda, then the real id is determined locally, thus
                
                info.track_id = id_
                

                # update to local track
                self.local_track["id"].append(id_)
                self.local_track["infos"].append([info])
        
        return

    def track_to_track_dyn(self):
        '''
        match tracks with dynamic R matrices (sensor model)
        '''
        for neighbor in self.infos_received:
            left_tracks_index = self.matching_history(neighbor)
            # add the rest info to local track set in self.tracker, thus it is convinent 
            # for track-to-track assignment of tracks outside FoV
            
            for i in left_tracks_index:
                info = neighbor.tracks[i]
                x0 = np.matrix(info.x).reshape(4,1)
                P0 = np.matrix(info.P).reshape(4,4)
                kf = dynamic_kf(self.tracker.F, self.tracker.H, x0, P0, self.tracker.Q, self.tracker.R, self.sensor_para["r0"], quality=self.sensor_para["quality"])
                id_ = len(self.tracker.track_list)
                new_track = track(self.t, id_, kf, self.tracker.DeletionThreshold, self.tracker.ConfirmationThreshold, isForeign=True)
                new_track.kf.predict()
                # make this track a confirmed one
                new_track.confirmed = True
                # if self.id==0:
                #     print(x0)
                # need to take care of this step, we should not give confident history record
                new_track.history = [1] * self.tracker.DeletionThreshold[1]
                new_track.bb_box_size = self.default_bbsize
                self.tracker.track_list.append(new_track)
                self.tracker.track_list_next_index.append(id_)
                # after adding it to the jpda, then the real id is determined locally, thus
                
                info.track_id = id_
                # update to local track
                self.local_track["id"].append(id_)
                self.local_track["infos"].append([info])
                
        return

    def consensus(self):
        if self.isVirtual:
            # consensus in horizon-based optimization process, just match the KF
            return self.consensusVirtual()

        if self.NoiseResistant:
            # a logic analysis of the system, 
            # if there are false alarms in detection,
            # we want to avergae all infos, 
            return self.consensusNoiseResistant()
        else:
            # if not, we cant to find the infos with detections to average
            return self.consensusTrackConsistent()

    def consensusNoiseResistant(self):
        self.track_to_track()
        self.info_msg = Info_sense()
        self.info_msg.id = self.id

        # do the consensus, and
        # adapt the local msg based on updated infos 
        for i in range(len(self.local_track["infos"])):

            if len(self.local_track["infos"][i]) == 1:
                newtrack = self.local_track["infos"][i][0]
            
            else:
                # implement the average weight
                denominator = 0     # represent the number of valid tracks
                Sigma_sum = np.array([0.0] * 16)
                q_sum = np.array([0.0] * 4)
                newtrack = Single_track()
                newtrack.track_id = self.local_track["id"][i]  
                isObs = False
                # release neighbor memory in self.tracker
                
                for k in range(len(self.local_track["infos"][i])):
                    Sigma_sum += np.array(self.local_track["infos"][i][k].Sigma) * self.local_track["infos"][i][k].isObs
                    # print(q_sum, self.local_track["infos"][i][k].q)
                    q_sum += np.array(self.local_track["infos"][i][k].q) * self.local_track["infos"][i][k].isObs
                    isObs += bool(self.local_track["infos"][i][k].isObs)
                    denominator += self.local_track["infos"][i][k].isObs
                Sigma_sum /= denominator
                q_sum /= denominator
                Sigma = np.matrix(Sigma_sum).reshape(4,4)
                q = np.matrix(q_sum).reshape(4,1)
                P = np.linalg.inv(Sigma)
                x = np.dot(P, q)
                newtrack.x = x.flatten().tolist()[0]
                newtrack.q = q_sum.tolist()
                newtrack.P = P.flatten().tolist()[0]
                newtrack.Sigma = Sigma_sum.tolist()
                newtrack.isObs = isObs
            
            if newtrack.isObs and self.l <= self.L:
                # only add the msg when there is observation infos
                self.info_msg.tracks.append(newtrack)
            
            # also update local track
            self.local_track["infos"][i] = [newtrack]
        
        self.info_msg.l = self.l

        # finally update the interation and prepare the info for next interation
        self.l += 1
        if self.l > self.L:
            self.l = 0

            self.info_msg.tracks = []
            
            # later update all jpda works
            for i in range(len(self.local_track["id"])):
                id_ = self.local_track["id"][i]
                x_k_k = np.matrix(self.local_track["infos"][i][0].x).reshape(4,1)
                P_k_k = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)
                self.tracker.track_list[id_].kf.x_k_k = x_k_k
                self.tracker.track_list[id_].kf.P_k_k = P_k_k
                self.tracker.track_list[id_].kf.predict()
                if self.local_track["infos"][i][0].isObs:
                    self.tracker.track_list[id_].history[-1] = 1
                else:
                    self.tracker.track_list[id_].history[-1] = 0
                
                if self.tracker.track_list[id_].history[-1] == 0 and self.tracker.track_list[id_].isForeign:
                    # remove it from interested track list
                    self.tracker.track_list[id_].deletion(self.t)
                    self.tracker.track_list_next_index.remove(id_)
                else:
                    self.info_msg.tracks.append(self.local_track["infos"][i][0])
        
           
        # clear the pool for information
        self.comm_neighbor = []
        self.infos_received = []
        
        # the info to show is all info of this agent


        return copy.deepcopy(self.info_msg)
    
    def consensusTrackConsistent(self):        
        

        self.track_to_track()
        self.info_msg = Info_sense()
        self.info_msg.id = self.id
        # do the consensus, and
        # adapt the local msg based on updated infos 
        for i in range(len(self.local_track["infos"])):
            if len(self.local_track["infos"][i]) == 1:
                newtrack = self.local_track["infos"][i][0]
                
            else:
                # implement the average weight
                denominator = 0     # represent the number of valid tracks
                Sigma_sum = np.array([0.0] * 16)
                q_sum = np.array([0.0] * 4)
                newtrack = Single_track()
                
                isObs = False
                # release neighbor memory in self.tracker
                isObsList = []
                traceList = []
                
                for k in range(len(self.local_track["infos"][i])):

                    sigmatemp = np.array(self.local_track["infos"][i][k].Sigma)
                    Sigma_sum += sigmatemp * self.local_track["infos"][i][k].isObs
                    traceList.append(np.trace(np.matrix(sigmatemp).reshape(4,4)))
                    q_sum += np.array(self.local_track["infos"][i][k].q) * self.local_track["infos"][i][k].isObs
                    isObs += bool(self.local_track["infos"][i][k].isObs)
                    isObsList.append(self.local_track["infos"][i][k].isObs)
                    denominator += self.local_track["infos"][i][k].isObs
                
                if not isObs:
                    # find the one with minimal P/maximum Sigma
                    index = np.argmax(traceList)
                    newtrack = self.local_track["infos"][i][index]
                elif sum(isObsList) == 1:
                    # find the only one with observation 
                    # print("it appears, ", isObsList)
                    index = isObsList.index(1)
                    newtrack = self.local_track["infos"][i][index]
                else:
                    # more than one, so just do average 
                    Sigma_sum /= denominator
                    q_sum /= denominator
                    Sigma = np.matrix(Sigma_sum).reshape(4,4)
                    q = np.matrix(q_sum).reshape(4,1)
                    P = np.linalg.inv(Sigma)
                    x = np.dot(P, q)
                    newtrack.x = x.flatten().tolist()[0]
                    newtrack.q = q_sum.tolist()
                    newtrack.P = P.flatten().tolist()[0]
                    newtrack.Sigma = Sigma_sum.tolist()
                
                newtrack.track_id = self.local_track["id"][i]  
                # isObs info out shows if you have local observation
                
                newtrack.isObs = int(isObs)
            
            # add all messages
            self.info_msg.tracks.append(newtrack)
            
            # also update local track
            self.local_track["infos"][i] = [newtrack]
        
        self.info_msg.l = self.l

        # finally update the interation and prepare the info for next interation
        self.l += 1
        if self.l > self.L:
            self.l = 0
            # later update all jpda works
            for i in range(len(self.local_track["id"])):
                id_ = self.local_track["id"][i]
                x_k_k = np.matrix(self.local_track["infos"][i][0].x).reshape(4,1)
                P_k_k = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)
                self.tracker.track_list[id_].kf.x_k_k = x_k_k
                self.tracker.track_list[id_].kf.P_k_k = P_k_k
                self.tracker.track_list[id_].kf.predict()
                if self.local_track["infos"][i][0].isObs:
                    self.tracker.track_list[id_].history[-1] = 1
                else:
                    self.tracker.track_list[id_].history[-1] = 0
                
        
        
        # clear the pool for information
        self.comm_neighbor = []
        self.infos_received = []
        # the info to show is all info of this agent
        return copy.deepcopy(self.info_msg)
   
    def consensusVirtual(self):

        self.info_msg = Info_sense()
        self.info_msg.id = self.id

        # consensus in Rollout, just do KF matching
        for i in range(len(self.local_track["infos"])):
            isObs = self.local_track["infos"][i][0].isObs

            if isObs:
                denominator = 1     # represent the number of valid tracks
                Sigma_sum = np.array(self.local_track["infos"][i][0].Sigma)
                q_sum = np.array(self.local_track["infos"][i][0].q)
                
            else:
                denominator = 0     # represent the number of valid tracks
                Sigma_sum = np.array([0.0] * 16)
                q_sum = np.array([0.0] * 4)

            for k in range(len(self.infos_received)):
                if self.infos_received[k].tracks[i].isObs:
                    # if this info indexed k contains observed info, update it
                    
                    Sigma_sum += np.array(self.infos_received[k].tracks[i].Sigma)
                    q_sum += np.array(self.infos_received[k].tracks[i].q)
                    isObs = True
                    denominator += 1
            if isObs:
                Sigma_sum /= denominator
                q_sum /= denominator
            else:
                Sigma_sum = np.array(self.local_track["infos"][i][0].Sigma)
                q_sum = np.array(self.local_track["infos"][i][0].q)

            Sigma = np.matrix(Sigma_sum).reshape(4,4)
            q = np.matrix(q_sum).reshape(4,1)
            P = np.linalg.inv(Sigma)
            x = np.dot(P, q)
            newtrack = Single_track()
            newtrack.x = x.flatten().tolist()[0]
            newtrack.q = q_sum.tolist()
            newtrack.P = P.flatten().tolist()[0]
            newtrack.Sigma = Sigma_sum.tolist()
            newtrack.isObs = isObs
            newtrack.track_id = i

            self.info_msg.tracks.append(newtrack)
            # also update local track
            self.local_track["infos"][i] = [newtrack]
        
        self.info_msg.l = self.l

        # finally update the interation and prepare the info for next interation
        self.l += 1
        if self.l > self.L:
            self.l = 0

            # later update all KF
            for i in range(len(self.local_track["infos"])):

                x_k_k = np.matrix(self.local_track["infos"][i][0].x).reshape(4,1)
                P_k_k = np.matrix(self.local_track["infos"][i][0].P).reshape(4,4)
                self.tracker[i].x_k_k = x_k_k
                self.tracker[i].P_k_k = P_k_k 
           
        # clear the pool for information
        self.comm_neighbor = []
        self.infos_received = []
        return copy.deepcopy(self.info_msg)
    
    def report_tracks(self) -> Info_sense:
        info_msg_all = Info_sense()
        info_msg_all.id = self.id

        # consensus in Rollout, just do KF matching
        for i in range(len(self.local_track["infos"])):
            
            x_k_k = self.local_track["infos"][i][0].x
            newtrack = Single_track()
            newtrack.x = x_k_k
            

            info_msg_all.tracks.append(newtrack)
            # also update local track
            
        
        return info_msg_all