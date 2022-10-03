import numpy as np 
import json
import os
from utils.metrics import ospa
import matplotlib.pyplot as plt
from scipy import stats
from utils.simulator import simulator
import pandas as pd
from typing import Tuple, Any, List


def get_all_error(
        date2run: str, 
        agentid: int, 
        dataPath: str, 
        iteration: int, 
        c: float=10.0, 
        p: float=2, 
        case: str='parking', 
        traj_type: str = 'normal', 
        repeated: int = -1, 
        start_index: int = 10
    ) -> Tuple[List[Any], List[Any], List[Any]]:
    '''
    returns the frequence of error over time of a certain target tracking quality
    '''
    path = os.getcwd()
    if repeated > 0:
        filename = os.path.join(dataPath, date2run + "_" + str(iteration) + "_" + str(repeated) + ".json")
    else:
        filename = os.path.join(dataPath, date2run + "_" + str(iteration) + ".json")
    
    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    if case == 'simple1':
        filename = os.path.join(path, "data", 'env', "simpletraj.json")
    elif case == 'simple2':
        filename = os.path.join(path, "data", 'env', "simpletraj2.json")
    elif traj_type == 'straight':
        filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
    else:
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
    
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    # gather all error of all files together
    errors = []
    ospas = []
    missing_cases = []
    step_num = len(record["agent_est"][agentid])
    for i in range(start_index, step_num):
        
        traj_k = []
        for ii in range(len(traj[0])):
            traj_k.append([traj[0][ii][i], traj[1][ii][i]])
        # massage the json data into [[x,y]] list format
        est_k = []
        
        for track in record["agent_est"][agentid][i]:
            est_k.append([track[0], track[1]])
        
        ospa_dist, missing, error_k = ospa.metric_counting_missing(traj_k, est_k, c, p)
        
        errors += error_k
        ospas.append(ospa_dist)
        missing_cases.append(missing)
        # if ospa_dist > 15:
        #     print(traj_k, est_k, error_k)
    
    return ospas, missing_cases, record["time"]

def error_frequency(
        para_list:dict, 
        agentid: int, 
        batch_num: int, 
        c: float =10.0, 
        p: float=2,
        start_index: int=10
    ):
    '''
    ospa value of different maps without repeatition
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    data2csv = {}
    mean2csv = {}
    
    for para in para_list:
        iscen = para['deci_Schema']
        date2run, dataPath = filename(para)
        total_error = []
        total_missing = []
        
        times = 0
        for i in range(batch_num):
            error, missing_cases, dt = get_all_error(date2run, agentid, dataPath, \
                                        i, c=c, p=p, traj_type = para["traj_type"], \
                                        start_index=start_index)
            times += dt
            total_error += error
            total_missing += missing_cases
        print("average time = %s" % (times/batch_num))
        
        dx = 0.05
        num = int(c // dx)
        res = stats.relfreq(total_error, numbins=num, defaultreallimits=(0.0, c))
        x = np.linspace(dx, c, num)  
        cdf = np.cumsum(res.frequency)
        
        ax.plot(x, cdf, label = 'H = %s'%(para["horizon"]))
        
        item_name = iscen + '_' + para["domain"] + '_' + str(para["lambda0"]) + '_'\
             + str(para["r"]) + '_' + str(para["horizon"]) + '_wtp_' + str(para["wtp"])
        
        data2csv[item_name] = cdf.tolist()
        mean2csv[item_name] = [np.mean(total_error), np.std(total_error).tolist()]
    data2csv["x"] = x.tolist()

    csv_freq_name, csv_mean_name, figname = filenames(iscen, para)
    # save data to csv
    df = pd.DataFrame(data2csv)
    df.to_csv(csv_freq_name, sep='\t')
    df_mean = pd.DataFrame(mean2csv)
    df_mean.to_csv(csv_mean_name,sep='\t')


    ax.set_title('Cumulative OSPA frequency, c = %s' % c)   
    ax.set_xticks(np.arange(0.0, c, 1.0))
    ax.set_yticks(np.arange(0.0, 1.2, 0.2))
    
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_xlim([x.min(), x.max()])
    plt.savefig(figname)

def filenames(
        iscen: bool, 
        para: dict
    ) -> Tuple[str, str, str]:

    '''generate filenames for statistical data'''
    path = os.path.join(path)
    if not os.path.exists(path):
        os.makedirs(path)

    if para["env"] == 'poisson':
        csv_freq_name = os.path.join(path, \
            "frequency_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.csv" % \
            (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
        csv_mean_name = os.path.join(path, \
            "mean_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.csv" % \
            (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
        figname = os.path.join(path,\
            "frequency_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.png" % \
            (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
    elif para["env"] == 'parksim':
        if para['optmethod'] == 'pso':
            csv_freq_name = os.path.join(path, \
                "frequency_analysis_parksim_%s_h_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))
            csv_mean_name = os.path.join(path, \
                "mean_analysis_parksim_%s_h_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))            
            figname = os.path.join(path, \
                "frequency_analysis_%s_h_%s_%s_%s.png" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))
    return csv_freq_name, csv_mean_name, figname

def time_series_analysis(
        para_list: dict, 
        agentid: int, 
        repeated_time: int, 
        c: float=50.0, 
        p: float=2, 
        start_index: int = 10
    ):
    '''
    OSPA value of same map but with more than 1 repeatition
    '''
    fig = plt.figure(figsize=(10, 8))
    from mpl_axes_aligner import align
    # usage https://matplotlib-axes-aligner.readthedocs.io/en/latest/align_usage.html

    ax = fig.add_subplot(1, 1, 1)

    ax2 = ax.twinx()

    color_bar = ['r', 'b', 'g', 'y']
    

    for index, para in enumerate(para_list):
        date2run, dataPath = filename(para)
        dist_list = []
        total_missing = []
        
        for i in range(repeated_time):
            
            error, missing_cases,_ = get_all_error(date2run, agentid, dataPath, 0, \
                            c=c, p=p, traj_type = para["traj_type"], repeated=i, \
                            start_index=start_index)
            
            dist_list.append(error)
            
            total_missing.append(missing_cases)

        mean_list = []
        conf_inter = []
        miss_list  = []
        
        mean_list = np.mean(np.array(dist_list), axis=0)
        std = np.std(np.array(dist_list), axis=0)
        conf_inter = 1.96 * std / np.sqrt(repeated_time)
        miss_list = np.mean(np.array(total_missing), axis=0)

        time_axis = list(range(start_index, start_index + len(mean_list)))
        
        ax.plot(time_axis, mean_list, color=color_bar[index],  label = '%s' % para["deci_Schema"])
        
        
        ax2.plot(time_axis, miss_list, color=color_bar[index], linestyle='dotted')
        ax.fill_between(time_axis, mean_list - conf_inter, mean_list + conf_inter, 
                    color=color_bar[index], alpha=.1)

    ax.legend()
    ax.set_title('Time series OSPA over %s repeats, H = %s, p = %s, c = %s, lambda = %s, R = %s'\
         % (repeated_time,para["horizon"], p, c, para["lambda0"], para["r"]))
 
    ax.grid() 
    ax.set_xlabel('time step')
    ax.set_ylabel('OSPA/m')
    ax2.set_ylabel('Track difference', color='b')
    align.yaxes(ax, 0, ax2, 0, 0.05)
    path = os.path.join(os.getcwd(), 'pics', 'time_series_analysis')
    if not os.path.exists(path):
        os.makedirs(path)
    figname = os.path.join(path, \
            "time_analysis_r=%s_lambda=%s_horizon_%s.png" % \
            (para["r"], para["lambda0"], para["horizon"]))
    plt.savefig(figname)

def filename(para: dict, seq: int =0, ckf: bool =True) -> Tuple[str, str]:
    '''get the saved records' file name'''

    data2save = simulator.filename_generator(para["horizon"], para["ftol"], para["gtol"], \
            para["wtp"], para["env"], seq, ckf, para["optmethod"], para["deci_Schema"], \
            para["domain"], lambda0 = para["lambda0"], r= para["r"], \
            MCSnum = para["MCSnum"], traj_type = para["traj_type"], info_gain=para["info_gain"], \
            )
    
    dataPath = os.path.join(os.getcwd(), 'data', 'result', para["domain"], para["env"], data2save)
    return data2save, dataPath