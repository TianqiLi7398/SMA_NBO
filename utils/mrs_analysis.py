import numpy as np 
import json
import os
from utils.metrics import ospa
import matplotlib.pyplot as plt
from scipy import stats
from utils.simulator import simulator
import pandas as pd

def main(date2run, IsBasePolicy, agentid, isbatch=True, opt_step = -1,
        iteration=0, isCentral=False, c=10.0, p=2, isKFOnly=True, horizon = 5, ftol=1e-3, gtol=7, endtime = -1,
        seq = -1, sigma=False, isrevised=False, dataPath=None, case = 'parking', traj_type = 'normal'):
    '''
    returns the ospa metrics of a certain target tracking quality
    '''
    path = os.getcwd()
    filename = os.path.join(dataPath, date2run + "_"+str(iteration)+".json")
    
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
    
    step_num = len(traj[0][0])
    track_num = len(traj[0])
    if endtime > 0:
        step_num = endtime
    distance = []
    missing_cases = []
    errors = []
    for i in range(track_num):
        errors.append([])
    for i in range(step_num):
        
        traj_k = []
        for ii in range(len(traj[0])):
            traj_k.append([traj[0][ii][i], traj[1][ii][i]])
        # massage the json data into [[x,y]] list format
        est_k = []
        
        for track in record["agent_est"][agentid][i]:
            est_k.append([track[0], track[1]])
        error, card, error_k = ospa.metric_counting_missing(traj_k, est_k, c, p)
        # distance.append(ospa.metrics(traj_k, est_k, c, p))
        distance.append(error)
        missing_cases.append(card)
        for j in range(track_num):
            errors[j].append(error_k[j])
    
    return distance, missing_cases, errors

def get_all_error(date2run, agentid, dataPath, iteration, c=10.0, p=2, case='parking', 
        traj_type = 'normal', repeated = -1, start_index = 10):
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
        # print("t = %s" % i)
        ospa_dist, missing, error_k = ospa.metric_counting_missing(traj_k, est_k, c, p)
        
        errors += error_k
        ospas.append(ospa_dist)
        missing_cases.append(missing)
        # if ospa_dist > 15:
        #     print(traj_k, est_k, error_k)
    # print(ospas)
    return ospas, missing_cases, record["time"]

def error_frequency(para_list, agentid, batch_num, c=10.0, p=2):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    data2csv = {}
    mean2csv = {}
    
    for para in para_list:
        iscen = para['deci_Schema']
        date2run, dataPath = filename(para)
        total_error = []
        total_missing = []
        
        print(dataPath)
        times = 0
        for i in range(batch_num):
            error, missing_cases, dt = get_all_error(date2run, agentid, dataPath, i, c=c, p=p, traj_type = para["traj_type"])
            times += dt
            total_error += error
            total_missing += missing_cases
        # print(max(missing_cases))
        # print(max(total_error))
        print("average time = %s" % (times/batch_num))
        
        dx = 0.05
        num = int(c // dx)
        res = stats.relfreq(total_error, numbins=num, defaultreallimits=(0.0, c))
        x = np.linspace(dx, c, num)  
        cdf = np.cumsum(res.frequency)
        # ax.plot(x, cdf, label = 'H = %s, wtp %s, r = %s, lambda = %s, average mssing = %s, %s'%(para["horizon"], para["wtp"], para["r"], para["lambda0"], np.average(total_missing), str(para["deci_Schema"]) + '_'+para["optmethod"]))
        ax.plot(x, cdf, label = 'H = %s, dv = %s'%(para["horizon"], para["dv"]))
        
        item_name = iscen + '_' + para["domain"] + '_' + str(para["lambda0"]) + '_' + str(para["r"]) + '_' + str(para["horizon"]) + '_wtp_' + str(para["wtp"])
        # if iscen: item_name = 'cen_' + item_name
        data2csv[item_name] = cdf.tolist()
        mean2csv[item_name] = [np.mean(total_error), np.std(total_error).tolist()]
    data2csv["x"] = x.tolist()

    csv_freq_name, csv_mean_name, figname = filenames(iscen, para)
    # save data to csv
    df = pd.DataFrame(data2csv)
    df.to_csv(csv_freq_name,sep='\t')
    df_mean = pd.DataFrame(mean2csv)
    df_mean.to_csv(csv_mean_name,sep='\t')


    ax.set_title('Cumulative OSPA frequency, c = %s' % c)   
    ax.set_xticks(np.arange(0.0, c, 1.0))
    ax.set_yticks(np.arange(0.0, 1.2, 0.2))
    
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_xlim([x.min(), x.max()])
    plt.savefig(figname)

def filenames(iscen, para):
    if para["env"] == 'poisson':
        csv_freq_name = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.csv" % \
                (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
        csv_mean_name = os.path.join(os.getcwd(), 'pics', 'analysis', "mean_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.csv" % \
            (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
        figname = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_%s_%s_wtp_%s_r_%s_%s_info_%s.png" % \
            (iscen, para["domain"], para["wtp"], para["r"], para["traj_type"], para["info_gain"]))
    elif para["env"] == 'parksim':
        if para['optmethod'] == 'discrete':
            csv_freq_name = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_parksim_%s_h_%s_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"], para["dv"]))
            csv_mean_name = os.path.join(os.getcwd(), 'pics', 'analysis', "mean_analysis_parksim_%s_h_%s_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"], para["dv"]))            
            figname = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_%s_h_%s_%s_%s_%s.png" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"], para["dv"]))
        elif para['optmethod'] == 'pso':
            csv_freq_name = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_parksim_%s_h_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))
            csv_mean_name = os.path.join(os.getcwd(), 'pics', 'analysis', "mean_analysis_parksim_%s_h_%s_%s_%s.csv" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))            
            figname = os.path.join(os.getcwd(), 'pics', 'analysis', "frequency_analysis_%s_h_%s_%s_%s.png" % \
                (iscen, para['horizon'], para['optmethod'], para["info_gain"]))
    return csv_freq_name, csv_mean_name, figname

def time_series_analysis(para_list, agentid, repeated_time, c=50.0, p=2, start_index = 10):
    fig = plt.figure(figsize=(10, 8))
    from mpl_axes_aligner import align
    # usage https://matplotlib-axes-aligner.readthedocs.io/en/latest/align_usage.html

    start_index = 5
    ax = fig.add_subplot(1, 1, 1)

    ax2 = ax.twinx()

    data2csv = {}
    mean2csv = {}
    color_bar = ['r', 'b', 'g', 'y']
    

    for index, para in enumerate(para_list):
        date2run, dataPath = filename(para)
        dist_list = []
        total_missing = []
        
        for i in range(repeated_time):
            if i in [4, 6, 8, 10, 11, 13, 15]:
                print("continue")
                continue
            error, missing_cases,_ = get_all_error(date2run, agentid, dataPath, 0, 
                    c=c, p=p, traj_type = para["traj_type"], repeated=i, start_index=start_index)
            
            dist_list.append(error)
            # print(missing_cases)
            total_missing.append(missing_cases)
        # print(max(total_error))
        mean_list = []
        conf_inter = []
        miss_list  = []
        
        # for dist_ in dist_list:
        #     mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        #     mean_list.append(mean)
            
        #     conf_inter.append(1.96 * std / np.sqrt(repeated_time))  # 97.5% percentile point of the standard normal distribution
        
        mean_list = np.mean(np.array(dist_list), axis=0)
        std = np.std(np.array(dist_list), axis=0)
        conf_inter = 1.96 * std / np.sqrt(repeated_time)
        miss_list = np.mean(np.array(total_missing), axis=0)

        # print(miss_list)
        time_axis = list(range(start_index, start_index + len(mean_list)))

        # mean_list, conf_inter = np.array(mean_list), np.array(conf_inter)
        
        ax.plot(time_axis, mean_list, color=color_bar[index],  label = '%s' % para["deci_Schema"])
        
        # ax.plot(time_axis, mean_list, color=color_bar[index], label = 'h=%s, drop_out prob = %s' % (str(para["horizon"]), str(para['dropout_prob'])))
        ax2.plot(time_axis, miss_list, color=color_bar[index], linestyle='dotted')
        ax.fill_between(time_axis, mean_list - conf_inter, mean_list + conf_inter, 
                    color=color_bar[index], alpha=.1)

    ax.legend()
    ax.set_title('Time series OSPA over %s repeats, H = %s, p = %s, c = %s, lambda = %s, R = %s' % (repeated_time,para["horizon"], p, c, para["lambda0"], para["r"]))
    # ax.set_xticks(np.arange(0.0, c, 1.0))
    # ax.set_yticks(np.arange(0.0, 1.2, 0.2))   
    ax.grid() 
    ax.set_xlabel('time step')
    ax.set_ylabel('OSPA/m')
    ax2.set_ylabel('Track difference', color='b')
    align.yaxes(ax, 0, ax2, 0, 0.05)
    figname = os.path.join(os.getcwd(), 'pics', 'time_series_analysis', "time_analysis_r=%s_lambda=%s_horizon_%s.png" % \
            (para["r"], para["lambda0"], para["horizon"]))
    plt.savefig(figname)

def filename(para, seq=0, ckf=True):
    # deci_Schema, wtp, horizon, ftol, gtol, optmethod, MCSnum, env, domain, lambda0, r, traj_type = para

    data2save = simulator.filename_generator(para["horizon"], para["ftol"], para["gtol"], para["wtp"], para["env"], 
            seq, ckf, para["optmethod"], para["deci_Schema"], para["domain"], lambda0 = para["lambda0"], r= para["r"], 
            MCSnum = para["MCSnum"], traj_type = para["traj_type"], info_gain=para["info_gain"], 
            dropout_pattern=para["dropout_pattern"], dropout_prob=para["dropout_prob"], dv = para['dv'])
    
    dataPath = os.path.join(os.getcwd(), 'data', 'result', para["domain"], para["env"], data2save)
    return data2save, dataPath