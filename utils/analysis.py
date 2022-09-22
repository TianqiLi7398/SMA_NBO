import numpy as np 
import json
import os
from utils.metrics import ospa
from utils.effects import effects
import matplotlib.pyplot as plt
import copy
from utils.msg import Agent_basic
from matplotlib.patches import Rectangle

def main(date2run, MCSnum, IsBasePolicy, agentid, isbatch=True, opt_step = -1,
        iteration=0, isCentral=False, c=10.0, p=2, isKFOnly=True, horizon = 5, ftol=1e-3, gtol=7, endtime = -1,
        seq = -1, sigma=False, isrevised=False, dataPath=None, case = 'parking'):
    '''
    returns the ospa metrics of a certain target tracking quality
    '''
    path = os.getcwd()
    filename = os.path.join(dataPath, date2run + "_"+str(iteration)+".json")
    
    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    if case == 'parking':
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
    else:
        filename = os.path.join(path, "data", "env", "simpletraj2.json")

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

def specific_scenario(MCSnum, batch_num, isCentral, IsBasePolicy=False, agentid=0, c=20.0):
    
    c= 20.0
    if isCentral:
        date2run = 'central_batch'
    else:
        date2run = 'distr_batch'
    
    dist_scenarios = []
    for i in range(batch_num):
        dist_scenarios.append(main(date2run, MCSnum, IsBasePolicy, 
            agentid, isbatch=True, iteration=i, isCentral=isCentral, c=c))

    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(batch_num):
        plt.plot(time, dist_scenarios[i])
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    filename = os.path.join(path2pic, date2run + '_analysis_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time')
    plt.ylabel('OSPA metric')
    plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    # plt.legend(keylist)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def compare_ospa(MCSnum, batch_num, agentid=0, c=20.0, track_num=4, endtime = -1):

    dist_list = []
    miss_list = []
    track_error = {}
    for i in range(track_num):
        track_error[i] = []

    for isCentral in [True, False]:
        if isCentral:
            date2run = 'central_batch'
            MCSnum = 50
        else:
            date2run = 'distr_batch'
            MCSnum = 50
        
        dist_scenarios = []
        miss_scenarios = []
        track_error_k = {}
        for j in range(track_num):
            track_error_k[j] = []
        # for i in range(batch_num):
        for i in range(50, 100):
        # b = list(range(30)) + list(range(50, 80))
        # for i in b:
            distance, missing_cases, errors_k = main(date2run, MCSnum, IsBasePolicy, 
                agentid, track_num, isbatch=True, iteration=i, isCentral=isCentral, c=c, endtime=endtime)
            dist_scenarios.append(distance)
            miss_scenarios.append(missing_cases)
            for j in range(track_num):
                track_error_k[j].append(errors_k[j])
        dist_list.append(dist_scenarios)
        miss_list.append(miss_scenarios)
        for j in range(track_num):
            track_error[j].append(track_error_k[j])

    mean_list = []
    color_bar = ['r', 'b']
    conf_inter = []
    for dist_ in dist_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(2):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    date2run += "new_"
    filename = os.path.join(path2pic, date2run + '_compare_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time/sec')
    plt.ylabel('OSPA metric')
    # plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["Centralized", "Distributed"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # plot all missing occasions
    mean_list = []
    conf_inter = []
    for dist_ in miss_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(2):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    
    filename = os.path.join(path2pic, date2run + '_compare_%d_trials_missing.png'% batch_num)
    print(filename)
    plt.xlabel('time')
    plt.ylabel('Number of targets missing')
    plt.title('Targets missing of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["Centralized", "Distributed"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for track_id in range(track_num):
        mean_list = []
        conf_inter = []
        trial_losses = []
        for dist_ in track_error[track_id]:
            # mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            dist_ = np.array(dist_)
            mean, std = [], []
            trial_loss = []
            for i in range(dist_.shape[1]):
                raw = dist_[:, i]
                filtered = raw[raw<20]
                if len(filtered) == 0:
                    mean.append(20)
                    std.append(0)
                else:
                    mean.append(np.mean(filtered))
                    std.append(np.std(filtered))
                trial_loss.append(batch_num - len(filtered))
            mean, std = np.array(mean), np.array(std)
            mean_list.append(mean)
            trial_losses.append(trial_loss)
            conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
        
        step_num = len(dist_scenarios[0])

        filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        dt = data["dt"]
        # generate report
        time = np.linspace(dt, step_num * dt, step_num)

        for i in range(2):
            plt.plot(time, mean_list[i], color=color_bar[i])
            plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
            plt.plot(time, trial_losses[i])
        
        
        filename = os.path.join(path2pic, date2run + '_compare_%d_trials_track_%d.png'% (batch_num, track_id))
        print(filename)
        # keylist = range(batch_num)
        plt.xlabel('time')
        plt.ylabel('Error m/s')
        plt.title('Trajectory %d error Monte Carlo for '%track_id+date2run+' of %d trials in agent id= %d, c= %s, p=2' % ( batch_num, agentid, c))
        plt.legend(["Centralized", "cen losses", "Distributed", 'dis losses'])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def ccta_result(MCSnum, batch_num, agentid=0, c=20.0, track_num=4, endtime = -1):

    dist_list = []
    miss_list = []
    track_error = {}
    for i in range(track_num):
        track_error[i] = []

    horizons = [5, 5, 1]
    file_range = list(range(0, 25)) + list(range(50, 75))
    for jj in range(len(horizons)):
        isCentral = [True, False, False][jj]
        horizon = horizons[jj]
        if isCentral:
            date2run = 'central_batch'
            MCSnum = 50
        else:
            date2run = 'distr_batch'
            MCSnum = 50
        
        dist_scenarios = []
        miss_scenarios = []
        track_error_k = {}
        for j in range(track_num):
            track_error_k[j] = []
        # for i in range(batch_num):
        for i in file_range:
        # b = list(range(30)) + list(range(50, 80))
        # for i in b:
            distance, missing_cases, errors_k = main(date2run, MCSnum, IsBasePolicy, 
                agentid, track_num, isbatch=True, iteration=i, isCentral=isCentral, c=c, 
                endtime=endtime, horizon=horizon)
            dist_scenarios.append(distance)
            miss_scenarios.append(missing_cases)
            for j in range(track_num):
                track_error_k[j].append(errors_k[j])
        dist_list.append(dist_scenarios)
        miss_list.append(miss_scenarios)
        for j in range(track_num):
            track_error[j].append(track_error_k[j])

    mean_list = []
    color_bar = ['r', 'b', 'g']
    conf_inter = []
    for dist_ in dist_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(len(horizons)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    date2run += "new_"
    filename = os.path.join(path2pic, 'ccta_paper_compare_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('t')
    plt.ylabel('OSPA metric')
    # plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["All-Agent-at-Once, N = 5", "Agent-by-Agent, N = 5", "Agent-by-Agent, N = 1"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # plot all missing occasions
    mean_list = []
    conf_inter = []
    for dist_ in miss_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(len(horizons)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    
    filename = os.path.join(path2pic, 'ccta_paper_compare_%d_trials_missing.png'% batch_num)
    print(filename)
    plt.xlabel('time')
    plt.ylabel('Number of targets missing')
    plt.title('Targets missing of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["All-Agent-at-Once, N = 5", "Agent-by-Agent, N = 5", "Agent-by-Agent, N = 1"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for track_id in range(track_num):
        mean_list = []
        conf_inter = []
        trial_losses = []
        for dist_ in track_error[track_id]:
            # mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            dist_ = np.array(dist_)
            mean, std = [], []
            trial_loss = []
            for i in range(dist_.shape[1]):
                raw = dist_[:, i]
                filtered = raw[raw<20]
                if len(filtered) == 0:
                    mean.append(20)
                    std.append(0)
                else:
                    mean.append(np.mean(filtered))
                    std.append(np.std(filtered))
                trial_loss.append(batch_num - len(filtered))
            mean, std = np.array(mean), np.array(std)
            mean_list.append(mean)
            trial_losses.append(trial_loss)
            conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
        
        step_num = len(dist_scenarios[0])

        filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        dt = data["dt"]
        # generate report
        time = np.linspace(dt, step_num * dt, step_num)

        for i in range(2):
            plt.plot(time, mean_list[i], color=color_bar[i])
            plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
            plt.plot(time, trial_losses[i])
        
        
        filename = os.path.join(path2pic, 'ccta_paper_compare_%d_trials_track_%d.png'% (batch_num, track_id))
        print(filename)
        # keylist = range(batch_num)
        plt.xlabel('time')
        plt.ylabel('Error m/s')
        plt.title('Trajectory %d error Monte Carlo for '%track_id+date2run+' of %d trials in agent id= %d, c= %s, p=2' % ( batch_num, agentid, c))
        plt.legend(["Centralized", "cen losses", "Distributed", 'dis losses', "Greedy", "gre losses"])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def no_compare_ospa(MCSnum, batch_num, agentid=0, c=20.0, track_num=4, endtime = -1):

    dist_list = []
    miss_list = []
    track_error = {}
    for i in range(track_num):
        track_error[i] = []

    date2run = 'nbo_standard'
    MCSnum = 50
        
    dist_scenarios = []
    miss_scenarios = []
    track_error_k = {}
    for j in range(track_num):
        track_error_k[j] = []
    # for i in range(batch_num):
    for i in range(batch_num, 2 * batch_num):
    # b = list(range(30)) + list(range(50, 80))
    # for i in b:
        distance, missing_cases, errors_k = main(date2run, MCSnum, IsBasePolicy, 
            agentid, isbatch=True, iteration=i, isCentral=isCentral, c=c, endtime=endtime)
        dist_scenarios.append(distance)
        miss_scenarios.append(missing_cases)
        for j in range(track_num):
            track_error_k[j].append(errors_k[j])
    dist_list.append(dist_scenarios)
    miss_list.append(miss_scenarios)
    for j in range(track_num):
        track_error[j].append(track_error_k[j])

    mean_list = []
    color_bar = ['r', 'b']
    conf_inter = []
    for dist_ in dist_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(1):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    date2run += "new_"
    filename = os.path.join(path2pic, date2run + '_compare_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time/sec')
    plt.ylabel('OSPA metric')
    # plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # plot all missing occasions
    mean_list = []
    conf_inter = []
    for dist_ in miss_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(1):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    
    filename = os.path.join(path2pic, date2run + '_%d_trials_missing.png'% batch_num)
    print(filename)
    plt.xlabel('time')
    plt.ylabel('Number of targets missing')
    plt.title('Targets missing of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for track_id in range(track_num):
        mean_list = []
        conf_inter = []
        trial_losses = []
        for dist_ in track_error[track_id]:
            # mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            dist_ = np.array(dist_)
            mean, std = [], []
            trial_loss = []
            for i in range(dist_.shape[1]):
                raw = dist_[:, i]
                filtered = raw[raw<20]
                if len(filtered) == 0:
                    mean.append(20)
                    std.append(0)
                else:
                    mean.append(np.mean(filtered))
                    std.append(np.std(filtered))
                trial_loss.append(batch_num - len(filtered))
            mean, std = np.array(mean), np.array(std)
            mean_list.append(mean)
            trial_losses.append(trial_loss)
            conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
        
        step_num = len(dist_scenarios[0])

        filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        dt = data["dt"]
        # generate report
        time = np.linspace(dt, step_num * dt, step_num)

        for i in range(1):
            plt.plot(time, mean_list[i], color=color_bar[i])
            plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
            plt.plot(time, trial_losses[i])
        
        
        filename = os.path.join(path2pic, date2run + '_%d_trials_track_%d.png'% (batch_num, track_id))
        print(filename)
        # keylist = range(batch_num)
        plt.xlabel('time')
        plt.ylabel('Error m/s')
        plt.title('Trajectory %d error Monte Carlo for '%track_id+date2run+' of %d trials in agent id= %d, c= %s, p=2' % ( batch_num, agentid, c))
        plt.legend(["NBO", "NBO losses"])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def sequence_compare_ospa(batch_num, seq_list, date2run, dataPath=None, agentid=0, 
        MCSnum=50, c=20.0, track_num=4, endtime = -1, horizon = 5, sigma=False, 
        gtol=7, isrevised=False, isCentral=False, IsBasePolicy=False, case='parking',
        domain='nbo'):

    dist_list = []
    miss_list = []
    track_error = {}
    for i in range(track_num):
        track_error[i] = []
    path = os.getcwd()
    if case == 'parking':
        filename = os.path.join(path, "data", 'env', "ParkingTraj2.json")
    else:
        filename = os.path.join(path, "data", "env", "simpletraj2.json")

    with open(filename) as json_file:
        traj = json.load(json_file)
    track_num = len(traj[0])

    for seq in seq_list:
        
        dist_scenarios = []
        miss_scenarios = []
        track_error_k = {}
        for j in range(track_num):
            track_error_k[j] = []
        
        for i in range(batch_num):
        # b = list(range(12)) + list(range(25, 36))
        # for i in b:
            distance, missing_cases, errors_k = main(date2run, MCSnum, IsBasePolicy, 
                agentid, isbatch=True, iteration=i, isCentral=isCentral, c=c, endtime=endtime,
                seq = seq, horizon=horizon, sigma=sigma, gtol=gtol, isrevised=isrevised, dataPath=dataPath,
                case=case)
            dist_scenarios.append(distance)
            miss_scenarios.append(missing_cases)
            for j in range(track_num):
                track_error_k[j].append(errors_k[j])
        dist_list.append(dist_scenarios)
        miss_list.append(miss_scenarios)
        for j in range(track_num):
            track_error[j].append(track_error_k[j])

    mean_list = []
    color_bar = ['r', 'b', 'g', 'yellow']
    conf_inter = []
    for dist_ in dist_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    if case == 'parking':
        filename = os.path.join(os.getcwd(), 'data', 'env', 'parkingSensorPara.json')
    else:
        filename = os.path.join(os.getcwd(), 'data', 'env', 'simplepara2.json')

    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(len(seq_list)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    date2run = domain + '_' + date2run
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    try:
        os.mkdir(path2pic)
    except OSError:
        pass
    
    filename = os.path.join(path2pic, date2run + '_compare_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time/sec')
    plt.ylabel('OSPA metric')
    # plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # plot all missing occasions
    mean_list = []
    conf_inter = []
    for dist_ in miss_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    

    for i in range(len(seq_list)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    
    filename = os.path.join(path2pic, date2run + '_%d_trials_missing.png'% batch_num)
    print(filename)
    plt.xlabel('time')
    plt.ylabel('Number of targets missing')
    plt.title('Targets missing of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for track_id in range(track_num):
        mean_list = []
        conf_inter = []
        trial_losses = []
        for dist_ in track_error[track_id]:
            # mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            dist_ = np.array(dist_)
            mean, std = [], []
            trial_loss = []
            for i in range(dist_.shape[1]):
                raw = dist_[:, i]
                filtered = raw[raw<20]
                if len(filtered) == 0:
                    mean.append(20)
                    std.append(0)
                else:
                    mean.append(np.mean(filtered))
                    std.append(np.std(filtered))
                trial_loss.append(batch_num - len(filtered))
            mean, std = np.array(mean), np.array(std)
            mean_list.append(mean)
            trial_losses.append(trial_loss)
            conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
        

        time = np.linspace(dt, step_num * dt, step_num)

        for i in range(len(seq_list)):
            plt.plot(time, mean_list[i], color=color_bar[i])
            plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
            plt.plot(time, trial_losses[i], color_bar[i])
        
        
        filename = os.path.join(path2pic, date2run + '_%d_trials_track_%d.png'% (batch_num, track_id))
        print(filename)
        # keylist = range(batch_num)
        plt.xlabel('time')
        plt.ylabel('Error m/s')
        plt.title('Trajectory %d error Monte Carlo for '%track_id+date2run+' of %d trials in agent id= %d, c= %s, p=2' % ( batch_num, agentid, c))
        plt.legend(["NBO", "NBO losses"])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def horizon_compare_ospa(MCSnum, batch_num, horizon_list, agentid=0, c=20.0, track_num=4, endtime = -1, seq = 0):

    dist_list = []
    miss_list = []
    track_error = {}
    for i in range(track_num):
        track_error[i] = []

    date2run = 'nbo_standard'
    MCSnum = 50

    for horizon in horizon_list:
        
        dist_scenarios = []
        miss_scenarios = []
        track_error_k = {}
        for j in range(track_num):
            track_error_k[j] = []
        # for i in range(batch_num):
        for i in range(batch_num):
        # b = list(range(30)) + list(range(50, 80))
        # for i in b:
            distance, missing_cases, errors_k = main(date2run, MCSnum, IsBasePolicy, 
                agentid, isbatch=True, iteration=i, isCentral=isCentral, c=c, endtime=endtime,
                seq = seq, horizon=horizon)
            dist_scenarios.append(distance)
            miss_scenarios.append(missing_cases)
            for j in range(track_num):
                track_error_k[j].append(errors_k[j])
        dist_list.append(dist_scenarios)
        miss_list.append(miss_scenarios)
        for j in range(track_num):
            track_error[j].append(track_error_k[j])

    mean_list = []
    color_bar = ['r', 'b', 'g', 'yellow']
    conf_inter = []
    for dist_ in dist_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(len(horizon_list)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    date2run = "horizon_compare_h_" + str(horizon)
    filename = os.path.join(path2pic, date2run + '_compare_%d_trials.png'% batch_num)
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time/sec')
    plt.ylabel('OSPA metric')
    # plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # plot all missing occasions
    mean_list = []
    conf_inter = []
    for dist_ in miss_list:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
    
    step_num = len(dist_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(len(horizon_list)):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    
    filename = os.path.join(path2pic, date2run + '_%d_trials_missing.png'% batch_num)
    print(filename)
    plt.xlabel('time')
    plt.ylabel('Number of targets missing')
    plt.title('Targets missing of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(["NBO"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for track_id in range(track_num):
        mean_list = []
        conf_inter = []
        trial_losses = []
        for dist_ in track_error[track_id]:
            # mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
            dist_ = np.array(dist_)
            mean, std = [], []
            trial_loss = []
            for i in range(dist_.shape[1]):
                raw = dist_[:, i]
                filtered = raw[raw<20]
                if len(filtered) == 0:
                    mean.append(20)
                    std.append(0)
                else:
                    mean.append(np.mean(filtered))
                    std.append(np.std(filtered))
                trial_loss.append(batch_num - len(filtered))
            mean, std = np.array(mean), np.array(std)
            mean_list.append(mean)
            trial_losses.append(trial_loss)
            conf_inter.append(1.96 * std / np.sqrt(batch_num))  # 97.5% percentile point of the standard normal distribution
        
        step_num = len(dist_scenarios[0])

        filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        dt = data["dt"]
        # generate report
        time = np.linspace(dt, step_num * dt, step_num)

        for i in range(len(horizon_list)):
            plt.plot(time, mean_list[i], color=color_bar[i])
            plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                        color=color_bar[i], alpha=.1)
            plt.plot(time, trial_losses[i], color_bar[i])
        
        
        filename = os.path.join(path2pic, date2run + '_%d_trials_track_%d.png'% (batch_num, track_id))
        print(filename)
        # keylist = range(batch_num)
        plt.xlabel('time')
        plt.ylabel('Error m/s')
        plt.title('Trajectory %d error Monte Carlo for '%track_id+date2run+' of %d trials in agent id= %d, c= %s, p=2' % ( batch_num, agentid, c))
        plt.legend(["NBO", "NBO losses"])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main_sep(date2run, MCSnum, IsBasePolicy, agentid, isbatch=True, 
        iteration=0, isCentral=False, c=10.0, p=2, isKFOnly=True, horizon = 5, opt_step=-1, ftol=1e-3, gtol=7):
    path = os.getcwd()
    prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    if isCentral:
        prefix += '_central_'
    if date2run == "nbo_standard":
        prefix = "parkingSIM_nbo_standard_horizon_" + str(int(horizon))+ '_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_optstep_' + str(opt_step) + '_'
    filename = os.path.join(path, 'data', date2run, prefix + "_"+str(iteration)+".json")
    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    filename = os.path.join(path, "data", "ParkingTraj2.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    step_num = len(traj[0][0])
    first, second = [], []
    for i in range(step_num):
        
        traj_k = []
        for ii in range(len(traj[0])):
            traj_k.append([traj[0][ii][i], traj[1][ii][i]])
        # massage the json data into [[x,y]] list format
        est_k = []
        for track in record["agent_est"][agentid][i]:
            est_k.append([track[0], track[1]])
        a, b = ospa.metric_sep(traj_k, est_k, c, p)
        first.append(a)
        second.append(b)
    
    return first, second

def specific_sep(MCSnum, batch_num, isCentral, IsBasePolicy=False, agentid=0, c=20.0):
    
    c= 20.0
    if isCentral:
        date2run = 'central_batch'
    else:
        date2run = 'distr_batch'
    
    error, penalty = [], []
    for i in range(30, 30 + batch_num):
        a, b = main_sep(date2run, MCSnum, IsBasePolicy, 
            agentid, isbatch=True, iteration=i, isCentral=isCentral, c=c, p=1)
        error.append(a)
        penalty.append(b)

    step_num = len(penalty[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(batch_num):
        plt.plot(time, error[i])
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    filename = os.path.join(path2pic, date2run + '_analysis_%d_trials_sep_1.png'% batch_num)
    print(filename)
    keylist = range(batch_num)
    plt.xlabel('time')
    plt.ylabel('OSPA metric')
    plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(keylist)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    for i in range(batch_num):
        plt.plot(time, penalty[i])
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    filename = os.path.join(path2pic, date2run + '_analysis_%d_trials_sep_2.png'% batch_num)
    print(filename)
    keylist = range(batch_num)
    plt.xlabel('time')
    plt.ylabel('OSPA metric')
    plt.title('OSPA metric analysis of Monte Carlo for '+date2run+' of %d trials in agent id= %d, c= %s, p=2' % (batch_num, agentid, c))
    plt.legend(keylist)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def ave_trace(date2run, MCSnum, agentid, isbatch=True, 
        iteration=0, isCentral=False, c=200.0, isKFOnly=True, horizon = 5, trace_num = 4,
        ftol=1e-3, gtol=7):

    path = os.getcwd()
    prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    if isCentral:
        prefix += '_central_'
    
    
        
    filename = os.path.join(path, 'data', date2run, prefix + ('_ftol_' + str(ftol) + '_gtol_' + str(gtol)) + "_"+str(iteration)+".json")
    with open(filename) as json_file:
        record = json.load(json_file)
    # except:
    #     filename = os.path.join(path, 'data', date2run, prefix + "_"+str(iteration)+".json")
    #     with open(filename) as json_file:
    #         record = json.load(json_file)
    step_num = len(record["agent_est"][agentid])
    trace = []
    for i in range(step_num):
        
        # massage the json data into [[x,y]] list format
        value = 0.0
        n = len(record["agent_est"][agentid][i])
        for track in record["agent_est"][agentid][i]:
            # print(np.matrix(track[2]))
            P = np.matrix(track[2]).reshape(4,4)
            value += np.trace(P)
        
        # penality from cardinality
        value += c * abs(trace_num - n)
        value /= max(n, trace_num)
        
        trace.append(min(value, c))
    
    return trace

def trace_plot(MCSnum, agentid, isCentral, batch_num, ftol, gtol, c=200.0):
    
    if isCentral:
        date2run = 'central_batch'
    else:
        date2run = 'distr_batch'
    
    trace_scenarios = []
    for i in range(batch_num):
        trace_scenarios.append(ave_trace(date2run, MCSnum, agentid, isbatch=True, 
        iteration=i, isCentral=isCentral, ftol=ftol, gtol=gtol))

    step_num = len(trace_scenarios[0])

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(batch_num):
        plt.plot(time, trace_scenarios[i])
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    filename = os.path.join(path2pic, date2run + '_Trace_analysis_%d_trials_%s_%s.png'% (batch_num, gtol, ftol))
    print(filename)
    # keylist = range(batch_num)
    plt.ylim([0, 250])
    plt.xlabel('time')
    plt.ylabel('Trace sum')
    plt.title('Trace sum of Monte Carlo for '+date2run+' of %d trials in agent id= %d' % (batch_num, agentid))
    # plt.legend(keylist)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def trace_compare(MCSnum, agentid, batch_num, ftol, gtol):
    trace_stat = []
    for isCentral in [True, False]:
        if isCentral:
            date2run = 'central_batch'
        else:
            date2run = 'distr_batch'
        
        trace_scenarios = []
        for i in range(batch_num):
            trace_scenarios.append(ave_trace(date2run, MCSnum, agentid, isbatch=True, 
            iteration=i, isCentral=isCentral, ftol=ftol, gtol=gtol))
        trace_stat.append(trace_scenarios)
    
    mean_list = []
    color_bar = ['r', 'b']
    conf_inter = []
    for dist_ in trace_stat:
        mean, std = np.mean(np.array(dist_), axis=0), np.std(np.array(dist_), axis=0)
        mean_list.append(mean)
        
        conf_inter.append(1.96 * std / np.sqrt(batch_num))    

    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    # generate report
    step_num = len(mean_list[0])

    time = np.linspace(dt, step_num * dt, step_num)

    for i in range(2):
        plt.plot(time, mean_list[i], color=color_bar[i])
        plt.fill_between(time, mean_list[i] - conf_inter[i], mean_list[i] + conf_inter[i], 
                    color=color_bar[i], alpha=.1)
    
    path2pic = os.path.join(os.getcwd(), 'pics', date2run)
    filename = os.path.join(path2pic, date2run + '_Trace_compare_%d_trials_%s_%s.png'% (batch_num, gtol, ftol))
    print(filename)
    # keylist = range(batch_num)
    plt.xlabel('time')
    plt.ylabel('Trace')
    plt.title('Trace of Monte Carlo for '+date2run+' of %d trials in agent id= %d' % (batch_num, agentid))
    plt.legend(["Centralized", "Distributed"])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajs(date2run, MCSnum, time_length, isbatch=True, seperate = False,
        iteration=0, isCentral=False, isKFOnly=True, horizon = 5, ftol=1e-3, gtol=7, endtime = -1, dt = 25, ddt=25):

    '''
    plots the trajectory of agents in the simulation
    '''
    path = os.getcwd()
    prefix = "parkingSIM_pure_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    if isCentral:
        prefix += '_central_'

    filename = os.path.join(path, 'data', date2run, 
            prefix + '_ftol_' + str(ftol) + '_gtol_' + str(gtol) + "_"+str(iteration)+".json")

    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    filename = os.path.join(path, "data", "ParkingTraj2.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'parking_map.json')
    with open(filename) as json_file:
        SemanticMap = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'parkingSensorPara.json')
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
                        xlim=(-30 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-30 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 12))
    
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
        e.set_alpha(0.2)
        ax.add_artist(e)

    # position of agents
    
    for i in range(len(data["sensors"])):
        rec_cen = record["agent_pos"][0][i]
        
        e = effects.make_rectangle(rec_cen[0], rec_cen[1], rec_cen[2], data["sensors"][i], color=color_list_gradient[i][3])
        ax.add_artist(e)
        
    
    for t in range(int(round(time_length/ddt))+1):
        x, y = [], []
        h = t * ddt / dt
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

    # fig 2, trajectories of uavs

    if seperate:
        h = 3
        num_of_plots = int(round(time_length/ddt))
        num_col = 2
        num_row = num_of_plots/num_col
        
        ratio = 132 / 102.0
        print(ratio)
        fig, axs = plt.subplots(num_row, num_col, figsize=(12, 16), sharex=True, sharey=True)
        
        permutations = []
        for row in range(num_row):
            for col in range(num_col):
                permutations.append((row, col))
        
        for ele in permutations:
            # add obstacles
            for block in SemanticMap["Nodes"]:
                e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                        max(block['feature']['x']) - min(block['feature']['x']), 
                        max(block['feature']['y']) - min(block['feature']['y']),
                        angle = 0,
                        fc ='g',  
                        ec ='g', 
                        lw = 1,
                        linestyle = '-')
                e.set_alpha(0.2)
                axs[ele].add_artist(e)
            t = ele[0] * num_col+ ele[1] + 1

            
            for i in range(len(data["sensors"])):
                rec_cen = record["agent_pos"][t*ddt][i]
                e = effects.make_rectangle(rec_cen[0], rec_cen[1], rec_cen[2], data["sensors"][i], color=color_list_gradient[i][h])
                axs[ele].add_artist(e)
            x, y = [], []
            
            for i in range(len(traj[0])):
                x.append(traj[0][i][t*ddt])
                y.append(traj[1][i][t*ddt])
            axs[ele].scatter(x, y, c=color_list_gradient[-1][h])
            

            # positions of targets in the record
            for i in range(len(traj[0])):
                axs[ele].plot(traj[0][i][0:time_length], traj[1][i][0:time_length], color='grey')
            
            axs[ele].set_xlim([-30 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2])
            axs[ele].set_ylim([-30 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 12])
            axs[ele].set_title("t = %d" %(t * ddt / 5), fontweight='bold')
            
        plt.tight_layout()
        filename = os.path.join(path, 'pics', 'Parking_slowmotion_combined.png')
        plt.savefig(filename, dpi=400)
        plt.close()
        

        return

    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-20 + SemanticMap['canvas'][0][0], SemanticMap['canvas'][0][1] + 2), 
                        ylim=(-20 + SemanticMap['canvas'][1][0], SemanticMap['canvas'][1][1] + 2))
    
    # add obstacles
    for block in SemanticMap["Nodes"]:
        e = Rectangle((min(block['feature']["x"]), min(block['feature']["y"])), 
                max(block['feature']['x']) - min(block['feature']['x']), 
                max(block['feature']['y']) - min(block['feature']['y']),
                angle = 0,
                fc ='None',  
                ec ='g', 
                lw = 1,
                linestyle = '-')
        ax.add_artist(e)

    # position of agents
    for t in range(int(round(time_length/ddt))+1):
        for i in range(len(data["sensors"])):
            rec_cen = record["agent_pos"][t*ddt][i]
            h = t*ddt//dt
            e = effects.make_rectangle(rec_cen[0], rec_cen[1], rec_cen[2], data["sensors"][i], color=color_list_gradient[i][h])
            ax.add_artist(e)
        x, y = [], []
        for i in range(len(traj[0])):
            x.append(traj[0][i][t*ddt])
            y.append(traj[1][i][t*ddt])
        plt.scatter(x, y, c=color_list_gradient[-1][h])

    # Indices to step through colormap
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    cmapx = np.linspace(10.0, 1.0, time_length)
    # c_ = cmapx[::-1]
    c_ = cmapx
    for i in range(len(data["sensors"])):
        # Plot the initial position of FoV
        # rec_cen = record["agent_pos"][0][i]
        # e = effects.make_rectangle(rec_cen[0], rec_cen[1], rec_cen[2], data["sensors"][i], color=color_list[i])
        # ax.add_artist(e)
        # Plot position
        x, y = [], []
        for t in range(time_length):
            x.append(record["agent_pos"][t][i][0])
            y.append(record["agent_pos"][t][i][1])
        # plt.plot(x, y, '--', color = color_list[i])
        plt.scatter(x, y, c=c_, cmap=gradient_list[i], s=3)

    # positions of targets in the record
    for i in range(len(traj[0])):
        plt.plot(traj[0][i][0:time_length], traj[1][i][0:time_length], color='grey')
    
    plt.tight_layout()
    filename = os.path.join(path, 'pics', 'Parking_slowmotion2.png')
    plt.savefig(filename, dpi=400)
    plt.close()

def resim(date2run, MCSnum, IsBasePolicy, isSigma, isbatch=False, iteration=30, isCentral=False, 
        ftol=1e-3, gtol=7, N=5):
    '''
    given policy, replay with variable on consensus step as output
    '''
    isMoving = True
    isObsDyn = True
    isRotate = False
    isFalseAlarm = False
    isKFOnly = True
    horizon = 5
    
    path = os.getcwd()
        
    filename = os.path.join(path, 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    
    filename = os.path.join(path, 'data', 'parking_map.json')
    with open(filename) as json_file:
        SemanticMap = json.load(json_file)
    
    # if isSigma:
    #     prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum) + '_sigma_'
    # else:
    #     prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    # filename = os.path.join(path, 'data', date2run, prefix + '_overall.json')
    # if isbatch:
    prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    if isCentral:
        prefix += '_central_'
        prefix = "parkingSIM_pure_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum) + '_central_'
    filename = os.path.join(path, 'data', date2run, prefix + '_ftol_' + str(ftol) + '_gtol_' + str(gtol) + "_"+str(iteration)+".json")
        
    with open(filename) as json_file:
        record = json.load(json_file)
    record["policy"].pop(0)
    
    sensor_para_list = data["sensors"]
    dt = data["dt"]
    # consensus parameters:
    
    # control parameters:
    cdt = data["control_dt"]

    # load trajectories of targets
    filename = os.path.join(path, "data", "ParkingTraj2.json")
    with open(filename) as json_file:
        traj = json.load(json_file)

    step_num = len(traj[0][0])
    # step_num = 7
    x_samples, y_samples = traj
    time_set = np.linspace(dt, dt * step_num, step_num)

    agent_list = []
    for i in range(len(sensor_para_list)):
        ego = decRollout(MCSnum, horizon, copy.deepcopy(sensor_para_list[i]), i, dt, cdt,
        L0 = N, isObsdyn__ = isObsDyn, isRotate = isRotate, TrackOriented = False, SemanticMap=SemanticMap)
        ego.tracker.DeletionThreshold = [30, 40]
        # ego = agent_simulation.agent(sensor_para_list[i], i, dc_list[i], dt_list[i], L = N)
        agent_list.append(ego)
        
    # broadcast information and recognize neighbors
    pub_basic = []
    for agent in agent_list:
        basic_msg = Agent_basic()
        basic_msg.id = agent.id
        basic_msg.x = agent.sensor_para["position"][0]
        basic_msg.y = agent.sensor_para["position"][1]
        pub_basic.append(basic_msg)
    
    # make each agent realize their neighbors
    for agent in agent_list:
        agent.basic_info(pub_basic)
        print("agent ", agent.id, "\'s neighbors are", agent.neighbor)

    # 2. consensus IF
    true_target_set = []
    
    total_z = []
    agent_est = []
    
    for i in range(len(sensor_para_list)):
        agent_est.append([])
        
    obs_set, bb_set = [], []
    con_agent_pos = []
    
    # last time control happened
    tc = 0.0

    if not isMoving:
        # don't need to go through dynamic function for each agent, just for convinent
        agent_pos_k = []
        for i in range(len(agent_list)):
            agent_pos_k.append(copy.deepcopy(agent.sensor_para["position"]))

    
    for i in range(step_num):
        t = i * dt
        z_k = []
        for ii in range(len(traj[0])):
            z_k.append([traj[0][ii][i], traj[1][ii][i]])

        total_z.append(z_k)

        info_list = []
        obs_k, bb_k = [], []
        
        # 1. feed info to sensors
        for agent in agent_list:
            info_0, z_k_out, size_k_out = agent.sim_detection_callback(copy.deepcopy(z_k), t)
            info_list.append(info_0)
            
            obs_k += z_k_out
            bb_k.append(size_k_out)
            # if agent.id == 2:
        
        obs_set.append(obs_k)
        bb_set.append(bb_k)

        # 2. consensus starts
        for l in range(N):
            # receive all infos
            for agent in agent_list:
                agent.grab_info_list(info_list)
            
            info_list = []
            # then do consensus for each agent, generate new info
            for agent in agent_list:
                info_list.append(agent.consensus())
            # if agent.id == 2:
            #     print(agent.local_track["infos"])

        # 3. after fixed num of average consensus, save result in menory
        for i in range(len(sensor_para_list)):

            agent_i_est_k = []
            
            for track in info_list[i].tracks:
                x = track.x[0]
                y = track.x[1]
                P = track.P
                agent_i_est_k.append([x, y, P, track.track_id])
                
            agent_est[i].append(agent_i_est_k)
            
        # 4. agent movement base policy
        if isMoving:
            agent_pos_k = []
            
            if (np.isclose(t - tc - cdt, 0) or t-tc >= cdt) and not np.isclose(t, time_set[-1]):
                # decision making frequency
                tc = t
                u = []
                
                for x in range(len(agent_list)):
                    agent = agent_list[x]   
                    if IsBasePolicy:                
                        agent.base_policy()
                    else:
                        if isCentral:
                            agent.v = [record["policy"][0][2*x], record["policy"][0][2*x + 1]]
                        else:
                            agent.v = record["policy"][0][x]
                
                record["policy"].pop(0)
            # update position given policy u
            for i in range(len(agent_list)):
             
                agent = agent_list[i]                
                agent.dynamics()
                sensor_para_list[i]["position"] = copy.deepcopy(agent.sensor_para["position"])
                agent_pos_k.append(agent.sensor_para["position"])
        
        con_agent_pos.append(copy.deepcopy(agent_pos_k))
    
    return agent_est

def consensus_analysis(date2run, MCSnum, IsBasePolicy, track_num, isbatch=True, 
        iteration=0, isCentral=False, c=10.0, p=2, isKFOnly=True, horizon = 5, ftol=1e-3, gtol=7, endtime = -1,
        tweak_consensus = True, N=5):
    '''
    returns the mean error among agents
    '''
    path = os.getcwd()
    prefix = "parkingSIM_" + str(isKFOnly) + "_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum)
    if isCentral:
        prefix += '_central_'
        prefix = "parkingSIM_pure_horizon_" + str(int(horizon))+ "_MCN_" + str(MCSnum) + '_central_'
    filename = os.path.join(path, 'data', date2run, 
            prefix + '_ftol_' + str(ftol) + '_gtol_' + str(gtol) + "_"+str(iteration)+".json")
    with open(filename) as json_file:
        record = json.load(json_file)
    
    # load trajectories of targets
    filename = os.path.join(path, "data", "ParkingTraj2.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    step_num = len(traj[0][0])
    if endtime > 0:
        step_num = endtime
    
    filename = os.path.join(os.getcwd(), 'data', 'parkingSensorPara.json')
    with open(filename) as json_file:
        data = json.load(json_file)
    dt = data["dt"]
    time = np.linspace(dt, step_num * dt, step_num)
    distance = []
    missing_cases = []
    error_by_agent = {}
    for agentid in range(len(record["agent_est"])):
        errors = {}
        for i in range(track_num):
            errors[i] = []
        error_by_agent[agentid] = errors

    if tweak_consensus:
        
        agent_est = resim(date2run, MCSnum, IsBasePolicy, False, iteration=iteration, N=N, isCentral=isCentral)
        for i in range(step_num):
            
            traj_k = []
            for ii in range(len(traj[0])):
                traj_k.append([traj[0][ii][i], traj[1][ii][i]])
            # massage the json data into [[x,y]] list format

            for agentid in range(len(agent_est)):
                est_k = []
                for track in agent_est[agentid][i]:
                    est_k.append([track[0], track[1]])
                error_k = ospa.sort_tracks(traj_k, est_k, c)

                for j in range(track_num):
                    error_by_agent[agentid][j].append(error_k[j])

    else:
        for i in range(step_num):
            
            traj_k = []
            for ii in range(len(traj[0])):
                traj_k.append([traj[0][ii][i], traj[1][ii][i]])
            # massage the json data into [[x,y]] list format

            for agentid in range(len(record["agent_est"])):
                est_k = []
                for track in record["agent_est"][agentid][i]:
                    est_k.append([track[0], track[1]])
                error_k = ospa.sort_tracks(traj_k, est_k, c)

                for j in range(track_num):
                    error_by_agent[agentid][j].append(error_k[j])
    
    for j in range(track_num):
        
        for agentid in range(len(record["agent_est"])):
            plt.plot(time, error_by_agent[agentid][j])
        plt.tight_layout()
        plt.legend(["agent 1", "agent 2", "agent 3"])
        filename = os.path.join(path, 'pics', prefix + "_consensus_error_track_"+str(j)+"_N_"+str(N)+".png")
        plt.savefig(filename, dpi=400)
        plt.close()
        print(filename)
    return 
    