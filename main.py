import os
from utils.simulator import simulator
import time
import json
import utils.analysis
import utils.generatevideo
import utils.mrs_analysis
from utils.notification import secretary

def main(task) -> None:
    sim_num = 1
    MCSnum, horizons = 50, [3, 5]      # 3.0, 5.0, 8.0
    horizon = 3
    deci_Schema = 'decPOMDP'             # 'cen' or 'sma' or 'decPOMDP' or 'pma'
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
    lambda0 = 5e-3
    lambda0_list = [1e-3, 3e-3]     # tree density   [1e-3, 3e-3, 5e-3] 
    r = 5                           # radius of tree
    traj_type = 'normal'            # 'straight' or 'normal' or 'static'
    info_gain = 'trace_sum'         # 'trace_sum', 'info_gain', 'info_acc', 'log_det', 'cov_gain'
    repeat_time = 20                # repeat on same map, will do repeat when >1
    dropout_pattern = None          # 'fail_publisher'
    dropout_prob = 0.2
    dv = 1e0

    if task == 'run':
        if case == 'poisson':
            for horizon in horizons:
                for lambda0 in lambda0_list:
                    run_trials(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
                        sigma, lite, ckf, method, case, wtp, ibr_num, lambda0, r, traj_type, info_gain, repeat_time,
                        dropout_pattern=dropout_pattern, dropout_prob=dropout_prob, dv = dv)
                secretary.send_email(horizon)
        else:
            for horizon in horizons:
                run_trials(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
                        sigma, lite, ckf, method, case, wtp, ibr_num, lambda0, r, traj_type, info_gain, repeat_time,
                        dropout_pattern=dropout_pattern, dropout_prob=dropout_prob, dv = dv)
            secretary.send_email(horizon)
    elif task == 'freq_analysis':
        freq_analysis(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num, traj_type, info_gain, dropout_pattern=dropout_pattern, 
            dropout_prob=dropout_prob, dv = dv)
    elif task == 'time_series_ospa':
        time_series_ospa(repeat_time, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num, traj_type, info_gain, dropout_pattern=dropout_pattern, 
            dropout_prob=dropout_prob, dv = dv)
    elif task == 'ospa_analysis':
        ospa_analysis(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num)
    elif task == 'test_email':
        test_email()
    else:
        print("please indicate the task")
    
    '''
    rename files https://www.networkworld.com/article/3433865/how-to-rename-a-group-of-files-on-linux.html
    '''

def test_email():
    secretary.send_email('text email')

def freq_analysis(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num, traj_type, info_gain, seq=0, dropout_pattern = None, 
            dropout_prob = 0.0, dv = 1e0):
    print("sim number is %s, deci_Schema = %s" % (sim_num, deci_Schema))
    
    horizon_list = [3, 5]
    lambda0_list = [5e-3]
    cen_list = [deci_Schema]
    r = 5
    paralist = []

    for deci_Schema in cen_list:
        for horizon in horizon_list:
            for lambda0 in lambda0_list:
                ele = {"deci_Schema": deci_Schema, "wtp": wtp, "horizon": horizon, 
                    "ftol": ftol, "gtol": gtol, "optmethod": method, "MCSnum": MCSnum, "env": case, 
                    "domain": domain, "lambda0": lambda0, "r": r, "traj_type": traj_type, "info_gain": info_gain,
                    "dropout_pattern": dropout_pattern, "dropout_prob": dropout_prob, "dv": dv}
                paralist.append(ele)
    print(ele)
    utils.mrs_analysis.error_frequency(paralist, 0, sim_num, c=50.0, p=2)

def time_series_ospa(repeated_time, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num, traj_type, info_gain, seq=0, dropout_pattern = None, 
            dropout_prob = 0.0, dv = 1e0):

    """ plot the OSPA mean and error bar over the time with repeated_time for a list of algs"""
    
    
    horizon_list = [3, 5]
    lambda0_list = [1e-3, 3e-3, 5e-3]
    cen_list = ['pma', 'sma', 'decPOMDP']
    # p_list = [0.2, 0.5, 0.8]
    r = 5
    lambda0 = 0.0

    # fix horzion, lambda as the parameter, compare different algirthms
    for horizon in horizon_list:
        for lambda0 in lambda0_list:
            # for dropout_prob in p_list:
            paralist = []
            print(lambda0)
            for deci_Schema in cen_list:
                ele = {"deci_Schema": deci_Schema, "wtp": wtp, "horizon": horizon, 
                    "ftol": ftol, "gtol": gtol, "optmethod": method, "MCSnum": MCSnum, "env": case, 
                    "domain": domain, "lambda0": lambda0, "r": r, "traj_type": traj_type, 
                    "info_gain": info_gain, "dropout_pattern": dropout_pattern, 
                    "dropout_prob": dropout_prob, "dv": dv}
                paralist.append(ele)

            utils.mrs_analysis.time_series_analysis(paralist, 0, repeated_time, c=50.0, p=2)

def ospa_analysis(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
            sigma, lite, ckf, method, case, wtp, ibr_num, seq=0):
    '''plot OSPA over time, check the stability of the alg'''
    
    if domain == 'nbo':
        if deci_Schema == 'cen':
            if wtp:
                date2run = 'cen_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
            else:
                date2run = 'cen_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
        elif deci_Schema == 'dis':
            if wtp:
                date2run = 'dis_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq) + '_wider'
            else:
                date2run = 'dis_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq)
    elif domain == 'MonteCarloRollout':
        if deci_Schema:
            if wtp:
                date2run = 'cen_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
            else:
                date2run = 'cen_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method
        else:
            if wtp:
                date2run = 'dis_wtp_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq) + '_wider'
            else:
                date2run = 'dis_parking_horizon_'+str(horizon)+'_ftol_' + str(ftol) + '_gtol_' + str(gtol) + '_' + method + '_seq_' + str(seq)
        date2run += '_MC_' + str(MCSnum)
    elif domain == 'ibr':
        if wtp:
            date2run = 'parallel_wtp_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
        else:
            date2run = 'parallel_parking_horizon_'+str(horizon) + 'itr_' + str(ibr_num)
    if ckf:
        date2run += '_ckf'
    
    path = os.getcwd()
    # load sensor model
    if case == 'parking':
        filename = os.path.join(path, 'data', 'env', 'parkingSensorPara.json')
    else:
        filename = os.path.join(path, 'data', 'env', 'simplepara.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    dataPath = os.path.join(path, 'data', 'result', domain, data['env'], date2run)

    utils.analysis.sequence_compare_ospa(sim_num, [0], date2run, dataPath, sigma=sigma, gtol=gtol, 
        isCentral=deci_Schema, case = case, domain=domain, c=20)

def run_trials(sim_num, MCSnum, horizon, deci_Schema, domain, ftol, gtol, opt_step, useSemantic,
    sigma, lite, ckf, optmethod, case, wtp, ibr_num, lambda0, r, traj_type, info_gain, repeat_time,
    dropout_pattern = None, dropout_prob = 0.0, dv = 1e0):

    if domain == 'base':
        simulator.base_policy(0, domain, MCSnum, horizon, ftol, gtol, opt_step=opt_step)
        return
    
    if repeat_time > 0:
        i = 0
        for repeated in range(repeat_time):
            start_time = time.time()
            if domain == 'nbo':
                if deci_Schema == 'cen':
                    # simulator.NBO_central(i, date2run, MCSnum, horizon, ftol, gtol, 0)
                    simulator.NBO_central(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = False, 
                            info_gain = info_gain, repeated = repeated)
                    
                    # if case == 'parking':
                    #     simulator.NBO_central(i, horizon, ftol, gtol, wtp, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                    # elif case == 'cheat':
                    #     simulator.deterministic_optimization(i, horizon, ftol, gtol, lite=lite, central_kf=ckf, 
                    #             optmethod=optmethod)
                    # else:
                    #     simulator.NBO_simple_central(i, horizon, ftol, gtol, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                elif deci_Schema == 'sma':
                    
                    simulator.NBO_distr(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                            info_gain = info_gain, repeated = repeated)
                    
                    # elif case == 'poisson':
                    #     simulator.occupancy_nbo_distr(i, horizon, ftol, gtol, wtp, lambda0, r, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                        
                    # else:
                    #     simulator.NBO_simple_distr(i, horizon, ftol, gtol, 0, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                # simulator.deterministic_optimization(i, date2run, MCSnum, horizon, ftol, gtol, opt_step = -1)
                elif deci_Schema == 'pma':
                    if optmethod == 'pso':
                        simulator.PMA_NBO(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                                optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = False, 
                                info_gain = info_gain, repeated = repeated, dropout_pattern=dropout_pattern, 
                                dropout_prob=dropout_prob)
                    elif optmethod == 'discrete':
                        simulator.PMA_NBO_discrete(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                                optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                                info_gain = info_gain, repeated = repeated, dropout_pattern=dropout_pattern, 
                                dropout_prob=dropout_prob, dv = dv)
                elif deci_Schema == 'decPOMDP':
                    simulator.decPOMDP_NBO(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                            info_gain = info_gain, repeated = repeated)
                else:
                    raise RuntimeError('%s_NBO not defined!' % deci_Schema)


            elif domain == 'MonteCarloRollout':
                if deci_Schema == 'cen':
                    simulator.MCRollout_central(i, horizon, ftol, gtol, MCSnum, wtp, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, leader = 0, repeated = repeated)
                elif deci_Schema == 'sma':
                    simulator.MCRollout_distributed(i, horizon, ftol, gtol, MCSnum, wtp, case, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = False, 
                            info_gain = info_gain, repeated = repeated)
                else:
                    raise RuntimeError('%s + MonteCarloRollout not defined!' % deci_Schema)
            elif domain == 'ibr':
                simulator.ibr_parallel(i, horizon, wtp, ibr_num, central_kf=ckf)
            else:
                print("Please indicate the main algorithm, nbo or mcrollout")
            print("Run time for %sth optimization: %s sec"%(i, time.time() - start_time))
    else:
        for i in range(sim_num):
            start_time = time.time()
            if domain == 'nbo':
                if deci_Schema == 'cen':
                    # simulator.NBO_central(i, date2run, MCSnum, horizon, ftol, gtol, 0)
                    simulator.NBO_central(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                            info_gain = info_gain)
                    
                    # if case == 'parking':
                    #     simulator.NBO_central(i, horizon, ftol, gtol, wtp, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                    # elif case == 'cheat':
                    #     simulator.deterministic_optimization(i, horizon, ftol, gtol, lite=lite, central_kf=ckf, 
                    #             optmethod=optmethod)
                    # else:
                    #     simulator.NBO_simple_central(i, horizon, ftol, gtol, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                elif deci_Schema == 'sma':
                    
                    simulator.NBO_distr(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                            info_gain = info_gain)
                    
                    # elif case == 'poisson':
                    #     simulator.occupancy_nbo_distr(i, horizon, ftol, gtol, wtp, lambda0, r, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                        
                    # else:
                    #     simulator.NBO_simple_distr(i, horizon, ftol, gtol, 0, lite=lite, central_kf=ckf, 
                    #         optmethod=optmethod)
                # simulator.deterministic_optimization(i, date2run, MCSnum, horizon, ftol, gtol, opt_step = -1)
                elif deci_Schema == 'pma':
                    simulator.PMA_NBO(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = True, 
                            info_gain = info_gain)
                elif deci_Schema == 'decPOMDP':
                    simulator.decPOMDP_NBO(i, horizon, ftol, gtol, wtp, case, useSemantic, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = False, 
                            info_gain = info_gain)
                else:
                    raise RuntimeError('%s_NBO not defined!' % deci_Schema)


            elif domain == 'MonteCarloRollout':
                if deci_Schema == 'cen':
                    simulator.MCRollout_central(i, horizon, ftol, gtol, MCSnum, wtp, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, leader = 0)
                elif deci_Schema == 'sma':
                    simulator.MCRollout_distributed(i, horizon, ftol, gtol, MCSnum, wtp, case, lite=lite, central_kf=ckf, 
                            optmethod=optmethod, lambda0 = lambda0, r = r, traj_type = traj_type, coverfile = False, 
                            info_gain = info_gain)
                else:
                    raise RuntimeError('%s + MonteCarloRollout not defined!' % deci_Schema)
            elif domain == 'ibr':
                simulator.ibr_parallel(i, horizon, wtp, ibr_num, central_kf=ckf)
            else:
                print("Please indicate the main algorithm, nbo or mcrollout")
            print("Run time for %sth optimization: %s sec"%(i, time.time() - start_time))

           
if __name__ == "__main__":
    # test_email()
    task = 'run'  # 'ospa_analysis', 'run', 'freq_analysis', 'time_series_ospa'
    main(task)