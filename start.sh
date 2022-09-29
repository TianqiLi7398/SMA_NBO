
python3 main.py\
    --task=run_trials\
    --horizon=5\
    --deci-Schema=sma\
    --case=poission\
    --useSemantic=True\
    --lambda0=5e-3\
    --r=5\
    --traj-type=normal\
    --info-gain=trace_sum\
    --domain=nbo\
    --MCSnum=50\
    --ckf=True\
    --wtp=False\
    --ftol=5e-4\
    --gtol=50\
    --optmethod=pso\
    --opt-step=1