
python3 main.py\
    --task=visualize\
    --horizon=5\
    --deci-Schema=test\
    --case=poisson\
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
    --opt-step=1\
    --horizon-list=5\
    --deci-Schema-list='test'\
    --lambda0-list=0.005\
    --r-list=5\
    --iteration=$1\
    --repeated-times=1