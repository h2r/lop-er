#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load Python module

export PYTHONPATH=$(pwd):$PYTHONPATH
source .venv/bin/activate
###### algo = {nystrom, nested, cg, ppo}
###### task = {cartpole, acrobot}, {walker2d, humanoid, humanoidstandup, inverted_pendulum, inverted_double_pendulum, pusher, hopper, reacher}

# python lop/permuted_mnist/online_expr.py -c temp_cfg/0.json
python lop/rl/run_ppo.py -c cfg/ant/std.json -s 0