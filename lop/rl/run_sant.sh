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

module load mesa/22.1.6-yi2tztm
module load patchelf/0.17.2-aqmx4qb
source ../../.venv/bin/activate
###### algo = {nystrom, nested, cg, ppo}
###### task = {cartpole, acrobot}, {walker2d, humanoid, humanoidstandup, inverted_pendulum, inverted_double_pendulum, pusher, hopper, reacher}

# run one seed
python run_ppo.py -c cfg/sant/er.yml -s 0