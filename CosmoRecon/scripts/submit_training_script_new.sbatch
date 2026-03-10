#!/bin/bash
#SBATCH -J train_delta       # job name
#SBATCH -o train_delta/train_delta_%j.out
#SBATCH -e train_delta/train_delta_%j.err
#SBATCH -p teogpu              # GPU partition
#SBATCH -w teogpu02
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks=1             # one main task
#SBATCH --cpus-per-task=16     # CPUs for preprocessing / data loading
#SBATCH --gres=gpu:4           # all GPUs on teogpu02
#SBATCH -t 12:00:00            # max 12 hours

module load cuda/12.4
source ~/.bashrc

# Activate Conda environment
# Update the path below to match your Miniconda installation
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate tf

# Align CPU threads to SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GPU check
nvidia-smi

# Launch training (unified script)
# Use --loss_type to select: mse, masked_mse, or masked_gradient
python train.py --param_file ../params.json --loss_type mse
