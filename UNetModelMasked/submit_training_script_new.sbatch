#!/bin/bash
#SBATCH -J train_delta       # nome del job
#SBATCH -o train_delta/train_delta_%j.out
#SBATCH -e train_delta/train_delta_%j.err
#SBATCH -p teogpu              # partition GPU
#SBATCH -w teogpu02
#SBATCH -N 1                   # 1 nodo
#SBATCH --ntasks=1             # un task principale
#SBATCH --cpus-per-task=16     # CPU per preprocessing / data loading
#SBATCH --gres=gpu:4           # tutte le GPU disponibili su teogpu02
#SBATCH -t 12:00:00            # massimo 12 ore

module load cuda/12.4
source ~/.bashrc

# Attiva l'ambiente virtuale
# Attiva Miniconda dal path corretto
source /farmdisk1/emaragliano/miniconda3/etc/profile.d/conda.sh
conda activate tf

# Allineamento thread CPU a SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Controllo GPU
nvidia-smi

# Lancia lo script Python
python training_script_new.py --param_file params.json