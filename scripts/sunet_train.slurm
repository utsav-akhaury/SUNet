#!/bin/bash
#SBATCH --job-name=sunet_test           # nom du job
#SBATCH --gres=gpu:1                    # nombre de GPU à réserver par nœud    #SBATCH --exclude=gpu[001-006,008]  # 003,007 
#SBATCH --partition shared-gpu
#SBATCH --mem 15000
#SBATCH --time=0-04:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=sunet_test_%j.out     # nom du fichier de sortie
#SBATCH --error=sunet_test_%j.out      # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/SUNet

module load GCC/8.3.0 CUDA/10.1.243 cuDNN/7.6.4.38 NCCL/2.7.3
conda activate sunet
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

echo STARTING AT `date`
CUDA_LAUNCH_BLOCKING=1 srun python train.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
