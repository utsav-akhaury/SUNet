#!/bin/bash
#SBATCH --job-name=sunet     # nom du job
#SBATCH --gpus 1
#SBATCH --partition shared-gpu
#SBATCH --mem 20000          #20000
#SBATCH --time=0-12:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=sunet_%j.out # nom du fichier de sortie
#SBATCH --error=sunet_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/SUNet

module load GCC/8.3.0 CUDA/10.1.243 cuDNN/7.6.4.38 NCCL/2.7.3
conda activate sunet

echo STARTING AT `date`
srun python train.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
