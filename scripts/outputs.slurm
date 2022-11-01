#!/bin/bash
#SBATCH --job-name=output     # nom du job
#SBATCH --gpus 1
#SBATCH --partition debug-gpu
#SBATCH --mem 0          #20000
#SBATCH --time=0-00:15:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_%j.out # nom du fichier de sortie
#SBATCH --error=output_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/SUNet

module load GCC/8.3.0 CUDA/10.1.243 cuDNN/7.6.4.38 NCCL/2.7.3
conda activate sunet

echo STARTING AT `date`
srun python outputs.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
