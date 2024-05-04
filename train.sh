#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=train-model
#SBATCH --output=train-model.out

module purge

singularity exec --nv \
            --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:rw \
                        /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif\
                        /bin/bash -c "source /ext3/env.sh; python train_model.py --start 0 --num-epochs 1 --batch-size 128 --save-every 1 --model-name vit_full"
