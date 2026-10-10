#!/bin/bash
#SBATCH --job-name="Corridor sweep"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-107

# One (width, length, seed) cell per array task: 6 widths x 6 lengths x 3 seeds = 108 cells.
# Once all tasks are done, aggregate with:
#   pixi run python -m adm.corridor_sweep gp --dir $SCRATCH/adm/corridor/dps_$DPS_WEIGHT --plot_only

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

DPS_WEIGHT=${1:-30}

pixi run python -m adm.corridor_sweep \
    gp \
    --dir $SCRATCH/adm/corridor/dps_$DPS_WEIGHT \
    --grid_index $SLURM_ARRAY_TASK_ID \
    --widths 0.1 0.15 0.2 0.3 0.45 0.6 \
    --lengths 0.5 1.0 1.5 2.0 2.5 3.0 \
    --seeds 0 1 2 \
    --dps_weight $DPS_WEIGHT \
    --gp_kernel rbf \
    --gp_lengthscale 0.1 \
    --feat_timestep 0.9 \
    --num_iters 200 \
    --samples_per_iter 64 \
    --num_steps 50 \
    --ft_steps 200 \
    --ft_min_dataset_size 64 \
    --no_visualize
