#!/bin/bash
#SBATCH --job-name="ABE vs ActFlow, reachability"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-29

# ActFlow vs ABE vs ABE with a path cost, on the corridor (detour ratio 1, the control) and the
# hook (detour ratio 3.9, where the route and the shortcut disagree).
#
# 2 families x 5 arms x 3 seeds = 30 cells, one per array task. Aggregate afterwards with:
#   pixi run python -m adm.evaluation.reachability_sweep gp --dir $SCRATCH/adm/reach --summary_only
#
# S_0 is the same in both families by construction, so the base model is pre-trained once per seed
# and shared across every cell of that seed.

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

FOLDER=${1:-$SCRATCH/adm/reach}

pixi run python -m adm.evaluation.reachability_sweep \
    gp \
    --dir $FOLDER \
    --grid_index $SLURM_ARRAY_TASK_ID \
    --families corridor hook \
    --seeds 0 1 2 \
    --corridor_width 0.2 \
    --corridor_length 1.5 \
    --corridor_gap 0.3 \
    --gp_kernel rbf \
    --gp_lengthscale 0.1 \
    --num_iters 100 \
    --abe_query_budget 32 \
    --abe_bridge cfm \
    --abe_bridge_steps 200 \
    --no_visualize
