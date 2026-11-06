#!/bin/bash
#SBATCH --job-name=islands-scale
#SBATCH --account=CHANGEME-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%A_%a.out

# ActFlow vs the best ABE so far (reports/reachability/degenerate_fixes.md) on the disconnected
# islands and on the islands joined by a corridor, as the gap grows, under four reference measures.
#
# 2 families x 5 gaps x 9 arms x 10 seeds = 900 cells, one per array task:
#   actflow                   DPS on GP uncertainty, dps_weight 30 as in abe_corridor
#   abe+mass@{gaussian,student,levy,poisson}       no path cost, positives carry no target mass
#   abe-hard+all@{gaussian,student,levy,poisson}   the best path-cost arm
# Every arm spends 3200 queries (ABE 100 x 32, ActFlow 50 x 64), and none stops at the first hit.
#
# Submit on CSD3 (pre-train the base models once, then the array once they exist):
#   jid=$(sbatch --parsable --array=0 jobs/islands_scale.sh pretrain)
#   sbatch --dependency=afterok:$jid --array=0-899%200 jobs/islands_scale.sh
# Smoke test first, with two cells and 2 rounds:
#   SMOKE=1 sbatch --array=0-1 --time=00:20:00 jobs/islands_scale.sh
# Aggregate once it is done (from the repo root):
#   pixi run python -m adm.evaluation.reachability_sweep gp --dir $OUT --summary_only
#   pixi run python -m adm.evaluation.plot_scale --dir $OUT --out reports/reachability/figures

export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=${REPO:-$HOME/rds/hpc-work/active_diffusion_models}
OUT=${OUT:-$HOME/rds/hpc-work/adm/islands_scale}
NUM_ITERS=100
if [ -n "$SMOKE" ]; then
    OUT=$OUT/smoke
    NUM_ITERS=2
fi

cd "$REPO"

ARGS=(
    gp
    --dir "$OUT"
    --families islands corridor
    --lengths 0.5 1.0 1.5 2.0 3.0
    --arms actflow
           abe+mass@gaussian abe+mass@student abe+mass@levy abe+mass@poisson
           abe-hard+all@gaussian abe-hard+all@student abe-hard+all@levy abe-hard+all@poisson
    --seeds 0 1 2 3 4 5 6 7 8 9
    --corridor_width 0.2
    --num_iters $NUM_ITERS
    --equal_budget
    # ABE, as in degenerate_fixes.md
    --abe_bridge none
    --abe_pool_size 256
    --abe_proposal_size 128
    --abe_query_budget 32
    --abe_distill_size 256
    --abe_trust_samples 32
    # ActFlow, as in abe_corridor
    --samples_per_iter 64
    --dps_weight 30
    --feat_timestep 0.9
    --ft_min_dataset_size 64
    # shared
    --ft_steps 100
    --ft_lr 1e-4
    --ft_batch_size 64
    --num_steps 100
    --gp_kernel rbf
    --gp_lengthscale 0.1
    --final_eval_samples 2048
    --no_visualize
)

if [ "$1" = "pretrain" ]; then
    pixi run python -m adm.evaluation.reachability_sweep "${ARGS[@]}" --pretrain_only
else
    pixi run python -m adm.evaluation.reachability_sweep "${ARGS[@]}" --grid_index $SLURM_ARRAY_TASK_ID
fi
