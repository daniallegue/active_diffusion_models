#!/bin/bash
#SBATCH --job-name="ABE QM9 CTMC"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --mem-bind=prefer
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest
#SBATCH --gpus=a100:1
#SBATCH --time=120:00:00

# One arm of the CTMC reference ablation on QM9, the categorical analogue of the --abe_reference
# sweep: the clock sets the tails, the twist is a learned proposal whose correction keeps the target
# exact. "regenerate" is the FlowMol-regeneration dispersal, reweighted by -int V only, as baseline.
#
#   sbatch jobs/qm9_ctmc_ablation.sh <clock> <twist> [param] [heat_time]
#
#   clock:      gaussian | levy | student | poisson | regenerate
#   twist:      0 | 1 (ignored for regenerate)
#   param:      alpha for levy (default 1.8), df for student (default 5), jump_rate for poisson
#               (default 0.2); "-" for the default
#   heat_time:  s of the heat clock over the molecular graph; unset gives one clock per molecule
#
# Grid:
#   for c in gaussian levy student poisson; do for t in 0 1; do sbatch jobs/qm9_ctmc_ablation.sh $c $t; done; done
#   sbatch jobs/qm9_ctmc_ablation.sh levy 1 1.5
#   for s in 0 0.5 1 2; do sbatch jobs/qm9_ctmc_ablation.sh levy 1 - $s; done   # localised vs scattered tails
#   sbatch jobs/qm9_ctmc_ablation.sh regenerate 0

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

CLOCK=${1:-gaussian}
TWIST=${2:-0}
PARAM=${3:-}
HEAT=${4:-}
if [[ "$PARAM" == "-" ]]; then
    PARAM=""
fi

# Run the codec check once per job: it is cheap and catches a bond written on one direction only
pixi run python tests/test_ctmc_molecules.py || exit 1

case "$CLOCK" in
    regenerate)
        REF_ARGS="--abe_reference gaussian"
        TWIST=0
        ;;
    levy)
        REF_ARGS="--abe_reference ctmc --abe_ctmc_clock levy --abe_reference_alpha ${PARAM:-1.8}"
        ;;
    student)
        REF_ARGS="--abe_reference ctmc --abe_ctmc_clock student --abe_reference_df ${PARAM:-5}"
        ;;
    poisson)
        REF_ARGS="--abe_reference ctmc --abe_ctmc_clock poisson --abe_jump_rate ${PARAM:-0.2}"
        ;;
    gaussian)
        REF_ARGS="--abe_reference ctmc --abe_ctmc_clock gaussian"
        ;;
    *)
        echo "unknown clock $CLOCK"; exit 1
        ;;
esac

TOPO_ARGS=""
if [[ -n "$HEAT" && "$CLOCK" != "regenerate" ]]; then
    TOPO_ARGS="--abe_ctmc_topology heat --abe_ctmc_heat_time $HEAT"
fi

TWIST_ARGS=""
if [[ "$TWIST" == "1" ]]; then
    TWIST_ARGS="--abe_twist"
fi

FOLDER="$SCRATCH/adm/qm9_ctmc/${CLOCK}${PARAM:+_$PARAM}${HEAT:+_heat$HEAT}_twist${TWIST}"

# Same setup as jobs/qm9_abe.sh, with the dispersal as a costed path of 8 steps
pixi run python -m adm.abe \
    qm9 \
    gp \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 256 \
    --eval_every 10 \
    --gp_kernel linear \
    --num_iters 1000 \
    --sample_batch_size 128 \
    --ft_lr 1e-4 \
    --ft_batch_size 64 \
    --ft_steps 500 \
    --abe_bridge none \
    --abe_pool_size 512 \
    --abe_proposal_size 128 \
    --abe_query_budget 16 \
    --abe_exclusion_radius 0.2 \
    --abe_distill_size 512 \
    --abe_trust_samples 64 \
    --abe_trust_radius 1.0 \
    --abe_disperse_steps 8 \
    --abe_state_cost surrogate \
    --abe_state_cost_form excess \
    $REF_ARGS \
    $TOPO_ARGS \
    $TWIST_ARGS
