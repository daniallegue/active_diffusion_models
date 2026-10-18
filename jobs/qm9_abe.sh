#!/bin/bash
#SBATCH --job-name="ABE QM9"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --mem-bind=prefer
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest
#SBATCH --gpus=a100:1
#SBATCH --time=120:00:00

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

GP_KERNEL=${1:-linear}
TRUST_RADIUS=${2:-1.0}

# Decide GP kernel arguments
if [[ "$GP_KERNEL" == "linear" ]]; then
    GP_KERNEL_ARG="--gp_kernel linear"
else
    # assume GP_KERNEL is a numeric lengthscale
    GP_KERNEL_ARG="--gp_kernel rbf --gp_lengthscale $GP_KERNEL"
fi

FOLDER="$SCRATCH/adm/qm9_abe/abe_${GP_KERNEL}_${TRUST_RADIUS}"

# Exclusion radius is a Tanimoto distance between Morgan fingerprints, and the trust radius is
# the Girsanov KL summed over all atoms and bonds, so both differ in scale from the toy setups
pixi run python -m adm.abe \
    qm9 \
    gp \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 256 \
    --eval_every 10 \
    $GP_KERNEL_ARG \
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
    --abe_trust_radius $TRUST_RADIUS
