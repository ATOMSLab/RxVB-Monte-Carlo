#!/bin/bash
#SBATCH --job-name=MFI-50w
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=16000
#SBATCH --time=14-00:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# One array task per chemical potential; each task runs 32 replicas on one node.
set -euo pipefail
source /path/to/feasst-env/bin/activate

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TEMP=373
NW=50
mu_list=(-550 -585)
MU="${mu_list[${SLURM_ARRAY_TASK_ID:-0}]}"
LOGNAME="${TEMP}-${NW}w-mu${MU#-}"

echo "[$(date)] temp=$TEMP num_particles=$NW mu=$MU" > "logs/${LOGNAME}.out"

python launch_2_acidic_zeolite.py \
    --temperature $TEMP \
    --num_particles $NW \
    --mu "$MU" \
    --procs_per_node 32 \
    2>&1 | tee -a "logs/${LOGNAME}.out"
