#!/bin/bash
#$ -cwd
#$ -M caguirr4@jhu.edu
#$ -m ea
#$ -l ram_free=1G,mem_free=1G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q
#$ -V
#$ -N hyper
#$ -t 1-5:1

declare jobs=(
"AGENT"
"PATIENT"
"INSTRUMENT"
"FORCES"
"MANNER"
)



# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu

source ~/.bashrc
conda activate semantics

python hyper_search.py --target_label ${jobs[${SGE_TASK_ID}-1]} --gpu
