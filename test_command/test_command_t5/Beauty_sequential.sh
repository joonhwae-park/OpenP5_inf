#!/bin/bash
#SBATCH --job-name=Beauty_sequential       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=a10g-8-gm192-c192-m768         # partition name
#SBATCH --gpus=2                       # Enter no.of gpus needed
#SBATCH --output=Beauty_sequential.out          # Name of the output file
#SBATCH --error=Beauty_sequential.err           # Name of the error file
#SBATCH --mem=128G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate openp5_t5

python ../../src/src_t5/main.py --datasets Beauty --distributed 1 --gpu 2,3 --tasks sequential,straightforward --item_indexing sequential --epochs 20 --batch_size 64 --master_port 2000 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 --dist_sampler 0 --max_his 20  --sample_num 3,3 --test_prompt seen:0 --lr 1e-3 --train 0 --model_name Beauty_sequential.pt

#python ../../src/src_t5/main.py --datasets Beauty --distributed 1 --gpu 0,1,2,3 --tasks sequential,straightforward --item_indexing sequential --epochs 20 --batch_size 64 --master_port 2000 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 --dist_sampler 0 --max_his 20  --sample_num 3,3 --test_prompt seen:0 --lr 1e-3 --train 0 --model_name Beauty_sequential.pt
