#!/bin/bash
#SBATCH --job-name=Movietweetings_lab_sequential       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=l4-4-gm96-c48-m192         # partition name
#SBATCH --gpus=2                       # Enter no.of gpus needed
#SBATCH --output=Movietweetings_lab_sequential.out          # Name of the output file
#SBATCH --error=Movietweetings_lab_sequential.err           # Name of the error file
#SBATCH --mem=128G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate openp5_t5_ML1M
python ../../src/src_t5/main.py --datasets Movietweetings_lab --distributed 1 --gpu 2,3 --tasks sequential,straightforward --item_indexing sequential --epochs 5 --batch_size 128 --master_port 1984 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 --dist_sampler 0 --max_his 20  --sample_num 3,3 --train 1 --test_prompt unseen:0 --lr 1e-3 --test_before_train 1 --test_epoch 0 --test_filtered 1 --test_filtered_batch 0 --model_name Movietweetings_sequential.pt