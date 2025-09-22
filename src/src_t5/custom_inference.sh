#!/bin/bash
#SBATCH --job-name=P5_inference       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=l4-4-gm96-c48-m192         # partition name
#SBATCH --gpus=2                       # Enter no.of gpus needed
#SBATCH --output=custom_inference.out          # Name of the output file
#SBATCH --error=custom_inference.err           # Name of the error file
#SBATCH --mem=64G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate openp5_t5

python custom_inference.py --model_path ../../test_command/checkpoint/Movietweetings_lab_sequential.pt --user_history "65 1001 1006 1008 1009 1011 1013 1026 1028 1033 1034 1035 1039 1043 1045 1052 1054 1057 1060 1075 1077 1070" --backbone t5-small --dataset Movietweetings --prompt_file /scratch/jpa2742/OpenP5_inf/prompt.txt --task sequential
