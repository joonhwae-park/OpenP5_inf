#!/bin/bash
#SBATCH --job-name=generate_dataset       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=a10g-8-gm192-c192-m768        # partition name
#SBATCH --gpus=2                       # Enter no.of gpus needed
#SBATCH --output=generate_dataset.out          # Name of the output file
#SBATCH --error=generate_dataset.err           # Name of the error file
#SBATCH --mem=256G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.edu   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate openp5_t5


ts=100
cluster=10
for dataset in Beauty ML100K ML1M Yelp Electronics Movies CDs Clothing Taobao LastFM
do
    for indexing in random sequential collaborative
    do
        python ./src/src_llama/generate_dataset.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode validation --prompt seen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt seen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}

        python ./src/src_llama/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --prompt unseen:0 --collaborative_token_size ${ts} --collaborative_cluster ${cluster}
    done
done
