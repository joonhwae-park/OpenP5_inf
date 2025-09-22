import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/jpa2742/OpenP5_inf/src/src_t5/hf_cache"
os.environ["HF_HOME"] = "/scratch/jpa2742/OpenP5_inf/src/src_t5/hf_cache"

import argparse
import torch

from transformers import AutoTokenizer, T5Config

from model.P5_T5 import P5_T5
from data.TestDataset import TestDataset
from utils.generation_trie import Trie, prefix_allowed_tokens_fn
from utils import utils
from data.MultiTaskDataset import MultiTaskDataset


#def recommend_items(
#    args,
#    model: torch.nn.Module,
#    tokenizer,
#    dataset_name: str,
#    task_name: str,
#    user_history: str,
#) -> list[int]:

def recommend_items(
    args,
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str,
    task_name: str,
    user_history: str,
) -> list[tuple[int, float]]:

    ds = TestDataset(args, dataset_name, task_name)  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

    candidates = set(ds.all_items) 


    bos_id = model.config.decoder_start_token_id 
    eos_id = tokenizer.eos_token_id 
    trie = Trie(
        [
            [bos_id]
            + tokenizer.encode(f"{dataset_name} item_{cid}", add_special_tokens=False)
            + [eos_id]
            for cid in candidates
        ]
    )
    prefix_fn = prefix_allowed_tokens_fn(trie)


    inputs = tokenizer(user_history, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)

    # generate (50, beam=5, return=5)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=args.max_length,
        num_beams=args.generate_num,
        num_return_sequences=args.generate_num,
        prefix_allowed_tokens_fn=prefix_fn,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    sequences      = outputs.sequences                   # Tensor[num_return_sequences, seq_len]
    sequence_scores = outputs.sequences_scores           # Tensor[num_return_sequences]
    
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    recs_ids  = []
    for txt, score in zip(texts, sequence_scores.tolist()):
        if "item_" in txt:
            item_id = int(txt.split("item_")[1])
            recs_ids.append((item_id, score))  # score is the beam log-prob

    return recs_ids

    # Parse ID after decoding
    #seqs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    #rec_ids = []
    #for s in seqs:
    #    if "item_" in s:
    #        rec_ids.append(int(s.split("item_")[1]))
    #return rec_ids


def main():
    parser = argparse.ArgumentParser(description="OpenP5 Custom Inference")


    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)

    # args for inference
    parser.add_argument(
        "--backbone", type=str, default="t5-small",
        help="backbone model (t5-small)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="checkpoint path"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="name of the dataset used for TestDataset"
    )
    parser.add_argument(
        "--task", type=str, default="test",
        help="task name for TestDataset"
    )
    parser.add_argument(
        "--user_history", type=str, required=True,
        help="History for recommendation"
    )
    parser.add_argument(
        "--max_length", type=int, default=50,
        help="generate max_length"
    )
    parser.add_argument(
        "--generate_num", type=int, default=5,
        help="num_beams and num_return_sequences"
    )


    # (C) TestDataset-dependent args (If missing, AttributeError)
    parser.add_argument("--candidate_neg_num", type=int, default=0)
    parser.add_argument("--candidate_sep", type=str, default=" , ")
    parser.add_argument("--test_filtered", type=int, default=0)
    parser.add_argument("--test_filtered_batch", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract vocab_size from checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    vocab_size = ckpt["shared.weight"].size(0)

    config = T5Config.from_pretrained(args.backbone)
    config.vocab_size = vocab_size
    model = P5_T5(config)
    model.load_state_dict(ckpt, strict=False)
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    recs = recommend_items(
        args,
        model,
        tokenizer,
        dataset_name=args.dataset,
        task_name=args.task,
        user_history=args.user_history,
    )

    print("Recommended items (id, log-prob):", recs)


if __name__ == "__main__":
    main()

