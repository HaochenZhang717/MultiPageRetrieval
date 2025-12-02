import pickle
import json
import random
import numpy as np
from metric_eval import evaluate_page, evaluate_layout
from tqdm import tqdm
import argparse


def batch_dot_product(query_vec, passage_vecs):
    return passage_vecs @ query_vec


def load_pickle(file_in):
    # Load pickled files
    with open(file_in, "rb") as fq:
        return pickle.load(fq)


def initialize_args():
    '''
    Example: encode.py BGE --encode query,page,layout
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name, e.g. BGE')
    parser.add_argument('--encode_path', type=str, default='encode')
    parser.add_argument('--encode', type=str, default="query,page,layout")
    return parser.parse_args()


if __name__ == "__main__":
    # ["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT"]

    # args = initialize_args()
    # model, encode, encode_path = args.model, args.encode, args.encode_path

    # if model.startswith("Col"):
    encode = ["page"]
    from metric_eval import colbert_score, pad_tok_len

    for finetune_epoch in [2]:
        # encode_path = f"./embeds_1110-embed128-lr1e-4-r32-Epoch{finetune_epoch}"
        encode_path = f"./embeds_ckpts_embed128-lr1e-4-r64-size1280-temperature0.05-Epoch{finetune_epoch}"

        encoded_query, query_indices = load_pickle(f"{encode_path}/encoded_queries.pkl")

        if "page" in encode:
            encoded_page, page_indices = load_pickle(f"{encode_path}/encoded_pages.pkl")

        # if "layout" in encode:
        #     encoded_layout, layout_indices = load_pickle(f"{encode_path}/encoded_layout_{model}.pkl")

        gt_list = []
        for line in open("./MMDocIRDataset/evaluate_dataset/MMDocIR_annotations.jsonl", 'r', encoding="utf-8"):
            item = json.loads(line.strip())
            for qa in item["questions"]:
                qa["domain"] = item["domain"]
                gt_list.append(qa)

        if len(gt_list) != len(query_indices):
            raise ValueError("number of indexed question do not match ground-truth")

        # To do this for every query in query_indices:
        for (query_id, start_pid, end_pid, start_lid, end_lid) in tqdm(query_indices):
            query_vec = encoded_query[query_id]
            query_vec = query_vec.squeeze(0).float().numpy()

            if "page" in encode:
                page_vecs = encoded_page[start_pid:end_pid + 1].float().numpy()
                irrelevant = [i for i in range(len(encoded_page)) if i<start_pid or i>end_pid]
                sampled_irrelevant = random.choices(irrelevant, k=2000)
                # sampled_irrelevant = irrelevant
                irrelavant_page_vecs = encoded_page[sampled_irrelevant].float().numpy()

                page_vecs = np.concatenate((page_vecs, irrelavant_page_vecs), axis=0)
                # if not model.startswith("Col"):
                #     scores_page = batch_dot_product(query_vec, page_vecs)
                # else:
                #     page_vecs_pad, masks_page = pad_tok_len(page_vecs)
                #     scores_page = colbert_score(query_vec, page_vecs_pad, masks_page)

                page_vecs_pad, masks_page = pad_tok_len(page_vecs)
                scores_page = colbert_score(query_vec, page_vecs_pad, masks_page)
                gt_list[query_id]["scores_page"] = scores_page.tolist()


        if "page" in encode:
            evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=1, metric="recall")
            evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=3, metric="recall")
            evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=5, metric="recall")

        # if "layout" in encode:
        #     evaluate_layout(gt_list, model_name=model, topk=1, metric="recall")
        #     evaluate_layout(gt_list, model_name=model, topk=5, metric="recall")
        #     evaluate_layout(gt_list, model_name=model, topk=10, metric="recall")
