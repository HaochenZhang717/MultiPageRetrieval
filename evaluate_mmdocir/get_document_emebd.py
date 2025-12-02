import pickle
import json
import random
import numpy as np
from metric_eval import evaluate_page, evaluate_layout
from tqdm import tqdm
import argparse
from copy import deepcopy

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
        encode_path = f"/root/tianyi/MMDocIR_embeds/embeds_ckpts_embed128-lr1e-4-r64-size1280-temperature0.05-Epoch{finetune_epoch}"

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

        doc_avg_pool_list = []
        doc_id = 0
        start_end_ids = dict()
        updated_gt_list = deepcopy(gt_list)
        for (query_id, start_pid, end_pid, start_lid, end_lid) in tqdm(query_indices):
            if (start_pid, end_pid) in start_end_ids.keys():
                continue

            query_vec = encoded_query[query_id]
            query_vec = query_vec.squeeze(0).float().numpy()
            if "page" in encode:
                page_vecs = encoded_page[start_pid:end_pid + 1].float().numpy()
                average_page_vec = np.mean(page_vecs, axis=0)

                doc_avg_pool_list.append({
                    "doc_index": doc_id,
                    "start_pid": start_pid,
                    "end_pid": end_pid,
                    "avg_page_embed": average_page_vec,  # convert to list for JSON compatibility
                })
                start_end_ids.update({(start_pid, end_pid): doc_id})
                doc_id += 1

                # page_vecs_pad, masks_page = pad_tok_len(page_vecs)
                # scores_page = colbert_score(query_vec, page_vecs_pad, masks_page)
                # gt_list[query_id]["scores_page"] = scores_page.tolist()
        out_path = f"/root/tianyi/MMDocIR_embeds/embeds_ckpts_embed128-lr1e-4-r64-size1280-temperature0.05-Epoch{finetune_epoch}/doc_avg_pool_embeddings.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(doc_avg_pool_list, f)

        print(f"✔ Saved {len(doc_avg_pool_list)} document avg pool embeddings to: {out_path}")


        for gt_item, (query_id, start_pid, end_pid, start_lid, end_lid)  in zip(updated_gt_list, query_indices):
            print(gt_item.keys())
            gt_item.update({'doc_id':[start_end_ids[(start_pid, end_pid)]]})

        with open("./MMDocIRDataset/evaluate_dataset/questions_w_docID.jsonl", "w", encoding="utf-8") as f:
            for item in updated_gt_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # if "page" in encode:
        #     evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=1, metric="recall")
        #     evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=3, metric="recall")
        #     evaluate_page(gt_list, model_name=f"Epoch{finetune_epoch}", topk=5, metric="recall")

