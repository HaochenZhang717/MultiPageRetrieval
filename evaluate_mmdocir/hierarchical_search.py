import pickle
import json
import random
import numpy as np
from metric_eval import evaluate_page, evaluate_layout, evaluate_doc
from tqdm import tqdm
import argparse
from metric_eval import colbert_score, pad_tok_len, top_k_indices


def batch_dot_product(query_vec, passage_vecs):
    return passage_vecs @ query_vec


def load_pickle(file_in):
    # Load pickled files
    with open(file_in, "rb") as fq:
        return pickle.load(fq)


def retrieve_document(topk, encode_path, document_embed_type):
    # encode_path = f"/root/tianyi/MMDocIR_embeds/embeds_ckpts_embed128-lr1e-4-r64-size1280-temperature0.05-Epoch{finetune_epoch}"
    encoded_query, query_indices = load_pickle(f"{encode_path}/encoded_queries.pkl")
    encoded_document = load_pickle(f"{encode_path}/doc_avg_pool_embeddings.pkl")
    gt_list = []
    for line in open("./MMDocIRDataset/evaluate_dataset/questions_w_docID.jsonl", 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        gt_list.append(item)

    for (query_id, start_pid, end_pid, start_lid, end_lid) in tqdm(query_indices):
        query_vec = encoded_query[query_id]
        query_vec = query_vec.squeeze(0).float().numpy()
        assert document_embed_type in encoded_document[0].keys()
        # document_vecs = [enc['avg_page_embed'] for enc in encoded_document]
        document_vecs = [enc[document_embed_type] for enc in encoded_document]
        # document_vecs = np.stack(document_vecs, axis=0)
        breakpoint()
        document_vecs_pad, masks_document = pad_tok_len(document_vecs)
        scores_document = colbert_score(query_vec, document_vecs_pad, masks_document, use_gpu=True)
        gt_list[query_id]["scores_doc"] = scores_document.tolist()
        gt_list[query_id]["topk_docs"] = top_k_indices(gt_list[query_id]["scores_doc"], topk)

    # evaluate_doc(gt_list, model_name=f"Document Retrieval", topk=1, metric="recall")
    # evaluate_doc(gt_list, model_name=f"Document Retrieval", topk=3, metric="recall")
    # evaluate_doc(gt_list, model_name=f"Document Retrieval", topk=5, metric="recall")

    return gt_list



def retrieve_page(encode_path, gt_list):
    encoded_query, query_indices = load_pickle(f"{encode_path}/encoded_queries.pkl")
    encoded_page, page_indices = load_pickle(f"{encode_path}/encoded_pages.pkl")

    if len(gt_list) != len(query_indices):
        raise ValueError("number of indexed question do not match ground-truth")

    mmdocir_annotations = []
    for line in open("./MMDocIRDataset/evaluate_dataset/MMDocIR_annotations.jsonl", 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        mmdocir_annotations.append(item)

    for (query_id, start_pid, end_pid, start_lid, end_lid) in tqdm(query_indices):
        query_vec = encoded_query[query_id]
        query_vec = query_vec.squeeze(0).float().numpy()
        candidate_docs = gt_list[query_id]["topk_docs"]
        candidate_pages = []
        for candidate_doc in candidate_docs:
            start, end = mmdocir_annotations[candidate_doc]["page_indices"]
            # mmdocir_annotations[70]["page_indices"]
            # mmdocir_annotations[62]["page_indices"]
            candidate_pages.extend(range(start, end + 1))
        page_vecs = encoded_page[candidate_pages].float().numpy()
        page_vecs_pad, masks_page = pad_tok_len(page_vecs)

        # after this, update the "page_id" in gt_list
        gt_list[query_id]['global_page_id'] = [a + start_pid - 1 for a in gt_list[query_id]['page_id']]
        # print(f"candidate pages: {candidate_pages}")

        for i, each_global_page_id in enumerate(gt_list[query_id]['global_page_id']):
            if each_global_page_id in candidate_pages:
                gt_list[query_id]["page_id"][i] = candidate_pages.index(each_global_page_id)
            else:
                gt_list[query_id]["page_id"][i] = -1

        scores_page = colbert_score(query_vec, page_vecs_pad, masks_page, use_gpu=True)
        gt_list[query_id]["scores_page"] = scores_page.tolist()


    evaluate_page(gt_list, model_name=f"||", topk=1, metric="recall")
    evaluate_page(gt_list, model_name=f"||", topk=3, metric="recall")
    evaluate_page(gt_list, model_name=f"||", topk=5, metric="recall")

def main(topk_doc, document_embed_type):
    assert document_embed_type in ["avg_page_embed", "cat_page_embed"]
    print(f"Hierarchical Retrieval with {topk_doc} documents...")
    finetune_epoch = 2
    encode_path = f"/root/tianyi/MMDocIR_embeds/embeds_ckpts_embed128-lr1e-4-r64-size1280-temperature0.05-Epoch{finetune_epoch}"
    updated_gt_list = retrieve_document(topk=topk_doc, encode_path=encode_path, document_embed_type=document_embed_type)
    retrieve_page(encode_path=encode_path, gt_list=updated_gt_list)



if __name__ == "__main__":
    for i in [2, 5, 10, 20]:
        main(i, document_embed_type="cat_page_embed")
