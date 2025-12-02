import pandas as pd
import json
import pickle
import argparse
from embed_models import DeepseekOCREmbeddingModel
import torch
from PIL import Image
import io
import tqdm
from torchvision import transforms
import os


def get_queries(file_in):
    query_list, query_indices = [], []
    q_count = 0
    for line in open(file_in, 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        doc_page = item["page_indices"]
        doc_layout = item["layout_indices"]
        for qa in item["questions"]:
            query_list.append(qa["Q"])
            # tuple of question index, start/end indices of doc
            query_indices.append((q_count, *doc_page, *doc_layout))
            q_count += 1
    return query_list, query_indices


def get_pages(file_in, mode="vlm_text"):
    q_list, q_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        q_list.append(row[mode])
        q_indices.append(row_index)
    return q_list, q_indices


def get_layouts(file_in, mode="vlm_text"):
    q_list, q_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        # page_size = row["page_size"]
        if mode == "image_binary":
            q_list.append(row["image_binary"])
        else:
            if layout_type in ["table", "image"]: q_list.append(row[mode])
            else: q_list.append(row["text"])
        q_indices.append((row_index, page_id, *bbox))
    return q_list, q_indices


def get_layouts_hybrid(file_in):
    q_img_list, q_img_indices = [], []
    q_txt_list, q_txt_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        if layout_type in ["table", "image"]: 
            q_img_list.append(row["image_binary"])
            q_img_indices.append((row_index, page_id, *bbox))
        else:
            q_txt_list.append(row["text"])
            q_txt_indices.append((row_index, page_id, *bbox))
    return q_img_list, q_img_indices, q_txt_list, q_txt_indices


# def get_retriever(model, bs):
#     if model == "BGE":
#         from text_wrapper import BGE
#         bs = bs if bs != -1 else 256
#         return BGE(bs=bs)
#     elif model == "E5":
#         from text_wrapper import E5
#         bs = bs if bs != -1 else 256
#         return E5(bs=bs)
#     elif model == "GTE":
#         from text_wrapper import GTE
#         bs = bs if bs != -1 else 256
#         return GTE(bs=bs)
#     elif model == "Contriever":
#         from text_wrapper import Contriever
#         bs = bs if bs != -1 else 256
#         return Contriever(bs=bs)
#     elif model == "DPR":
#         from text_wrapper import DPR
#         bs = bs if bs != -1 else 256
#         return DPR(bs=bs)
#     elif model == "ColBERT":
#         from text_wrapper import ColBERTReranker
#         bs = bs if bs != -1 else 256
#         return ColBERTReranker(bs=bs)
#
#     elif model == "ColPali":
#         from vision_wrapper import ColPaliRetriever
#         bs = bs if bs != -1 else 10
#         return ColPaliRetriever(bs=bs)
#
#     elif model == "ColQwen":
#         from vision_wrapper import ColQwen2Retriever
#         bs = bs if bs != -1 else 8
#         return ColQwen2Retriever(bs=bs)
#
#     elif model == "DSE-docmatix":
#         from vision_wrapper import DSE
#         bs = bs if bs != -1 else 2
#         return DSE(model_name="checkpoint/dse-phi3-docmatix-v2", bs=bs)
#
#     elif model == "DSE-wikiss":
#         from vision_wrapper import DSE
#         bs = bs if bs != -1 else 2
#         return DSE(model_name="checkpoint/dse-phi3-v1", bs=bs)
#
#     else:
#         raise ValueError("the model name is not correct!")





def get_args():
    parser = argparse.ArgumentParser(description="Finetune Deepseek OCR into an embedding model")

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=128)
    # Lora parameters
    parser.add_argument("--save_dir", type=str, default="none")
    parser.add_argument("--ckpt_dir", type=str, default="none")
    parser.add_argument("--init_weights", type=str, default="default", help="How to initialize LoRA weights")
    parser.add_argument("--target_modules", type=str, nargs="+", help="List of module names to apply LoRA to")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_dim", type=int, default=8,
                        help="LoRA rank dimension r")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def get_embedding_model(args, do_debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = DeepseekOCREmbeddingModel(
        init_weights=args.init_weights,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_dim=args.lora_dim,
        use_dora=args.use_dora,
        debug= do_debug,
        embed_dim=args.embed_dim,
    ).to(device)
    if not do_debug:
        embed_model.load_from_ckpt(args.ckpt_dir, device=device)

    embed_model.eval()
    embed_model.to(device)
    return embed_model


def main():
    args = get_args()

    embed_model = get_embedding_model(args, do_debug=args.debug)   # must have batch encode
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Load Data =====
    query_list, query_indices = get_queries("./MMDocIRDataset/evaluate_dataset/MMDocIR_annotations.jsonl")
    quote_list, quote_indices = get_pages("./MMDocIRDataset/evaluate_dataset/MMDocIR_pages.parquet", mode="image_binary")

    # B = args.batch_size if hasattr(args, "batch_size") else 2

    # =========================
    # Encode Queries in Batch
    # =========================
    print(f"Encoding {len(query_list)} queries with batch={1} ...")
    all_q_embs = []
    start = 0
    B = 1
    while start < len(query_list):
        batch_queries = query_list[start : min(start + B, len(query_list))]
        with torch.no_grad():
            batch_emb = embed_model.encode_text(batch_queries)  # [B, D]
        all_q_embs.append(batch_emb.cpu())
        start = min(start + B, len(query_list))

    # query_embs = torch.stack(all_q_embs, dim=0)  # [Nq, D]

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/encoded_queries.pkl", "wb") as f:
        pickle.dump((all_q_embs, query_indices), f)
    print(f"[✓] Saved query embeddings to {args.save_dir}/encoded_queries.pkl")

    # =========================
    # Encode Images in Batch
    # =========================
    B = args.batch_size if hasattr(args, "batch_size") else 2
    print(f"Encoding {len(quote_list)} images with batch={B} ...")
    all_img_embs = []
    start = 0
    transform = transforms.Compose([transforms.ToTensor()])
    while start < len(quote_list):
        batch_imgs_bytes = quote_list[start : min(start+ B, len(quote_list))]

        # bytes → PIL list
        batch_tensor = []
        for b in batch_imgs_bytes:
            b_pil = Image.open(io.BytesIO(b)).convert("RGB")
            b_tensor = transform(b_pil.resize((args.image_size, args.image_size)))
            batch_tensor.append(b_tensor)
        batch_tensor = torch.stack(batch_tensor, dim=0)


        with torch.no_grad():
            batch_emb = embed_model.encode_image(batch_tensor)  # [B, D]
        all_img_embs.append(batch_emb.cpu())

        start = min(start+ B, len(quote_list))


    img_embs = torch.cat(all_img_embs, dim=0)  # [Np, D]

    with open(f"{args.save_dir}/encoded_pages.pkl", "wb") as f:
        pickle.dump((img_embs, quote_indices), f)
    print(f"[✓] Saved image embeddings to {args.save_dir}/encoded_pages.pkl")

    print("\n✅ All done!")
if __name__ == "__main__":
    main()