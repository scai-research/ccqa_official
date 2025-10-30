import os
import json
import glob
import torch
import pickle
from tqdm import tqdm
import traceback
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import pipeline
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore", message="It is recommended to enable `effective_order`")

'''
이 코드는 생성된 질문들의 유사도를 계산하는 파일임
BLEU, Rouge, cosine 등 여러가지 다 측정함함
'''
# ===== 전역 변수 =====
MAX_WORKERS = 5            # CPU 코어 수에 맞게 조정
DEVICE_ID   = 0           # 기본 GPU ID

# ===== 모델 초기화 함수 =====
def initialize_models(device_id: int = 0):
    """각 프로세스에서 호출되는 모델·스코어러 초기화"""
    print(f"[PID {os.getpid()}] Initializing models (GPU {device_id})")

    models = {
        "nli_pipeline": pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=device_id if torch.cuda.is_available() else -1
        ),
        "bert_scorer": BERTScorer(
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            idf=False
        ),
        "sbert_model": SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "contrastive_model": SentenceTransformer(
            "BAAI/bge-large-en-v1.5"
        ),
        # ─── NEW: BLEU / ROUGE scorers ────────────────────────────────────────
        "bleu_scorer": BLEU(effective_order=True),                                            # 무상태
        "rouge_scorer": rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        ),
        # ────────────────────────────────────────────────────────────────────
    }

    # GPU 메모리 분산
    if torch.cuda.is_available():
        models["sbert_model"] = models["sbert_model"].to(f"cuda:{device_id}")
        models["contrastive_model"] = models["contrastive_model"].to(
            f"cuda:{device_id}"
        )

    return models


# ===== 단일 파일 처리 함수 =====
def process_file(
    json_file: str,
    input_directory: str,
    output_directory: str,
    device_id: int = 0,
):
    """각 JSON 파일에 대해 유사도 지표 계산"""
    filename = os.path.basename(json_file)
    try:
        models = initialize_models(device_id)
        output_file = os.path.join(output_directory, filename)

        print(f"[PID {os.getpid()}] Processing {filename}")

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        if "results" not in data:
            return {"status": "skipped", "filename": filename,
                    "reason": "'results' key not found"}

        for result in data["results"]:
            original_question = result.get("question", "").strip()
            if not original_question:
                continue

            # ── generated questions 추출 ─────────────────────────────────────
            generated_questions, gen_keys = [], []
            for j in range(1, 21):
                key = f"generated_question_{j}"
                if result.get(key):
                    generated_questions.append(result[key])
                    gen_keys.append(j)
            if not generated_questions:
                continue

            # ── 1. BERTScore Precision ──────────────────────────────────────
            P, _, _ = models["bert_scorer"].score(
                [original_question] * len(generated_questions),
                generated_questions,
            )
            precision_scores = [p.item() for p in P]

            # ── 2. NLI entailment ───────────────────────────────────────────
            entailment_scores = []
            for gen_q in generated_questions:
                back_txt = f"{original_question} </s> {gen_q}"
                fwd_txt  = f"{gen_q} </s> {original_question}"

                back_prob = next(
                    (r["score"] for r in models["nli_pipeline"](back_txt)
                     if r["label"].upper() == "ENTAILMENT"), 0.0
                )
                fwd_prob = next(
                    (r["score"] for r in models["nli_pipeline"](fwd_txt)
                     if r["label"].upper() == "ENTAILMENT"), 0.0
                )
                entailment_scores.append((fwd_prob, back_prob))

            # ── 3. SBERT cosine ─────────────────────────────────────────────
            all_sent = [original_question] + generated_questions
            sbert_emb = models["sbert_model"].encode(all_sent, convert_to_tensor=True)
            cosine_scores = util.cos_sim(sbert_emb[0], sbert_emb[1:]).squeeze().tolist()
            if not isinstance(cosine_scores, list):
                cosine_scores = [cosine_scores]

            # ── 4. BGE contrastive cosine ───────────────────────────────────
            bge_emb = models["contrastive_model"].encode(
                all_sent, convert_to_tensor=True, normalize_embeddings=True
            )
            contrastive_scores = util.cos_sim(bge_emb[0], bge_emb[1:]).squeeze().tolist()
            if not isinstance(contrastive_scores, list):
                contrastive_scores = [contrastive_scores]

            # ── 5. NEW: BLEU & ROUGE‑L ──────────────────────────────────────
            bleu_scores, rougeL_scores = [], []
            for gen_q in generated_questions:
                bleu = (
                    models["bleu_scorer"]
                    .sentence_score(gen_q, [original_question])
                    .score
                    / 100.0          # 0‑1 범위로 정규화
                )
                rougeL_f = models["rouge_scorer"].score(
                    original_question, gen_q
                )["rougeL"].fmeasure
                bleu_scores.append(bleu)
                rougeL_scores.append(rougeL_f)

            # ── 6. 결과 저장 ────────────────────────────────────────────────
            result["similarity_scores"] = {}
            combined_scores = []

            for idx, key_idx in enumerate(gen_keys):
                norm_entail = (entailment_scores[idx][1] + 1) / 2  # [-1,1]→[0,1]
                combined     = 1.0 * precision_scores[idx] + 0.0 * norm_entail

                result["similarity_scores"][f"question_{key_idx}"] = {
                    "precision": round(precision_scores[idx], 4),
                    "entailment_forward": round(entailment_scores[idx][0], 4),
                    "entailment_backward": round(entailment_scores[idx][1], 4),
                    "normalized_entailment": round(norm_entail, 4),
                    "cosine": round(cosine_scores[idx], 4),
                    "contrastive": round(contrastive_scores[idx], 4),
                    # NEW
                    "bleu": round(bleu_scores[idx], 4),
                    "rougeL_f": round(rougeL_scores[idx], 4),
                    "combined_score": round(combined, 4),
                }
                combined_scores.append((key_idx, combined))

            # ── 7. ranking & meta info ──────────────────────────────────────
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            result["most_similar_idxs"] = [idx for idx, _ in combined_scores]
            result["ranking_method"]    = "1.0*Precision"

        # ===== 파일 저장 =====
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved → {output_file}")
        return {"status": "success", "filename": filename}

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        traceback.print_exc()
        return {"status": "error", "filename": filename, "error": str(e)}


# ===== 메인 함수 =====
def main():
    input_directory  = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_qwen_result"
    output_directory = os.path.join(input_directory, "precision_nli")
    os.makedirs(output_directory, exist_ok=True)

    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    print(f"Found {len(json_files)} JSON files")

    num_workers = min(MAX_WORKERS, len(json_files))
    print(f"Using {num_workers} parallel workers")

    # 다중 GPU 환경인 경우 파일별로 GPU 라운드‑로빈 할당
    gpu_map = {f: (i % torch.cuda.device_count() if torch.cuda.is_available() else -1)
               for i, f in enumerate(json_files)}

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file, jf, input_directory,
                            output_directory, gpu_map[jf])
            for jf in json_files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Overall Progress"):
            results.append(fut.result())

    # ===== 결과 요약 =====
    success = sum(r["status"] == "success" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    errors  = sum(r["status"] == "error"   for r in results)

    print("\n===== Processing Complete =====")
    print(f"Success: {success}/{len(json_files)}")
    print(f"Skipped: {skipped}")
    print(f"Errors : {errors}")

    if errors:
        print("\n[Error files]")
        for r in results:
            if r["status"] == "error":
                print(f"  • {r['filename']}: {r['error']}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
