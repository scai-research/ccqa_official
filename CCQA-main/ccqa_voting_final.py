import os
import json
import re
from collections import Counter
from tqdm import tqdm
import csv
import random
import numpy as np
'''
이 파일은 CCQA 정확도 실행 파일임
먼저 generation 파일로 답변들 생성, ccqa파일로 질문 생성, similarity_score파일로 유사도 계산 후
사용하는 파일임임
'''
# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def extract_numerical_answer(response):
    if not response:
        return None
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)',
        r'the (?:correct )?answer is\s*:?\s*\(?([A-E])\)?',
        r'(?:correct )?answer is\s*:?\s*\(?([A-Ea-e])\)?',
    ]
    text = response.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    return None

def is_answer_correct(extracted_answer, correct_answer):
    if extracted_answer is None or correct_answer is None:
        return False
    extracted_numeric = re.sub(r'[^\d.]', '', str(extracted_answer).strip())
    correct_numeric = re.sub(r'[^\d.]', '', str(correct_answer).strip())
    try:
        if extracted_numeric and correct_numeric:
            return abs(float(extracted_numeric) - float(correct_numeric)) < 1e-5
        else:
            return str(extracted_answer).strip() == str(correct_answer).strip()
    except ValueError:
        return str(extracted_answer).strip() == str(correct_answer).strip()

def evaluate_cot_response(results):
    """첫 번째 응답만 사용하여 CoT 평가"""
    for item in tqdm(results, desc="evaluating CoT"):
        item["cot_answer"] = extract_numerical_answer(item.get("response_1"))
    return results

def apply_original_self_consistency(results):
    """기존 self-consistency: 가장 많이 나온 답변 = 정답"""
    for item in tqdm(results, desc="original self-consistency"):
        responses = [item.get(f"response_{i}") for i in range(1,4) if item.get(f"response_{i}")]
        if not responses:
            item["original_sc_answer"] = None
            continue
        
        extracted_answers = [extract_numerical_answer(r) for r in responses]
        item["extracted_answers"] = extracted_answers
        counts = Counter(a for a in extracted_answers if a)
        top = counts.most_common(1)
        item["original_sc_answer"] = top[0][0] if top else None
    return results

def apply_conditional_unified_voting(results, alpha=0.5, beta=0.5):
    """조건부 통합 방식"""
    for item in tqdm(results, desc="conditional unified voting"):
        # 응답/선택지 추출
        responses = [(i, extract_numerical_answer(item.get(f"response_{i}")))
                     for i in range(1, 4) if item.get(f"response_{i}")]
        valid = [(idx,a) for (idx,a) in responses if a]
        
        if not valid:
            item["ccqa_answer"] = None
            item["ccqa_method"] = "no_valid_response"
            continue

        # 득표 계산
        c = Counter(ans for (_,ans) in valid)
        top_choice, top_count = c.most_common(1)[0]

        # 조건부 판단
        if top_count >= 2:
            # 과반수면 그대로
            item["ccqa_answer"] = top_choice
            item["ccqa_method"] = "majority_vote"
        else:
            # 빈도+유사도 통합으로 전체 스코어 계산
            all_choices = list(c.keys())
            final_scores = {}
            
            for choice in all_choices:
                rel_sum = 0.0
                for idx, ans in valid:
                    if ans == choice:
                        sim = item.get("similarity_scores", {}).get(f"question_{idx}", {})
                        cos = sim.get("cosine", 0.0)
                        bleu = sim.get("bleu", 0.0)
                        rel_score = alpha*bleu + beta*cos
                        rel_sum += rel_score
                
                final_scores[choice] = rel_sum

            best_choice = max(final_scores, key=final_scores.get)
            item["ccqa_answer"] = best_choice
            item["ccqa_method"] = "freq+similarity_unified"
    
    return results

def process_file(file_path, alpha=0.5, beta=0.5):
    filename = os.path.basename(file_path)
    model = filename.replace(".json","")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data["results"] if isinstance(data, dict) and "results" in data else data

    # 평가 단계들 적용
    results = evaluate_cot_response(results)
    results = apply_original_self_consistency(results)
    results = apply_conditional_unified_voting(results, alpha=alpha, beta=beta)

    # 정확도 계산
    total = len(results)
    cot_correct, orig_correct, ccqa_correct = 0, 0, 0
    
    for item in results:
        c_ans = (item.get("correct_answer") or item.get("original_answer"))
        
        if is_answer_correct(item.get("cot_answer"), c_ans):
            cot_correct += 1
        if is_answer_correct(item.get("original_sc_answer"), c_ans):
            orig_correct += 1
        if is_answer_correct(item.get("ccqa_answer"), c_ans):
            ccqa_correct += 1

    return {
        "model_name": model,
        "total_items": total,
        "cot_acc": cot_correct / total if total > 0 else 0,
        "orig_acc": orig_correct / total if total > 0 else 0,
        "ccqa_acc": ccqa_correct / total if total > 0 else 0,
        "processed_results": results
    }

def create_comparison_csv(all_results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "cot_acc", "orig_acc", "ccqa_acc",
            "cot_to_sc_imp", "cot_to_ccqa_imp", "sc_to_ccqa_imp",
            "total_items"
        ])
        writer.writeheader()
        for row in all_results:
            cot_to_sc_imp = row["orig_acc"] - row["cot_acc"]
            cot_to_ccqa_imp = row["ccqa_acc"] - row["cot_acc"]
            sc_to_ccqa_imp = row["ccqa_acc"] - row["orig_acc"]
            
            writer.writerow({
                "model_name": row["model_name"],
                "cot_acc": f"{row['cot_acc']:.4f}",
                "orig_acc": f"{row['orig_acc']:.4f}",
                "ccqa_acc": f"{row['ccqa_acc']:.4f}",
                "cot_to_sc_imp": f"{cot_to_sc_imp:.4f}",
                "cot_to_ccqa_imp": f"{cot_to_ccqa_imp:.4f}",
                "sc_to_ccqa_imp": f"{sc_to_ccqa_imp:.4f}",
                "total_items": row["total_items"]
            })
    print(f"CSV 저장 완료: {out_path}")

def main():
    input_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_result_10/precision_nli"  # 입력 경로 수정 필요
    output_dir = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/ccqa_result_10/precision_nli"  # 출력 경로 수정 필요
    os.makedirs(output_dir, exist_ok=True)
    
    # CCQA 파라미터
    alpha = 0.4
    beta = 0.6
    
    all_results = []
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for file_name in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(input_dir, file_name)
        res = process_file(file_path, alpha=alpha, beta=beta)
        all_results.append({
            "model_name": res["model_name"],
            "cot_acc": res["cot_acc"],
            "orig_acc": res["orig_acc"],
            "ccqa_acc": res["ccqa_acc"],
            "total_items": res["total_items"]
        })
    
    out_csv = os.path.join(output_dir, "method_comparison.csv")
    create_comparison_csv(all_results, out_csv)
    
    print(f"✓ 모든 처리 완료. 결과는 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()