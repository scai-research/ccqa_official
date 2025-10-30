import os, sys
import json
import re
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 모델 경로 및 프롬프트 설정
MODEL_PATH = "/home/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-qgen"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 수치 정확성 프롬프트 템플릿
QUESTION_PROMPT_TEMPLATE = (
    "CRITICAL: Do not change ANY numeric values in the answer. "
    "Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. "
    "Generate a question that would have this as its answer: {}"
)

class T5QuestionGenerator:
    def __init__(self, model_path: str = MODEL_PATH):
        """T5 모델을 초기화합니다."""
        print(f"Loading model from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.device = DEVICE
        print(f"Model loaded on {self.device}")

    def generate_question(self, 
                           answer: str, 
                           max_new_tokens: int = 100, 
                           temperature: float = 0.2, 
                           top_p: float = 0.9) -> str:
        """답변에서 질문을 생성합니다."""
        if not answer:
            return ""

        prompt = QUESTION_PROMPT_TEMPLATE.format(answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_similarity(self, 
                            original_question: str, 
                            generated_questions: List[str],
                            max_new_tokens: int = 50) -> Tuple[int, str]:

        # 객관식 형태의 프롬프트를 구성 (1-5 범위)
        options_prompt = "Which generated question (1-5) is most similar to the original question?\n\n"
        options_prompt += f"Original: {original_question}\n\n"
        
        for i, question in enumerate(generated_questions, 1):
            if not question:
                options_prompt += f"{i}: [No question generated]\n"
            else:
                options_prompt += f"{i}: {question}\n"
        
        options_prompt += "\nAnswer with just the number (1-5):"
        
        inputs = self.tokenizer(options_prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                num_beams=4,
                early_stopping=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        clean_response = re.sub(re.escape(options_prompt) + r"\s*", "", full_response)

        # 응답에서 숫자 추출
        digit_match = re.search(r'\b([1-5])\b', clean_response)
        if digit_match:
            return int(digit_match.group(1)), clean_response
        
        # 숫자를 찾지 못한 경우 기본값 반환
        return 1, clean_response  # 기본값은 첫 번째 질문

    def extract_answer(self, 
                       question: str, 
                       response: str,
                       is_mcq: bool = False,
                       max_new_tokens: int = 30) -> str:
        """응답에서 정답을 추출합니다."""
        if is_mcq:
            prompt = f"Extract the correct answer from this response. Question: {question} Generated response: {response}"
        else:
            prompt = f"Extract the numerical answer from this solution. Question: {question} Solution: {response}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_benchmark_results(results_path: str) -> List[Dict[str, Any]]:
    """벤치마크 결과 파일을 로드합니다."""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return []


def generate_questions_from_answers(
    generator: T5QuestionGenerator,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 100,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> List[Dict[str, Any]]:
    """답변에서 질문을 재생성합니다."""
    ccqa_results = []
    
    for i, item in enumerate(tqdm(results, desc="Generating questions")):
        ccqa_item = item.copy()
        
        # 답변에서 질문 재생성
        for resp_idx in range(1, 6):
            response_key = f"response_{resp_idx}"
            if response_key not in item:
                continue
                
            answer = item[response_key]
            
            if not answer:
                ccqa_item[f"generated_question_{resp_idx}"] = None
                continue
            
            try:
                regenerated_question = generator.generate_question(
                    answer, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                ccqa_item[f"generated_question_{resp_idx}"] = regenerated_question
                
            except Exception as e:
                print(f"Error regenerating question for response {resp_idx}: {e}")
                ccqa_item[f"generated_question_{resp_idx}"] = f"Error: {str(e)}"
        
        ccqa_results.append(ccqa_item)
        
        # 진행 상황 저장
        current_results = {
            "completed_questions": i + 1,
            "total_questions": len(results),
            "results": ccqa_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(results) - 1:
            print(f"Progress: {i + 1}/{len(results)} questions ({((i + 1) / len(results) * 100):.1f}%)")
    
    return ccqa_results


def evaluate_similarities(
    generator: T5QuestionGenerator,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 50
) -> List[Dict[str, Any]]:
    """원본 질문과 재생성된 질문 간의 유사성을 평가합니다."""
    updated_results = []
    
    for i, item in enumerate(tqdm(results, desc="Evaluating similarities")):
        updated_item = item.copy()
        
        if "question" not in item or not item["question"]:
            updated_results.append(updated_item)
            continue
            
        # 재생성된 질문들 수집
        regenerated_questions = []
        for idx in range(1, 6):
            key = f"generated_question_{idx}"
            if key in item and item[key]:
                regenerated_questions.append(item[key])
            else:
                regenerated_questions.append("")
        
        # 재생성된 질문이 충분하지 않으면 건너뜀
        if len([q for q in regenerated_questions if q]) < 2:
            updated_results.append(updated_item)
            continue
            
        # 유사성 평가
        try:
            most_similar_idx, full_response = generator.evaluate_similarity(
                item["question"],
                regenerated_questions,
                max_new_tokens=max_new_tokens
            )
            
            # 결과 저장
            updated_item["most_similar_idx"] = most_similar_idx
            # 전체 응답 저장
            updated_item["similarity_full_response"] = full_response
            
        except Exception as e:
            print(f"Error evaluating similarity for item {i}: {e}")
        
        updated_results.append(updated_item)
        
        # 정기적으로 결과 저장
        if (i + 1) % 10 == 0 or i == len(results) - 1:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": updated_results}, f, ensure_ascii=False, indent=2)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(results)} questions ({((i + 1) / len(results) * 100):.1f}%)")
    
    return updated_results


def extract_answer_from_similarity(
    generator: T5QuestionGenerator,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 30
) -> List[Dict[str, Any]]:
    """유사성이 가장 높은 응답에서 정답을 추출합니다."""
    updated_results = []
    
    for i, item in enumerate(tqdm(results, desc="Extracting answers")):
        updated_item = item.copy()
        
        # similarity_idx가 있는지 확인
        if "most_similar_idx" not in item:
            print(f"Item {i} does not have a most_similar_idx field. Skipping.")
            updated_results.append(updated_item)
            continue
            
        similarity_idx = item["most_similar_idx"]
        response_key = f"response_{similarity_idx}"
        
        # 유사성이 가장 높은 응답이 있는지 확인
        if response_key not in item or not item[response_key]:
            print(f"Item {i} does not have {response_key} field. Skipping.")
            updated_results.append(updated_item)
            continue
            
        # 유사성이 가장 높은 응답 및 원본 질문 추출
        similar_response = item[response_key]
        original_question = item.get("question", "")
        
        # 데이터셋 유형 확인 (있는 경우)
        dataset_type = item.get("dataset_type", "")
        is_mcq = dataset_type == "CommonSenseQA" or "choice" in original_question.lower() or "option" in original_question.lower()
        
        try:
            extracted_answer = generator.extract_answer(
                original_question,
                similar_response,
                is_mcq=is_mcq,
                max_new_tokens=max_new_tokens
            )
            
            # 결과 저장
            extracted_clean = extracted_answer.strip()
            updated_item["extracted_answer"] = extracted_clean
            updated_item["used_response"] = response_key
            
        except Exception as e:
            print(f"Error extracting answer for item {i}: {e}")
            updated_item["extraction_error"] = str(e)
        
        updated_results.append(updated_item)
        
        # 정기적으로 결과 저장
        save_interval = 10  # 10개 항목마다 저장
        if (i + 1) % save_interval == 0 or i == 0 or i == len(results) - 1:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": updated_results}, f, ensure_ascii=False, indent=2)
            
            if (i + 1) % save_interval == 0:
                print(f"Progress: {i + 1}/{len(results)} items ({((i + 1) / len(results) * 100):.1f}%)")
    
    print(f"\n처리 완료! 총 {len(updated_results)}개 항목의 정답을 추출했습니다.")
    return updated_results
