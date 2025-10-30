import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
'''
이 파일은 질문생성 파일임
처음 여러개의 답변을 생성해서 결과폴더에 저장하는 역할
'''

sys.path.append('/data3/jykim/Projects/CCQA_official')
from LLM_runner import LLMRunner

def load_gsm8k_dataset() -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face datasets
    
    Returns:
        List of GSM8K questions with processed fields
    """
    # Load GSM8K dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")
    
    # Use the 'test' split as specified
    data = dataset["test"]
    
    formatted_questions = []
    for item in data:
        question_dict = {
            "question": item.get("question", ""),
            "final_ans": item.get("answer", "")
        }
        
        # Format the prompt for the model using the specified format with few-shot examples
        prompt = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {question_dict["question"]} A:"""
        
        question_dict["prompt"] = prompt
        formatted_questions.append(question_dict)
    
    return formatted_questions

def load_existing_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load existing results from a file if it exists
    
    Args:
        results_path: Path to the results file
        
    Returns:
        List of existing result items or empty list if file doesn't exist
    """
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different result formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "results" in data:
                return data["results"]
            else:
                return []
        except Exception as e:
            print(f"Error loading existing results from {results_path}: {e}")
            # Create a backup of the problematic file
            if os.path.exists(results_path):
                backup_path = f"{results_path}.bak.{int(time.time())}"
                try:
                    os.rename(results_path, backup_path)
                    print(f"Created backup of problematic file at {backup_path}")
                except Exception as be:
                    print(f"Failed to create backup: {be}")
            return []
    return []

def run_gsm8k_benchmark(
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    parallel: bool = True,
    total_responses: int = 10  # 각 문제당 총 필요한 응답 수
) -> str:
    """
    특정 모델에 대해 GSM8K 데이터셋 벤치마크를 실행하고 결과 저장
    
    Args:
        model_name: 사용할 모델 이름
        output_dir: 결과 저장 디렉토리
        max_new_tokens: 생성할 최대 토큰 수
        temperature: 생성 온도
        top_p: Top-p 샘플링 파라미터
        parallel: 병렬로 응답 생성 여부
        total_responses: 각 문제당 총 필요한 응답 수
        
    Returns:
        저장된 결과 파일 경로
    """
    # Hugging Face에서 데이터셋 로드
    questions = load_gsm8k_dataset()
    print(f"Loaded {len(questions)} questions from GSM8K test dataset")
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 파일 생성
    results_filename = f"gsm8k_{model_name}.json"
    results_path = os.path.join(output_dir, results_filename)
    
    # 기존 결과 확인
    existing_results = load_existing_results(results_path)
    
    # 모든 질문에 대한 처리가 필요함 (건너뛰지 않음)
    questions_to_process = questions
    
    print(f"Will process {len(questions_to_process)} questions for {model_name}")
    
    # 모델 초기화
    runner = LLMRunner(model_name)
    
    # 기존 결과 복사
    all_results = existing_results.copy()
    
    # tqdm 진행 바와 함께 각 질문 처리
    for q_idx, question in enumerate(tqdm(questions_to_process, desc=f"Processing {model_name}")):
        # 이미 생성된 응답 확인
        existing_item = None
        if q_idx < len(all_results):
            existing_item = all_results[q_idx]
        
        # 새로운 항목 생성 또는 기존 항목 복사
        if existing_item is None:
            result_item = {
                "question": question["question"],
                "original_answer": question["final_ans"]
            }
        else:
            result_item = existing_item
        
        # 이미 생성된 응답 개수 확인
        existing_response_count = 0
        for key in result_item.keys():
            if key.startswith("response_"):
                existing_response_count += 1
        
        # 필요한 추가 응답 개수 계산
        responses_to_add = max(0, total_responses - existing_response_count)
        
        if responses_to_add > 0:
            print(f"Generating {responses_to_add} additional responses for question {q_idx+1}")
            try:
                # 추가 응답 생성
                new_responses = runner.generate_responses(
                    prompt=question["prompt"],
                    num_responses=responses_to_add,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    parallel=parallel
                )
                
                # 새 응답 추가, 번호는 이어서
                for i, response in enumerate(new_responses, existing_response_count + 1):
                    result_item[f"response_{i}"] = response
                
                # 결과 항목 업데이트 또는 추가
                if q_idx < len(all_results):
                    all_results[q_idx] = result_item
                else:
                    all_results.append(result_item)
                
                # 각 항목 후 전체 결과 리스트 저장
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                
                # 주기적으로 진행 상황 출력
                if (q_idx + 1) % 10 == 0 or q_idx == 0 or q_idx == len(questions) - 1:
                    print(f"Completed {q_idx + 1}/{len(questions)} questions ({((q_idx + 1) / len(questions) * 100):.1f}%)")
                    
            except Exception as e:
                error_msg = f"Error generating responses for question {q_idx+1}: {e}"
                print(error_msg)
                
                # 오류가 있어도 진행 상황 저장
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                continue
        else:
            print(f"Question {q_idx+1} already has {existing_response_count} responses, skipping")
    
    print(f"Benchmark completed. Results saved to {results_path}")
    return results_path

def run_all_models_parallel(
    output_dir: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Optional[str]]:
    
    available_models = list(LLMRunner.AVAILABLE_MODELS.keys())
    results = {}
    
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit jobs for each model
        future_to_model = {}
        for model_name in available_models:
            future = executor.submit(
                run_gsm8k_benchmark,  # Updated function name
                model_name,
                output_dir,
                max_new_tokens,
                temperature,
                top_p,
                True  # Use parallel response generation
            )
            future_to_model[future] = model_name
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                output_path = future.result()
                results[model_name] = output_path
                print(f"Completed model: {model_name}")
            except Exception as e:
                print(f"Error running model {model_name}: {e}")
                results[model_name] = None
    
    return results


OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/gsm8k_numgeneration_20"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
RUN_PARALLEL = True

# Main execution
if __name__ == "__main__":
    if RUN_PARALLEL:
        print("Starting parallel benchmark for all models...")
        results = run_all_models_parallel(
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        
        # Report results
        print("\nBenchmark completed. Results summary:")
        for model_name, output_path in results.items():
            if output_path:
                print(f"SUCCESS: {model_name}: {output_path}")
            else:
                print(f"FAILED: {model_name}")
    else:
        # Run for a single model
        model_name = "gemma-1b"  # Change to desired model
        print(f"Running benchmark for model: {model_name}")
        output_path = run_gsm8k_benchmark(  # Updated function name
            model_name=model_name,
            output_dir=OUTPUT_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            parallel=True  
        )
        print(f"Benchmark completed. Results saved to: {output_path}")