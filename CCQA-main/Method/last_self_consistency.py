import os
import json
import re
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

def load_benchmark_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from a JSON file
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        List of result items
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Check if the results are in the new format with time info
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return []

def extract_numerical_answer(response: str) -> Optional[str]:
    """
    답변의 마지막 문장에서 정답을 찾는 함수
    the answer is , the answer is : 다음 숫자를 찾음
    논문 그대로 사용용
    """
    if not response:
        return None
    
    # Common answer patterns
    patterns = [
        r'the (?:correct )?answer is (?:[$€£¥₩]|\+|−|±|×|÷|=|≈)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
        
        r"the (?:correct )?answer is\s*:\s*(?:\()?([A-E])(?:\))?",
        r'(?:correct )?answer is\s*:\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is \\\( \\\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?) \\\)'
        r'the (?:correct )?answer is\s*:\s*[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
        r'the (?:correct )?answer is\s*:[\n\r]+\s*(?:\()?([A-Ea-e])(?:\))?',
        
    ]
    
    # Split the response into sentences using regex for better accuracy
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    
    # Get the last non-empty sentence
    last_sentence = None
    if sentences:
        last_sentence = sentences[-1].lower().strip()
        
        # If the last part doesn't end with a sentence-ending punctuation,
        # it might be a continuation or a separate sentence
        if not re.search(r'[.!?]$', last_sentence) and len(sentences) > 1:
            # Consider it anyway as it might be the conclusion
            pass
    
    if not last_sentence:
        return None
    
    # Try to match patterns only in the last sentence
    for pattern in patterns:
        match = re.search(pattern, last_sentence)
        if match:
            # Extract the matched number and remove commas if present
            answer = match.group(1).strip()
            if ',' in answer and answer.replace(',', '').isdigit():
                answer = answer.replace(',', '')
            return answer
    
    # If no patterns match, return None
    return None

def apply_self_consistency(
    results: List[Dict[str, Any]],
    prefix: str = "response_"
) -> List[Dict[str, Any]]:
    consistent_results = []
    
    for item in tqdm(results, desc="Applying self-consistency"):
        consistent_item = item.copy()
        
        # Extract all available responses
        responses = []
        for i in range(1, 11):
            response_key = f"{prefix}{i}"
            if response_key in item:
                responses.append(item[response_key])
            else:
                break
                
        if not responses:
            print(f"Warning: No responses found for question: {item.get('question', 'unknown')}")
            consistent_item["self_consistency_answer"] = None
            consistent_item["self_consistency_extraction"] = []
            consistent_item["self_consistency_vote"] = {}
            consistent_results.append(consistent_item)
            continue
            
        # Extract numerical answers from each response
        extracted_answers = []
        for response in responses:
            answer = extract_numerical_answer(response)
            extracted_answers.append(answer)
            
        # Count the frequency of each answer
        answer_counter = Counter(extracted_answers)
        
        # Add extraction info to results
        consistent_item["self_consistency_extraction"] = extracted_answers
        
        if not answer_counter:
            # No valid answers extracted
            consistent_item["self_consistency_answer"] = None
            consistent_item["self_consistency_vote"] = {}
        else:
            # Find the most common answer(s)
            most_common_answers = answer_counter.most_common()
            max_count = most_common_answers[0][1]
            
            # Get all answers with the maximum count
            top_answers = [ans for ans, count in most_common_answers if count == max_count]
            
            if len(top_answers) == 1:
                consistent_item["self_consistency_answer"] = top_answers[0]
            else:
                # If there's a tie, choose randomly among the most frequent answers
                consistent_item["self_consistency_answer"] = random.choice(top_answers)
        
            consistent_item["self_consistency_vote"] = {ans: count for ans, count in most_common_answers}
            
        consistent_results.append(consistent_item)
        
    return consistent_results

def run_self_consistency_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    benchmark_name: str
) -> Optional[Dict]:
    """
    Run self-consistency for a specific model's results
    
    Args:
        model_name: Model name to use
        results_dir: Directory containing the result JSON files
        output_dir: Directory to save the enhanced results
        benchmark_name: Name of the benchmark (e.g., "svamp", "gsm8k")
        
    Returns:
        Information about the saved self-consistency results or None if failed
    """
    try:
        # Find results file for this model
        results_filename = f"{benchmark_name}_{model_name}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        if not os.path.exists(results_path):
            print(f"Results file not found for model {model_name}: {results_path}")
            return None
            
        # Load results
        results_data = load_benchmark_results(results_path)
            
        if not results_data:
            print(f"No results found in {results_path}")
            return None
            
        print(f"Loaded {len(results_data)} results for model {model_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to save self-consistency results
        consistent_filename = f"{benchmark_name}_{model_name}_self_consistency.json"
        consistent_path = os.path.join(output_dir, consistent_filename)
        
        # Apply self-consistency
        consistent_results = apply_self_consistency(results_data)
        
        # Save final results
        with open(consistent_path, 'w', encoding='utf-8') as f:
            json.dump(consistent_results, f, ensure_ascii=False, indent=2)
            
        print(f"Self-consistency results saved to {consistent_path}")
        
        return {
            "path": consistent_path,
        }
        
    except Exception as e:
        print(f"Error running self-consistency for model {model_name}: {e}")
        return None

def run_all_models_self_consistency(
    results_dir: str,
    output_dir: str,
    benchmark_name: str,
    parallel: bool = False
) -> Dict[str, Dict]:
    """
    Run self-consistency for all models with results
    
    Args:
        results_dir: Directory containing the result JSON files
        output_dir: Directory to save the enhanced results
        benchmark_name: Name of the benchmark (e.g., "svamp", "gsm8k")
        parallel: Whether to run models in parallel
        
    Returns:
        Dictionary mapping model names to output file paths
    """
    # Find all result files in the directory
    prefix = f"{benchmark_name}_"
    suffix = ".json"
    result_files = [f for f in os.listdir(results_dir) 
                   if f.startswith(prefix) 
                   and f.endswith(suffix)
                   and not f.endswith("_refined.json") 
                   and not f.endswith("_self_consistency.json")]
    
    model_names = [f.replace(prefix, "").replace(suffix, "") for f in result_files]
    
    results = {}
    
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit jobs for each model
            future_to_model = {}
            for model_name in model_names:
                future = executor.submit(
                    run_self_consistency_for_model,
                    model_name,
                    results_dir,
                    output_dir,
                    benchmark_name
                )
                future_to_model[future] = model_name
                
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_info = future.result()
                    results[model_name] = result_info
                    if result_info:
                        print(f"Self-consistency for model {model_name} completed")
                    else:
                        print(f"Self-consistency for model {model_name} failed")
                except Exception as e:
                    print(f"Error running self-consistency for model {model_name}: {e}")
                    results[model_name] = None
    else:
        # Run sequentially
        for model_name in model_names:
            try:
                result_info = run_self_consistency_for_model(
                    model_name=model_name,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    benchmark_name=benchmark_name
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"Error running self-consistency for model {model_name}: {e}")
                results[model_name] = None
                
    return results

def process_benchmarks(
    base_dir: str,
    benchmarks: List[str],
    parallel: bool = False
) -> None:
    """
    Process multiple benchmarks
    
    Args:
        base_dir: Base directory containing benchmark results
        benchmarks: List of benchmark names to process
        parallel: Whether to run models in parallel
    """
    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"Processing benchmark: {benchmark}")
        print(f"{'='*50}")
        
        results_dir = os.path.join(base_dir, benchmark, "Results", f"{benchmark}_results")
        output_dir = os.path.join(base_dir, benchmark, "Results", "self_consistency_result")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting self-consistency for models in {results_dir}")
        results = run_all_models_self_consistency(
            results_dir=results_dir,
            output_dir=output_dir,
            benchmark_name=benchmark,
            parallel=parallel
        )
        
        # Report results
        print("\nSelf-consistency completed. Results summary:")
        
        for model_name, result_info in results.items():
            if result_info:
                path = result_info["path"]
                print(f"success {model_name}:")
                print(f"   - Output path: {path}")
            else:
                print(f" {model_name}: Failed")
                
        print(f"\nResults saved to {output_dir}")