import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

sys.path.append('/home/elicer/ccqa')
from LLM_runner import LLMRunner

def load_gsm8k_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load GSM8K results JSON file
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        List of result items
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Handle different results formats
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        print(f"Error loading results {results_path}: {e}")
        return []

def load_existing_self_refinement(output_path: str) -> Dict[str, Any]:
    """
    Load existing self-refinement results if available
    
    Args:
        output_path: Path to the self-refinement output file
        
    Returns:
        Dictionary containing existing self-refinement results or empty structure
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading existing self-refinement results from {output_path}: {e}")
            
            # Create a backup of problematic file
            backup_path = f"{output_path}.bak.{int(time.time())}"
            try:
                os.rename(output_path, backup_path)
                print(f"Created backup of problematic file at {backup_path}")
            except Exception as be:
                print(f"Failed to create backup: {be}")
    
    # Return empty structure
    return {
        "completed_questions": 0,
        "total_questions": 0,
        "results": []
    }

def apply_self_refinement(
    runner: LLMRunner,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.9
) -> List[Dict[str, Any]]:
    
    # Load existing self-refinement results
    existing_data = load_existing_self_refinement(output_path)
    refined_results = existing_data.get("results", [])
    
    # Calculate how many questions have already been processed
    processed_count = len(refined_results)
    total_questions = len(results)
    
    print(f"Found {processed_count} already processed questions")
    
    # Skip already processed questions
    results_to_process = results[processed_count:]
    
    if not results_to_process:
        print(f"All questions already processed. Nothing to do.")
        return refined_results
    
    print(f"Will process {len(results_to_process)} remaining questions")
    
    for i, item in enumerate(tqdm(results_to_process, desc="Applying self-refinement")):
        global_idx = i + processed_count
        refined_item = item.copy()
        
        # Get the first response
        first_response = item.get("response_1", "")
        
        if not first_response:
            print(f"Warning: Could not find first response. Question: {item.get('question', 'unknown')}")
            refined_results.append(refined_item)
            
            # Save current progress
            current_data = {
                "completed_questions": global_idx + 1,
                "total_questions": total_questions,
                "results": refined_results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
                
            continue
        
        # Generate review prompt
        review_prompt = f"""Question: Can you solve the following math problem? {item.get('body', '')}{item.get('question', '')}

Your answer was:
{first_response}

Do you think the question and answer pair above is correct? The answer has a logical chain. If you think the answer is incorrect, please identify where the logic is wrong in the logical chain."""
        
        try:
            # Generate review response
            review_responses = runner.generate_responses(
                prompt=review_prompt,
                num_responses=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=False
            )
            
            review_response = review_responses[0] if review_responses else ""
            refined_item["first_refine"] = review_response
            
            # Generate improvement prompt
            improvement_prompt = f"""Question: Can you solve the following math problem? {item.get('body', '')}{item.get('question', '')}

Your answer was:
{first_response}

Review of your answer:
{review_response}

Based on the responses above, answer the original question. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."""
            
            # Generate improvement response
            improvement_responses = runner.generate_responses(
                prompt=improvement_prompt,
                num_responses=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=False
            )
            
            improvement_response = improvement_responses[0] if improvement_responses else ""
            refined_item["second_refine"] = improvement_response
            
        except Exception as e:
            print(f"Error applying self-refinement: {e}")
            refined_item["first_refine"] = f"Error: {str(e)}"
            refined_item["second_refine"] = f"Error: {str(e)}"
        
        refined_results.append(refined_item)
        
        # Save progress after each question
        current_data = {
            "completed_questions": global_idx + 1,
            "total_questions": total_questions,
            "results": refined_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        # Print progress periodically
        if (global_idx + 1) % 10 == 0 or global_idx == 0 or global_idx == total_questions - 1:
            print(f"Progress: {global_idx + 1}/{total_questions} questions completed ({((global_idx + 1) / total_questions * 100):.1f}%)")
    
    return refined_results


def run_self_refinement_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.9
) -> Optional[Dict]:
    """
    Run self-refinement for a specific model
    
    Args:
        model_name: Model name to use
        results_dir: Directory containing result JSON files
        output_dir: Directory to save refined results to
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary containing refined result information (path and processing time) or None if failed
    """
    try:
        start_time = time.time()
        
        # Find this model's results file
        results_filename = f"gsm8k_{model_name}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        if not os.path.exists(results_path):
            print(f"Could not find results file for model {model_name}: {results_path}")
            return None
        
        # Load results
        results_data = load_gsm8k_results(results_path)
            
        if not results_data:
            print(f"No results found in {results_path}")
            return None
        
        print(f"Loaded {len(results_data)} results for model {model_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to save refined results
        refined_filename = f"gsm8k_{model_name}_refined.json"
        refined_path = os.path.join(output_dir, refined_filename)
        
        # Initialize model
        runner = LLMRunner(model_name)
        
        # Apply self-refinement (saving progress after each question)
        refined_results = apply_self_refinement(
            runner=runner,
            results=results_data,
            output_path=refined_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time information
        time_info = {
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "processing_time_hours": processing_time / 3600,
            "questions_count": len(results_data),
            "avg_time_per_question": processing_time / len(results_data) if len(results_data) > 0 else 0
        }
        
        # Save final results with time information
        final_data = {
            "time_info": time_info,
            "results": refined_results
        }
        
        with open(refined_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"Refined results saved to {refined_path}")
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        return {
            "path": refined_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"Error running self-refinement for model {model_name}: {e}")
        return None

def run_all_models_refinement(
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    parallel: bool = False
) -> Dict[str, Dict]:
    """
    Run self-refinement for all models with results
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save refined results to
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        parallel: Whether to run models in parallel
        
    Returns:
        Dictionary mapping model names to output file paths and time information
    """
    # Find all result files in the directory
    result_files = [f for f in os.listdir(results_dir) if f.startswith("gsm8k_") and f.endswith(".json")]
    model_names = [f.replace("gsm8k_", "").replace(".json", "") for f in result_files]
    
    results = {}
    
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit jobs for each model
            future_to_model = {}
            for model_name in model_names:
                future = executor.submit(
                    run_self_refinement_for_model,
                    model_name,
                    results_dir,
                    output_dir,
                    max_new_tokens,
                    temperature,
                    top_p
                )
                future_to_model[future] = model_name
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_info = future.result()
                    results[model_name] = result_info
                    if result_info:
                        time_info = result_info["time_info"]
                        print(f"Self-refinement completed for model {model_name} (processing time: {time_info['processing_time_minutes']:.2f} minutes)")
                    else:
                        print(f"Self-refinement failed for model {model_name}")
                except Exception as e:
                    print(f"Error running self-refinement for model {model_name}: {e}")
                    results[model_name] = None
    else:
        # Run sequentially
        for model_name in model_names:
            try:
                result_info = run_self_refinement_for_model(
                    model_name=model_name,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"Error running self-refinement for model {model_name}: {e}")
                results[model_name] = None
    
    return results

# Configuration parameters
RESULTS_DIR = "/home/elicer/ccqa/Benchmarks/GSM8K/Results/gsm8k_result"
OUTPUT_DIR = "/home/elicer/ccqa/Benchmarks/GSM8K/Results/self_correction_result"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
RUN_PARALLEL = True  # Set to True to run models in parallel

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run self-refinement for all models
    print(f"Starting self-refinement for models in {RESULTS_DIR}")
    results = run_all_models_refinement(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        parallel=RUN_PARALLEL
    )
    
    # Report results
    print("\nSelf-refinement completed. Results summary:")
    
    # Save all model processing time information to a single JSON file
    time_summary = {}
    
    for model_name, result_info in results.items():
        if result_info:
            time_info = result_info["time_info"]
            path = result_info["path"]
            time_summary[model_name] = time_info
            print(f"SUCCESS: {model_name}:")
            print(f"   - Processing time: {time_info['processing_time_seconds']:.2f} seconds ({time_info['processing_time_minutes']:.2f} minutes)")
            print(f"   - Average time per question: {time_info['avg_time_per_question']:.2f} seconds")
            print(f"   - Output path: {path}")
        else:
            print(f"FAILED: {model_name}")
    
    # Save time information summary
    time_summary_path = os.path.join(OUTPUT_DIR, "refinement_time_summary.json")
    with open(time_summary_path, 'w', encoding='utf-8') as f:
        json.dump(time_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll processing completed. Results saved to {OUTPUT_DIR}")
    print(f"Time summary information saved to {time_summary_path}")