import os
import json
import sys
import time
import re
from typing import List, Dict, Any
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

sys.path.append('/data3/jykim/Projects/CCQA_official')
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

def load_existing_usc_results(output_path: str) -> Dict[str, Any]:
    """
    Load existing USC results if available
    
    Args:
        output_path: Path to the USC output file
        
    Returns:
        Dictionary containing existing USC results or empty structure
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading existing USC results from {output_path}: {e}")
            
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

def extract_response_index(usc_evaluation: str) -> int:
    """
    Extract the chosen response index from USC evaluation
    
    Args:
        usc_evaluation: The evaluation response from USC
        
    Returns:
        Index of the chosen response or -1 if not found
    """
    # Look for "The most consistent response is Response X" format
    choice_match = re.search(r'most consistent response is [Rr]esponse (\d+)', usc_evaluation)
    if choice_match:
        return int(choice_match.group(1))
    
    return -1  # Return -1 if no index is found

def apply_universal_self_consistency(
    runner: LLMRunner,
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Apply Universal Self-Consistency to generate a consensus answer
    
    Args:
        runner: LLMRunner instance for generating responses
        results: List of GSM8K result items
        output_path: Path to save USC results
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        List of result items with USC consensus answers
    """
    # Load existing USC results
    existing_data = load_existing_usc_results(output_path)
    usc_results = existing_data.get("results", [])
    
    # Calculate how many questions have already been processed
    processed_count = len(usc_results)
    total_questions = len(results)
    
    print(f"Found {processed_count} already processed questions")
    
    # Skip already processed questions
    results_to_process = results[processed_count:]
    
    if not results_to_process:
        print(f"All questions already processed. Nothing to do.")
        return usc_results
    
    print(f"Will process {len(results_to_process)} remaining questions")
    
    for i, item in enumerate(tqdm(results_to_process, desc="Applying Universal Self-Consistency")):
        global_idx = i + processed_count
        usc_item = item.copy()
        
        # Get all available responses
        responses = []
        for j in range(1, 11):  # Assuming up to 5 responses are available
            response_key = f"response_{j}"
            if response_key in item and item[response_key]:
                responses.append(item[response_key])
        
        if not responses:
            print(f"Warning: No responses found for question: {item.get('question', 'unknown')}")
            usc_results.append(usc_item)
            
            # Save current progress
            current_data = {
                "completed_questions": global_idx + 1,
                "total_questions": total_questions,
                "results": usc_results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
                
            continue
        
        # Build USC prompt with all complete responses
        question_text = f"{item.get('body', '')}{item.get('question', '')}"
        consensus_prompt = f"""I have generated the following responses to the question: {question_text}

"""
        # Add each complete response to the prompt with 1-based indexing
        for idx, resp in enumerate(responses):
            consensus_prompt += f"Response {idx+1}: {resp}\n\n"
        
        consensus_prompt += """... Evaluate these responses. Select the most consistent response based on majority consensus. Start your answer with "The most consistent response is Response X" (without quotes)."""
        
        try:
            # Generate evaluation response using USC
            consensus_responses = runner.generate_responses(
                prompt=consensus_prompt,
                num_responses=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                parallel=False
            )
            
            # Store USC's full evaluation response
            usc_evaluation = consensus_responses[0] if consensus_responses else ""
            usc_item["usc_response"] = usc_evaluation
            
            # Extract the chosen response index (1-based)
            chosen_idx = extract_response_index(usc_evaluation)
            if chosen_idx != -1 and 1 <= chosen_idx <= len(responses):
                # Store the chosen original response as the USC answer (convert to 0-based for array access)
                chosen_response = responses[chosen_idx-1]
                usc_item["usc_answer"] = chosen_response
                # Store the chosen response index (as 1-based)
                usc_item["usc_chosen_idx"] = chosen_idx
            else:
                usc_item["usc_answer"] = "Unable to determine chosen response"
                usc_item["usc_chosen_idx"] = -1
                if chosen_idx != -1:
                    usc_item["usc_error"] = f"Invalid response index: {chosen_idx}"
            
        except Exception as e:
            print(f"Error applying USC evaluation: {e}")
            usc_item["usc_response"] = f"Error: {str(e)}"
            usc_item["usc_error"] = str(e)
            usc_item["usc_chosen_idx"] = -1
        
        # Add evaluation of correctness compared to original answer (this part would need to be implemented elsewhere)
        # Here we just pass the chosen response along without attempting to extract a numerical answer
        
        usc_results.append(usc_item)
        
        # Save progress after each question
        current_data = {
            "completed_questions": global_idx + 1,
            "total_questions": total_questions,
            "results": usc_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        # Print progress periodically
        if (global_idx + 1) % 10 == 0 or global_idx == 0 or global_idx == total_questions - 1:
            print(f"Progress: {global_idx + 1}/{total_questions} questions completed ({((global_idx + 1) / total_questions * 100):.1f}%)")
    
    return usc_results

def run_usc_for_model(
    model_name: str,
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> Dict:
    """
    Run Universal Self-Consistency for a specific model
    
    Args:
        model_name: Model name to use
        results_dir: Directory containing result JSON files
        output_dir: Directory to save USC results to
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary containing USC result information (path and processing time) or None if failed
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
        
        # Path to save USC results
        usc_filename = f"gsm8k_{model_name}_usc.json"
        usc_path = os.path.join(output_dir, usc_filename)
        
        # Initialize model
        runner = LLMRunner(model_name)
        
        # Apply USC (saving progress after each question)
        usc_results = apply_universal_self_consistency(
            runner=runner,
            results=results_data,
            output_path=usc_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Calculate metrics for reporting
        total_count = len(usc_results)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time information
        time_info = {
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "processing_time_hours": processing_time / 3600,
            "questions_count": len(results_data),
            "avg_time_per_question": processing_time / len(results_data) if len(results_data) > 0 else 0,
            "total_count": total_count
        }
        
        # Save final results with time information
        final_data = {
            "time_info": time_info,
            "results": usc_results
        }
        
        with open(usc_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"USC results saved to {usc_path}")
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        return {
            "path": usc_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"Error running USC for model {model_name}: {e}")
        return None

def run_all_models_usc(
    results_dir: str,
    output_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    parallel: bool = True
) -> Dict[str, Dict]:
    """
    Run Universal Self-Consistency for all models with results
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save USC results to
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
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
                    run_usc_for_model,
                    model_name,
                    results_dir,
                    output_dir,
                    max_new_tokens,
                    temperature,
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
                        print(f"USC completed for model {model_name} (processing time: {time_info['processing_time_minutes']:.2f} minutes)")
                    else:
                        print(f"USC failed for model {model_name}")
                except Exception as e:
                    print(f"Error running USC for model {model_name}: {e}")
                    results[model_name] = None
    else:
        # Run sequentially
        for model_name in model_names:
            try:
                result_info = run_usc_for_model(
                    model_name=model_name,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"Error running USC for model {model_name}: {e}")
                results[model_name] = None
    
    return results
# Configuration parameters
RESULTS_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/gsm8k_numgeneration_20"
OUTPUT_DIR = "/data3/jykim/Projects/CCQA_official/Benchmarks/GSM8K/Results/usc_result_response10"
MAX_NEW_TOKENS = 256  # Reduced token limit for USC evaluation
TEMPERATURE = 0.6     # USC temperature
RUN_PARALLEL = True   # Set to True to run models in parallel

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run USC for all models
    print(f"Starting Universal Self-Consistency for models in {RESULTS_DIR}")
    results = run_all_models_usc(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        parallel=RUN_PARALLEL
    )
    
    # Report results
    print("\nUniversal Self-Consistency completed. Results summary:")
    
    # Save all model processing time information to a single JSON file
    summary = {}
    
    for model_name, result_info in results.items():
        if result_info:
            time_info = result_info["time_info"]
            path = result_info["path"]
            summary[model_name] = time_info
            print(f"SUCCESS: {model_name}:")
            print(f"   - Processing time: {time_info['processing_time_seconds']:.2f} seconds ({time_info['processing_time_minutes']:.2f} minutes)")
            print(f"   - Average time per question: {time_info['avg_time_per_question']:.2f} seconds")
            print(f"   - Output path: {path}")
        else:
            print(f"FAILED: {model_name}")
    
    # Save summary information
    summary_path = os.path.join(OUTPUT_DIR, "usc_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll processing completed. Results saved to {OUTPUT_DIR}")
    print(f"Summary information saved to {summary_path}")