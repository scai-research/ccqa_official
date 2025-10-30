import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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

def apply_self_correction(
    runner, 
    results: List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> List[Dict[str, Any]]:

    corrected_results = []
    
    for i, item in enumerate(tqdm(results, desc="Applying self-correction")):
        corrected_item = item.copy()
        
        # Get the first response
        first_response = item.get("response_1", "")
        
        if not first_response:
            print(f"Warning: First response not found for question: {item.get('question', 'unknown')}")
            corrected_results.append(corrected_item)
            
            # Save results after each question
            current_results = {
                "completed_questions": i + 1,
                "total_questions": len(results),
                "results": corrected_results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
                
            continue
        
        # Create review prompt
        body = item.get('body', '')
        question = item.get('question', '')
        review_prompt = f"""Q: {body}{question}

Your answer was:
{first_response}

Review your previous answer and find problems with your answer."""
        
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
            corrected_item["first_refine"] = review_response
            
            # Create improvement prompt
            improvement_prompt = f"""Q: {body}{question}

Your answer was:
{first_response}

Review of your answer:
{review_response}

Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form /the answer is {{answer}}."""
            
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
            corrected_item["second_refine"] = improvement_response
            
        except Exception as e:
            print(f"Error applying self-correction: {e}")
            corrected_item["first_refine"] = f"Error: {str(e)}"
            corrected_item["second_refine"] = f"Error: {str(e)}"
        
        corrected_results.append(corrected_item)
        
        # Save results after each question
        current_results = {
            "completed_questions": i + 1,
            "total_questions": len(results),
            "results": corrected_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0 or i == len(results) - 1:
            print(f"Progress: {i + 1}/{len(results)} questions completed ({((i + 1) / len(results) * 100):.1f}%)")
    
    return corrected_results

def run_self_correction_for_model(
    model_name: str,
    runner_class,
    results_dir: str,
    output_dir: str,
    benchmark_name: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9
) -> Optional[Dict]:
    """
    Run self-correction for a specific model
    
    Args:
        model_name: Model name to use
        runner_class: The LLMRunner class to instantiate
        results_dir: Directory containing the result JSON files
        output_dir: Directory to save the enhanced results
        benchmark_name: Name of the benchmark (e.g., "svamp", "gsm8k")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Information about the saved self-correction results (path and processing time) or None if failed
    """
    try:
        start_time = time.time()
        
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
        
        # Path to save self-correction results
        corrected_filename = f"{benchmark_name}_{model_name}_refined.json"
        corrected_path = os.path.join(output_dir, corrected_filename)
        
        # Initialize model
        runner = runner_class(model_name)
        
        # Apply self-correction
        corrected_results = apply_self_correction(
            runner=runner,
            results=results_data,
            output_path=corrected_path,
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
        result_with_time = {
            "time_info": time_info,
            "results": corrected_results
        }
        
        with open(corrected_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_time, f, ensure_ascii=False, indent=2)
            
        print(f"Self-correction results saved to {corrected_path}")
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        return {
            "path": corrected_path,
            "time_info": time_info
        }
        
    except Exception as e:
        print(f"Error running self-correction for model {model_name}: {e}")
        return None

def run_all_models_self_correction(
    results_dir: str,
    output_dir: str,
    benchmark_name: str,
    runner_class,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9,
    parallel: bool = False
) -> Dict[str, Dict]:
    """
    Run self-correction for all models with results
    
    Args:
        results_dir: Directory containing the result JSON files
        output_dir: Directory to save the enhanced results
        benchmark_name: Name of the benchmark (e.g., "svamp", "gsm8k")
        runner_class: The LLMRunner class to instantiate
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        parallel: Whether to run models in parallel
        
    Returns:
        Dictionary mapping model names to output file paths and time information
    """
    import concurrent.futures
    
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
                    run_self_correction_for_model,
                    model_name,
                    runner_class,
                    results_dir,
                    output_dir,
                    benchmark_name,
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
                        print(f"Self-correction for model {model_name} completed (processing time: {time_info['processing_time_minutes']:.2f} minutes)")
                    else:
                        print(f"Self-correction for model {model_name} failed")
                except Exception as e:
                    print(f"Error running self-correction for model {model_name}: {e}")
                    results[model_name] = None
    else:
        # Run sequentially
        for model_name in model_names:
            try:
                result_info = run_self_correction_for_model(
                    model_name=model_name,
                    runner_class=runner_class,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    benchmark_name=benchmark_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                results[model_name] = result_info
            except Exception as e:
                print(f"Error running self-correction for model {model_name}: {e}")
                results[model_name] = None
                
    return results