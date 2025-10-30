import os
import json
import torch
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class LLMRunner:
    """Class for loading LLM models and generating responses"""
    
    # Available models mapping
    AVAILABLE_MODELS = {
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
        # "qwen-7b": "Qwen/Qwen2.5-7B-Instruct-1M",
        "gemma-1b": "google/gemma-3-1b-it",
        # "smollm-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        # "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
        # "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "deepseek-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "llama-1b" : "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3b" : "meta-llama/Llama-3.2-3B-Instruct",
        # "xfinder" : "IAAR-Shanghai/xFinder-qwen1505",
    }
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize the LLM model

        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_path = self.AVAILABLE_MODELS[model_name]
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model '{self.model_path}'... Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        print(f"Model '{self.model_path}' loaded successfully")
    
    def _generate_single_response(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int = None) -> str:
        """
        Generate a single response (helper method for parallel generation)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def generate_responses(
        self, 
        prompt: str, 
        num_responses: int = 5,
        max_new_tokens: int = 2048,
        temperature: float = 0.5,
        top_p: float = 0.9,
        parallel: bool = True
    ) -> List[str]:
        """
        Generate multiple responses for a given prompt, optionally in parallel
        """
        if not parallel:
            # Sequential generation
            responses = []
            for i in range(num_responses):
                print(f"Generating response {i+1}/{num_responses}...")
                responses.append(self._generate_single_response(
                    prompt, max_new_tokens, temperature, top_p, seed=i
                ))
            return responses
        else:
            # Parallel generation using ThreadPoolExecutor
            # Note: This creates multiple threads but uses the same model instance
            # May not improve performance with GPU due to synchronization, but helps with CPU
            print(f"Generating {num_responses} responses in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_responses) as executor:
                future_to_idx = {
                    executor.submit(
                        self._generate_single_response, 
                        prompt, max_new_tokens, temperature, top_p, i
                    ): i for i in range(num_responses)
                }
                
                responses = [None] * num_responses
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response = future.result()
                        responses[idx] = response
                        # Removed the "Completed response" print statement
                    except Exception as e:
                        print(f"Error generating response {idx+1}: {e}")
                        responses[idx] = f"Error: {str(e)}"
            
            return responses