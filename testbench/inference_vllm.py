#!/usr/bin/env python3
"""
InteriorGS Test Set Inference Script using vLLM

Supports:
1. Normal inference: with image input
2. Blind eval: without image input (text-only)

Usage:
    # Normal inference with official model
    python inference_vllm.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --test_dir /path/to/test \
        --output_dir ./results

    # Blind eval (no image)
    python inference_vllm.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --test_dir /path/to/test \
        --output_dir ./results \
        --blind

    # With fine-tuned model
    python inference_vllm.py \
        --model /path/to/finetuned/model \
        --test_dir /path/to/test \
        --output_dir ./results
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Please install with: pip install vllm")
    exit(1)


# System prompt for spatial reasoning tasks
SYSTEM_PROMPT = """You are an AI assistant specializing in spatial analysis from images.

[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided image.

[Output Instruction]
Provide your answer based on the question. End your response with the answer in the specified format."""


BLIND_SYSTEM_PROMPT = """You are an AI assistant specializing in spatial reasoning.

[Task]
Your task is to analyze spatial questions about objects. Note: No image is provided, so you need to reason based on the question alone.

[Output Instruction]
Provide your best answer based on the question. End your response with the answer in the specified format."""


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load an image and handle potential errors."""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            return img
        else:
            print(f"Image not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_test_data(test_dir: str) -> List[Dict[str, Any]]:
    """Load test data from questions.jsonl."""
    questions_file = os.path.join(test_dir, "questions.jsonl")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    data = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    
    return data


def prepare_requests(
    questions: List[Dict[str, Any]],
    test_dir: str,
    tokenizer,
    blind_mode: bool = False
) -> tuple:
    """
    Prepare requests for vLLM batch processing.
    
    Args:
        questions: List of question dictionaries
        test_dir: Base directory for images
        tokenizer: Model tokenizer for applying chat template
        blind_mode: If True, don't include images (blind eval)
    
    Returns:
        (requests_to_process, original_data_list)
    """
    requests_to_process = []
    original_data_list = []
    skipped = 0
    
    system_prompt = BLIND_SYSTEM_PROMPT if blind_mode else SYSTEM_PROMPT
    
    for question_item in tqdm(questions, desc="Preparing requests"):
        # Get image path
        image_rel_path = question_item.get("image", "")
        if image_rel_path:
            image_path = os.path.join(test_dir, image_rel_path)
        else:
            image_path = None
        
        # Load image if not blind mode
        images = []
        if not blind_mode and image_path:
            img = load_image(image_path)
            if img:
                images.append(img)
        
        # Skip if normal mode but no valid image
        if not blind_mode and not images:
            skipped += 1
            continue
        
        # Get question text
        question_text = question_item.get("question", "")
        
        # Build content list for chat template
        if blind_mode:
            # Text only
            content_list = [{"type": "text", "text": f"\n[Question]\n{question_text}"}]
        else:
            # Image + text
            content_list = [{"type": "image"}] * len(images)
            content_list.append({"type": "text", "text": f"\n[Question]\n{question_text}"})
        
        # Apply chat template
        try:
            final_prompt = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback for models without system prompt support
            final_prompt = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": content_list}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Build request
        if blind_mode:
            request_dict = {"prompt": final_prompt}
        else:
            request_dict = {
                "prompt": final_prompt,
                "multi_modal_data": {"image": images}
            }
        
        requests_to_process.append(request_dict)
        original_data_list.append(question_item)
    
    if skipped > 0:
        print(f"Skipped {skipped} questions due to missing images")
    
    return requests_to_process, original_data_list


def run_inference(
    llm: LLM,
    requests: List[Dict],
    original_data: List[Dict],
    output_file: str,
    max_tokens: int = 1024
):
    """Run inference and save results."""
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "</s>"]
    )
    
    print(f"\nRunning inference on {len(requests)} samples...")
    outputs = llm.generate(requests, sampling_params)
    print("Inference completed.")
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, output in enumerate(tqdm(outputs, desc="Saving results")):
            original_item = original_data[i]
            model_answer = output.outputs[0].text.strip()
            
            result = {
                "question_id": original_item.get("question_id", f"q_{i}"),
                "question_type": original_item.get("question_type", "unknown"),
                "question": original_item.get("question", ""),
                "ground_truth": original_item.get("answer", ""),
                "model_prediction": model_answer,
                "image": original_item.get("image", ""),
            }
            
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return results


def evaluate_results(results: List[Dict]) -> Dict[str, Any]:
    """Evaluation of results with support for numeric answers."""
    from collections import defaultdict
    import re
    import ast
    
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "close": 0})
    
    def extract_answer(text: str):
        """Extract answer value from prediction text."""
        text = str(text).strip()
        
        # Try to find {'answer': ...} or {"answer": ...} pattern
        patterns = [
            r"\{'answer':\s*([^}]+)\}",
            r'\{"answer":\s*([^}]+)\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer_str = match.group(1).strip()
                try:
                    return ast.literal_eval(answer_str)
                except:
                    return answer_str
        
        return text
    
    def compare_answers(gt, pred, tolerance=0.3):
        """Compare answers with tolerance for numeric values."""
        # Extract answer values
        gt_val = extract_answer(gt)
        pred_val = extract_answer(pred)
        
        # Handle list comparisons (e.g., [0.8, 0.0, 1.1])
        if isinstance(gt_val, (list, tuple)) and isinstance(pred_val, (list, tuple)):
            if len(gt_val) != len(pred_val):
                return False, False
            
            all_close = True
            for g, p in zip(gt_val, pred_val):
                try:
                    g_num = float(g)
                    p_num = float(p)
                    if abs(g_num - p_num) > tolerance:
                        all_close = False
                except (ValueError, TypeError):
                    if str(g).strip().lower() != str(p).strip().lower():
                        all_close = False
            
            return all_close, all_close
        
        # Handle numeric comparisons
        try:
            gt_num = float(gt_val)
            pred_num = float(pred_val)
            is_close = abs(gt_num - pred_num) <= tolerance
            is_exact = abs(gt_num - pred_num) <= 0.1
            return is_exact, is_close
        except (ValueError, TypeError):
            pass
        
        # String comparison
        gt_str = str(gt_val).strip().lower()
        pred_str = str(pred_val).strip().lower()
        is_match = gt_str == pred_str or gt_str in pred_str or pred_str in gt_str
        return is_match, is_match
    
    for r in results:
        qtype = r["question_type"]
        stats[qtype]["total"] += 1
        stats["overall"]["total"] += 1
        
        gt = r["ground_truth"]
        pred = r["model_prediction"]
        
        is_exact, is_close = compare_answers(gt, pred)
        
        if is_exact:
            stats[qtype]["correct"] += 1
            stats["overall"]["correct"] += 1
        if is_close:
            stats[qtype]["close"] += 1
            stats["overall"]["close"] += 1
    
    # Calculate accuracy
    eval_results = {}
    for qtype, s in stats.items():
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        close_acc = s["close"] / s["total"] * 100 if s["total"] > 0 else 0
        eval_results[qtype] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": f"{acc:.2f}%",
            "close_correct": s["close"],
            "close_accuracy": f"{close_acc:.2f}%"
        }
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="InteriorGS Test Set Inference with vLLM"
    )
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g., Qwen/Qwen3-VL-4B-Instruct or /path/to/finetuned)")
    parser.add_argument("--test_dir", type=str, 
                        default="/scratch/by2593/project/sceneshift/data/full_generation/0267_840790/test",
                        help="Directory containing test data (questions.jsonl and images/)")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                        default="/scratch/by2593/project/sceneshift/sft_training/inference_results",
                        help="Output directory for results")
    
    # Inference mode
    parser.add_argument("--blind", action="store_true",
                        help="Blind evaluation mode (no image input)")
    
    # Model settings
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (default: all available)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU memory utilization (0-1)")
    
    # Limit samples
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Run evaluation after inference")
    
    # Experiment name for output files
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name to include in output filename (e.g., mst_vqa_eval_with_image)")
    
    args = parser.parse_args()
    
    # Determine model name for output file
    model_name = args.model.strip("/").split("/")[-1]
    mode_suffix = "blind" if args.blind else "with_image"
    
    # Build experiment identifier for filename
    if args.experiment_name:
        exp_identifier = args.experiment_name
    else:
        # Extract from test_dir path
        test_dir_name = os.path.basename(args.test_dir.rstrip("/"))
        exp_identifier = f"{test_dir_name}_{mode_suffix}"
    
    print("=" * 70)
    print("InteriorGS Test Set Inference")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Test directory: {args.test_dir}")
    print(f"Mode: {'Blind (no image)' if args.blind else 'Normal (with image)'}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Initialize vLLM
    # For Qwen3-VL-4B with 32 attention heads, tensor_parallel_size must divide 32
    available_gpus = torch.cuda.device_count()
    valid_tp_sizes = [1, 2, 4, 8]  # Sizes that divide 32
    
    if args.tensor_parallel_size:
        tensor_parallel_size = args.tensor_parallel_size
    else:
        # Find the largest valid TP size that doesn't exceed available GPUs
        tensor_parallel_size = max([tp for tp in valid_tp_sizes if tp <= available_gpus])
    
    print(f"\nInitializing vLLM with {tensor_parallel_size} GPU(s) (available: {available_gpus})...")
    
    # Use v0 engine for better stability with multimodal models
    import os
    os.environ["VLLM_USE_V1"] = "0"
    
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},  # Limit to 1 image per prompt
    )
    
    # Get tokenizer
    tokenizer = llm.get_tokenizer()
    
    # Load test data
    print("\nLoading test data...")
    questions = load_test_data(args.test_dir)
    print(f"Loaded {len(questions)} questions")
    
    # Limit samples if specified
    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"Limited to {len(questions)} samples")
    
    # Prepare requests
    print("\nPreparing requests...")
    requests, original_data = prepare_requests(
        questions, args.test_dir, tokenizer, blind_mode=args.blind
    )
    print(f"Prepared {len(requests)} requests")
    
    # Run inference
    output_file = os.path.join(
        args.output_dir, 
        f"{model_name}_{exp_identifier}_results.jsonl"
    )
    
    results = run_inference(
        llm, requests, original_data, output_file, args.max_tokens
    )
    
    print(f"\nResults saved to: {output_file}")
    
    # Evaluate
    if args.evaluate:
        print("\n" + "=" * 70)
        print("Evaluation Results (Exact | Close ±0.3)")
        print("=" * 70)
        
        eval_results = evaluate_results(results)
        
        # Print results
        for qtype, stats in sorted(eval_results.items()):
            if qtype != "overall":
                print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']}) | {stats['close_correct']}/{stats['total']} ({stats['close_accuracy']})")
        
        print("-" * 70)
        overall = eval_results.get("overall", {})
        print(f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} ({overall.get('accuracy', '0%')}) | {overall.get('close_correct', 0)}/{overall.get('total', 0)} ({overall.get('close_accuracy', '0%')})")
        
        # Save evaluation results
        eval_file = os.path.join(
            args.output_dir,
            f"{model_name}_{exp_identifier}_evaluation.json"
        )
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation saved to: {eval_file}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
