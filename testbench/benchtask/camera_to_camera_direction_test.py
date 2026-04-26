#!/usr/bin/env python3
"""
Camera-to-Camera Direction Test Script using vLLM

This script reads subfolders in the specified directory, each containing:
- meta.json: Metadata with the question and choices
- preview.png: Preview image (not used in inference)
- begin.png: Start image for the question
- end.png: End image for the question

Supports:
1. Normal inference: with image input (begin.png and end.png)
2. Blind eval: without image input (text-only)

Usage:
    # Normal inference
    python Data_generation/testbench/benchtask/camera_to_camera_direction_test.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --input_dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/BenchTask/camera_to_camera_direction \
        --output_dir ./results

    # Blind eval (no image)
    python camera_to_camera_direction_test.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --input_dir /path/to/camera_to_camera_direction \
        --output_dir ./results \
        --blind
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Please install with: pip install vllm")
    exit(1)

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

def load_meta(meta_path: str) -> Dict[str, Any]:
    """Load metadata from meta.json."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_requests(
    input_dir: str,
    tokenizer,
    blind_mode: bool = False
) -> tuple:
    """
    Prepare requests for vLLM batch processing.

    Args:
        input_dir: Directory containing subfolders with meta.json, begin.png, and end.png
        tokenizer: Model tokenizer for applying chat template
        blind_mode: If True, don't include images (blind eval)

    Returns:
        (requests_to_process, original_data_list)
    """
    requests_to_process = []
    original_data_list = []

    system_prompt = BLIND_SYSTEM_PROMPT if blind_mode else SYSTEM_PROMPT

    for subfolder in tqdm(os.listdir(input_dir), desc="Preparing requests"):
        subfolder_path = os.path.join(input_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        meta_path = os.path.join(subfolder_path, "meta.json")
        begin_image_path = os.path.join(subfolder_path, "begin.png")
        end_image_path = os.path.join(subfolder_path, "end.png")

        if not os.path.exists(meta_path):
            print(f"Warning: meta.json not found in {subfolder_path}. Skipping.")
            continue

        meta = load_meta(meta_path)
        question = meta.get("question", "")
        choices_map = meta.get("choices_map", [])

        if not blind_mode:
            if not os.path.exists(begin_image_path) or not os.path.exists(end_image_path):
                print(f"Warning: Images not found in {subfolder_path}. Skipping.")
                continue

            request = {
                "prompt": f"{system_prompt}\n\nQuestion: {question}\nChoices: {', '.join(choices_map)}",
                "images": [
                    Image.open(begin_image_path),
                    Image.open(end_image_path)
                ]
            }
        else:
            request = {
                "prompt": f"{system_prompt}\n\nQuestion: {question}\nChoices: {', '.join(choices_map)}",
                "images": []
            }

        requests_to_process.append(request)
        original_data_list.append({"question": question, "choices_map": choices_map, "subfolder": subfolder})

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
        for request, output, original in zip(requests, outputs, original_data):
            result = {
                "question": original["question"],
                "choices_map": original["choices_map"],
                "subfolder": original["subfolder"],
                "output": output.outputs[0].text.strip()
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Camera-to-Camera Direction Test with vLLM"
    )

    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g., Qwen/Qwen3-VL-4B-Instruct or /path/to/finetuned)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing subfolders with meta.json, begin.png, and end.png")
    parser.add_argument("--output_dir", type=str, required=True,
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

    args = parser.parse_args()

    print("=" * 70)
    print("Camera-to-Camera Direction Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input directory: {args.input_dir}")
    print(f"Mode: {'Blind (no image)' if args.blind else 'Normal (with image)'}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Initialize vLLM
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = args.tensor_parallel_size or min(available_gpus, 4)

    print(f"\nInitializing vLLM with {tensor_parallel_size} GPU(s) (available: {available_gpus})...")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Reduce parallelism to save memory
        max_model_len=args.max_model_len,
        dtype="float16",
        gpu_memory_utilization=0.5,  # Lower memory utilization
        enable_prefix_caching=False,  # Disable unsupported feature
        enable_chunked_prefill=False  # Disable unsupported feature
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()

    # Prepare requests
    print("\nPreparing requests...")
    requests, original_data = prepare_requests(
        args.input_dir, tokenizer, blind_mode=args.blind
    )
    print(f"Prepared {len(requests)} requests")

    # Run inference
    output_file = os.path.join(
        args.output_dir, "camera_to_camera_direction_results.jsonl"
    )

    results = run_inference(
        llm, requests, original_data, output_file, args.max_tokens
    )

    print(f"\nResults saved to: {output_file}")

    print("\n✅ Done!")

if __name__ == "__main__":
    main()