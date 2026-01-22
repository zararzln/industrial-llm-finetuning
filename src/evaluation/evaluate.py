"""
Evaluation script for fine-tuned models.

Implements multiple evaluation metrics:
- Perplexity
- ROUGE-L
- BLEU
- BERTScore
- Custom safety and completeness scores

Usage:
    python src/evaluation/evaluate.py --model results/mistral-7b-industrial --test-data data/processed/test.json
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import evaluate

# Load evaluation metrics
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
bertscore = evaluate.load('bertscore')


class SafetyScorer:
    """
    Custom metric to evaluate if response includes critical safety steps.
    
    Based on analysis of industrial maintenance procedures, critical safety
    steps include: PPE, lockout/tagout, ventilation, etc.
    """
    
    SAFETY_KEYWORDS = [
        'safety', 'ppe', 'gloves', 'goggles', 'lockout', 'tagout',
        'loto', 'ventilation', 'disconnect', 'power off', 'de-energize',
        'warning', 'caution', 'danger', 'protective equipment',
        'harness', 'respirator', 'confined space'
    ]
    
    def score(self, text: str) -> float:
        """
        Score text based on safety keyword mentions.
        Returns score between 0 and 1.
        """
        text_lower = text.lower()
        mentions = sum(1 for keyword in self.SAFETY_KEYWORDS if keyword in text_lower)
        
        # Normalize: cap at 5 mentions for score of 1.0
        return min(mentions / 5.0, 1.0)


class CompletenessScorer:
    """
    Custom metric to evaluate procedural completeness.
    
    Checks if response includes typical procedure elements:
    - Preparation/prerequisites
    - Tools/materials needed
    - Step-by-step instructions
    - Verification/testing
    - Cleanup/restoration
    """
    
    COMPLETENESS_MARKERS = {
        'preparation': ['before', 'prepare', 'prerequisite', 'ensure', 'verify'],
        'tools': ['tool', 'equipment', 'wrench', 'socket', 'multimeter', 'needed', 'required'],
        'steps': ['step', 'first', 'next', 'then', 'finally', 'procedure'],
        'verification': ['test', 'check', 'verify', 'confirm', 'measure'],
        'cleanup': ['clean', 'restore', 'reassemble', 'after', 'complete']
    }
    
    def score(self, text: str) -> float:
        """
        Score text based on presence of procedural elements.
        Returns score between 0 and 1.
        """
        text_lower = text.lower()
        category_scores = []
        
        for category, keywords in self.COMPLETENESS_MARKERS.items():
            has_category = any(keyword in text_lower for keyword in keywords)
            category_scores.append(1.0 if has_category else 0.0)
        
        return np.mean(category_scores)


def load_model(model_path: str, device: str = "cuda"):
    """Load fine-tuned model and tokenizer."""
    
    # Check if LoRA adapters exist
    lora_path = Path(model_path) / "lora_adapters"
    
    if lora_path.exists():
        # Load base model and apply LoRA
        config_path = lora_path / "adapter_config.json"
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"Loading LoRA adapters from: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        # Load full model
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Generate response for given instruction."""
    
    prompt = f"""Below is an instruction for a maintenance procedure. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def compute_perplexity(model, tokenizer, dataset) -> float:
    """Compute perplexity on test set."""
    
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    
    for example in tqdm(dataset, desc="Computing perplexity"):
        text = example['text']
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        total_loss += loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    
    return perplexity.item()


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    sample_size: int = None
) -> Dict[str, float]:
    """
    Run full evaluation suite on model.
    
    Returns dictionary of metric scores.
    """
    
    if sample_size:
        test_data = test_data[:sample_size]
    
    # Initialize custom scorers
    safety_scorer = SafetyScorer()
    completeness_scorer = CompletenessScorer()
    
    # Storage for metrics
    predictions = []
    references = []
    safety_scores = []
    completeness_scores = []
    
    print(f"Evaluating on {len(test_data)} examples...")
    
    for example in tqdm(test_data):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        reference = example.get('output', '')
        
        # Generate prediction
        prediction = generate_response(model, tokenizer, instruction, input_text)
        
        predictions.append(prediction)
        references.append(reference)
        
        # Compute custom scores
        safety_scores.append(safety_scorer.score(prediction))
        completeness_scores.append(completeness_scorer.score(prediction))
    
    # Compute ROUGE
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=['rougeL']
    )
    
    # Compute BLEU
    # Need to tokenize for BLEU
    predictions_tok = [pred.split() for pred in predictions]
    references_tok = [[ref.split()] for ref in references]
    
    bleu_results = bleu.compute(
        predictions=predictions_tok,
        references=references_tok
    )
    
    # Compute BERTScore
    bertscore_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli"
    )
    
    # Aggregate results
    results = {
        'rouge_l': rouge_results['rougeL'],
        'bleu': bleu_results['bleu'],
        'bertscore_f1': np.mean(bertscore_results['f1']),
        'safety_score': np.mean(safety_scores),
        'completeness_score': np.mean(completeness_scores),
        'num_examples': len(test_data)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--sample-size", type=int, help="Evaluate on subset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--compute-perplexity", action="store_true", help="Compute perplexity")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model)
    
    # Load test data
    print("Loading test data...")
    with open(args.test_data, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Run evaluation
    results = evaluate_model(model, tokenizer, test_data, args.sample_size)
    
    # Optionally compute perplexity
    if args.compute_perplexity:
        print("\nComputing perplexity...")
        dataset = load_dataset('json', data_files={'test': args.test_data})['test']
        
        # Format dataset
        def format_prompt(example):
            text = f"""Below is an instruction for a maintenance procedure. Write a response that appropriately completes the request.

### Instruction:
{example.get('instruction', '')}

### Input:
{example.get('input', '')}

### Response:
{example.get('output', '')}"""
            return {"text": text}
        
        dataset = dataset.map(format_prompt)
        
        perplexity = compute_perplexity(model, tokenizer, dataset)
        results['perplexity'] = perplexity
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"{metric:25s}: {score:.4f}")
        else:
            print(f"{metric:25s}: {score}")
    print("="*50)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
