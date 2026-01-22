"""
Main training script for fine-tuning LLMs on industrial documentation.

Usage:
    python src/training/train.py --config configs/mistral_qlora.yaml
    python src/training/train.py --config configs/mistral_qlora.yaml --wandb-project my-project
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Initialize model with QLoRA configuration."""
    
    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, config['model']['bnb_4bit_compute_dtype']),
        bnb_4bit_quant_type=config['model']['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=config['model']['bnb_4bit_use_double_quant']
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || "
          f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    return model, tokenizer


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """Load and prepare dataset for training."""
    
    # Load datasets
    data_files = {
        'train': config['data']['train_file'],
        'validation': config['data']['val_file']
    }
    
    dataset = load_dataset('json', data_files=data_files)
    
    # Format prompts
    def format_prompt(example):
        template = config['data']['prompt_template']
        text = template.format(
            instruction=example.get(config['data']['instruction_key'], ""),
            input=example.get(config['data']['input_key'], ""),
            output=example.get(config['data']['output_key'], "")
        )
        return {"text": text}
    
    dataset = dataset.map(format_prompt)
    
    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['data']['max_seq_length'],
            padding=False,
            return_tensors=None
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset


def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Configure training arguments."""
    
    args = TrainingArguments(
        output_dir=config['output']['output_dir'],
        run_name=config['output']['run_name'],
        
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        
        max_grad_norm=config['training']['max_grad_norm'],
        optim=config['training']['optim'],
        
        logging_steps=config['training']['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        report_to="wandb" if config['wandb']['enabled'] else "none",
        
        seed=config['seed']
    )
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on industrial documentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--wandb-project", type=str, help="W&B project name (overrides config)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override W&B settings if provided
    if args.wandb_project:
        config['wandb']['enabled'] = True
        config['wandb']['project'] = args.wandb_project
    
    # Initialize W&B
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity'),
            name=config['output']['run_name'],
            config=config
        )
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(config, tokenizer)
    
    # Training arguments
    training_args = setup_training_args(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator
    )
    
    # Train
    print("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(config['output']['output_dir'])
    tokenizer.save_pretrained(config['output']['output_dir'])
    
    # Save LoRA adapters separately
    model.save_pretrained(os.path.join(config['output']['output_dir'], "lora_adapters"))
    
    print("Training complete!")
    
    if config['wandb']['enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()
