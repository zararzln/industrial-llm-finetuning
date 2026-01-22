"""
Data preparation script for industrial maintenance documentation.

This script processes raw maintenance documentation into instruction-following format
suitable for fine-tuning causal language models.

Input format: Various (PDFs, text files, scraped HTML)
Output format: JSON Lines with instruction/input/output structure

Usage:
    python src/preprocessing/prepare_data.py --source data/raw --output data/processed
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import defaultdict


class DataPreprocessor:
    """Process and clean industrial maintenance documentation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\/]', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def extract_qa_pairs(self, text: str, source: str) -> List[Dict]:
        """
        Extract question-answer pairs from forum/Q&A style content.
        
        This is a simplified heuristic - in practice, used more sophisticated
        parsing based on HTML structure.
        """
        pairs = []
        
        # Simple heuristic: split on common question markers
        questions = re.split(r'\n(?:Q:|Question:|How )', text)
        
        for q in questions[1:]:  # Skip first split (before any question)
            # Try to split on answer marker
            parts = re.split(r'(?:A:|Answer:|Solution:)', q, maxsplit=1)
            
            if len(parts) == 2:
                question = "How " + self.clean_text(parts[0])
                answer = self.clean_text(parts[1])
                
                # Filter out very short or very long entries
                if 20 < len(question) < 500 and 50 < len(answer) < 2000:
                    pairs.append({
                        'instruction': question,
                        'input': '',
                        'output': answer,
                        'source': source,
                        'type': 'qa'
                    })
        
        return pairs
    
    def extract_procedures(self, text: str, title: str, source: str) -> List[Dict]:
        """
        Extract maintenance procedures from manuals.
        
        Procedures typically have a structure like:
        - Title/description
        - Tools needed
        - Steps (numbered or bulleted)
        - Warnings/cautions
        """
        procedures = []
        
        # Look for numbered procedures
        procedure_pattern = r'(?:Procedure:|Steps:|To \w+:)(.*?)(?:\n\n|\Z)'
        matches = re.finditer(procedure_pattern, text, re.DOTALL)
        
        for match in matches:
            procedure_text = self.clean_text(match.group(1))
            
            if 100 < len(procedure_text) < 2000:
                # Create instruction based on title
                instruction = f"Provide the maintenance procedure for {title.lower()}"
                
                procedures.append({
                    'instruction': instruction,
                    'input': '',
                    'output': procedure_text,
                    'source': source,
                    'type': 'procedure'
                })
        
        return procedures
    
    def create_troubleshooting_examples(self, text: str, source: str) -> List[Dict]:
        """
        Create troubleshooting examples from diagnostic guides.
        
        Pattern: Symptom → Possible Causes → Solution
        """
        examples = []
        
        # Pattern to match symptom-solution pairs
        symptom_pattern = r'(?:Symptom|Problem|Issue):\s*(.*?)\s*(?:Cause|Solution|Fix):\s*(.*?)(?:\n\n|\Z)'
        matches = re.finditer(symptom_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            symptom = self.clean_text(match.group(1))
            solution = self.clean_text(match.group(2))
            
            if 20 < len(symptom) < 300 and 50 < len(solution) < 1500:
                instruction = f"Diagnose and provide a solution for: {symptom}"
                
                examples.append({
                    'instruction': instruction,
                    'input': '',
                    'output': solution,
                    'source': source,
                    'type': 'troubleshooting'
                })
        
        return examples
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file and extract examples."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        examples = []
        source = file_path.stem
        
        # Determine type based on filename or content
        if 'qa' in file_path.name.lower() or 'forum' in file_path.name.lower():
            examples.extend(self.extract_qa_pairs(content, source))
        
        if 'manual' in file_path.name.lower() or 'procedure' in file_path.name.lower():
            title = file_path.stem.replace('_', ' ')
            examples.extend(self.extract_procedures(content, title, source))
        
        if 'troubleshooting' in file_path.name.lower() or 'diagnostic' in file_path.name.lower():
            examples.extend(self.create_troubleshooting_examples(content, source))
        
        return examples
    
    def split_dataset(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets with stratification by type."""
        
        # Group by type
        by_type = defaultdict(list)
        for ex in examples:
            by_type[ex['type']].append(ex)
        
        train, val, test = [], [], []
        
        for type_name, type_examples in by_type.items():
            random.shuffle(type_examples)
            n = len(type_examples)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train.extend(type_examples[:n_train])
            val.extend(type_examples[n_train:n_train + n_val])
            test.extend(type_examples[n_train + n_val:])
        
        # Shuffle final sets
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
    
    def save_dataset(self, examples: List[Dict], output_file: Path):
        """Save examples in JSON Lines format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Source data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(seed=args.seed)
    
    # Process all files
    source_dir = Path(args.source)
    all_examples = []
    
    print(f"Processing files from: {source_dir}")
    
    for file_path in source_dir.rglob('*.txt'):
        print(f"Processing: {file_path.name}")
        examples = preprocessor.process_file(file_path)
        all_examples.extend(examples)
        print(f"  Extracted {len(examples)} examples")
    
    print(f"\nTotal examples extracted: {len(all_examples)}")
    
    # Count by type
    type_counts = defaultdict(int)
    for ex in all_examples:
        type_counts[ex['type']] += 1
    
    print("\nExamples by type:")
    for type_name, count in type_counts.items():
        print(f"  {type_name}: {count}")
    
    # Split dataset
    print("\nSplitting dataset...")
    train, val, test = preprocessor.split_dataset(
        all_examples,
        args.train_ratio,
        args.val_ratio
    )
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save datasets
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_dir}")
    preprocessor.save_dataset(train, output_dir / 'train.json')
    preprocessor.save_dataset(val, output_dir / 'val.json')
    preprocessor.save_dataset(test, output_dir / 'test.json')
    
    # Save statistics
    stats = {
        'total_examples': len(all_examples),
        'train_examples': len(train),
        'val_examples': len(val),
        'test_examples': len(test),
        'type_distribution': dict(type_counts),
        'seed': args.seed
    }
    
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset preparation complete!")
    print(f"Statistics saved to: {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
