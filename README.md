# Fine-tuning LLMs for Industrial Maintenance Documentation

A research project exploring domain-specific fine-tuning of open-source language models on industrial maintenance and technical documentation. This work investigates whether smaller, fine-tuned models can match or exceed GPT-3.5 performance on specialized technical tasks while being deployable in resource-constrained industrial environments.

## Motivation

Industrial settings often require AI systems that can:
- Operate in air-gapped environments (no API access)
- Process domain-specific technical language
- Provide consistent, deterministic outputs
- Run on edge devices with limited compute

This project tests the hypothesis that a well-fine-tuned 7B parameter model can outperform general-purpose LLMs on industrial maintenance tasks while using <10% of the computational resources.

## Research Questions

1. **Domain Transfer**: How much does performance improve when fine-tuning on industrial documentation vs. general instruction-following?
2. **Model Selection**: Which base model architecture (Llama 2, Mistral, Phi-2) transfers best to technical domains?
3. **Data Efficiency**: What's the minimum training data needed to achieve acceptable performance?
4. **Evaluation Metrics**: How do we measure "quality" for technical troubleshooting responses beyond perplexity?

## Methodology

### Base Models Evaluated
- **Llama 2 7B** - Meta's foundational model
- **Mistral 7B v0.1** - Strong performance/size ratio
- **Phi-2 2.7B** - Efficient smaller model for edge deployment

### Dataset Construction

Created a curated dataset from public sources:
- **Equipment Manuals**: 2,400+ maintenance procedures from manufacturer docs
- **Technical Forums**: 1,800 Q&A pairs from industrial maintenance communities
- **Troubleshooting Guides**: 950 diagnostic procedures
- **Safety Documentation**: 600 safety protocols and warnings

**Total**: ~5,200 training examples, ~650 validation, ~650 test

**Data Sources**:
- Public maintenance manuals (John Deere, Caterpillar documentation)
- Reddit r/industrialmaintenance
- Engineering Stack Exchange
- OSHA technical documents

### Training Approach

**QLoRA (Quantized Low-Rank Adaptation)**:
- 4-bit quantization for memory efficiency
- LoRA rank = 64 (ablation study: 16, 32, 64, 128)
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4 with cosine decay
- Batch size: 4 (gradient accumulation x8 = effective 32)
- 3 epochs with early stopping

**Why QLoRA?**
- Full fine-tuning requires 4x A100 GPUs (~$10K+ compute cost)
- QLoRA achieves 99% of performance on single consumer GPU
- Enables rapid experimentation and iteration

### Evaluation Framework

#### Automatic Metrics
1. **Perplexity** - Standard language modeling metric
2. **ROUGE-L** - Overlap with reference procedures
3. **BLEU** - N-gram precision for technical terminology
4. **BERTScore** - Semantic similarity

#### Task-Specific Metrics
1. **Safety Score** - Mentions critical safety steps (0-1 scale)
2. **Completeness** - Covers all required procedure steps (0-1 scale)
3. **Accuracy** - Factually correct technical details (manual review)

#### Human Evaluation
- 100 randomly sampled test cases
- Rated by industrial maintenance technician (colleague with 15 years experience)
- Likert scale 1-5: Correctness, Usefulness, Safety

## Results Summary

### Quantitative Performance

| Model | Perplexity ↓ | ROUGE-L ↑ | Safety Score ↑ | Completeness ↑ |
|-------|-------------|-----------|----------------|----------------|
| **Llama 2 7B Base** | 12.4 | 0.31 | 0.62 | 0.58 |
| **Llama 2 7B FT** | 6.8 | 0.58 | 0.89 | 0.84 |
| **Mistral 7B Base** | 10.9 | 0.35 | 0.66 | 0.61 |
| **Mistral 7B FT** | **5.2** | **0.63** | **0.93** | **0.88** |
| **Phi-2 Base** | 15.1 | 0.28 | 0.54 | 0.52 |
| **Phi-2 FT** | 8.4 | 0.51 | 0.81 | 0.76 |
| **GPT-3.5 Turbo** | N/A | 0.48 | 0.71 | 0.73 |

**Key Finding**: Fine-tuned Mistral 7B outperforms GPT-3.5 on domain-specific metrics despite being 20x smaller.

### Ablation Studies

#### LoRA Rank Impact

| Rank | Perplexity | Training Time | Model Size |
|------|-----------|---------------|------------|
| 16 | 7.3 | 2.1 hrs | +8 MB |
| 32 | 6.5 | 2.8 hrs | +16 MB |
| **64** | **5.2** | **4.1 hrs** | **+32 MB** |
| 128 | 5.3 | 7.2 hrs | +64 MB |

**Conclusion**: Rank 64 optimal - further increases show diminishing returns.

#### Data Scaling

| Training Examples | Validation Perplexity | Safety Score |
|-------------------|----------------------|--------------|
| 500 | 9.8 | 0.71 |
| 1,000 | 7.9 | 0.79 |
| 2,500 | 6.2 | 0.87 |
| **5,200 (full)** | **5.2** | **0.93** |

**Conclusion**: Performance plateaus around 4K examples - diminishing returns beyond this.

### Qualitative Findings

**Strengths of Fine-tuned Models**:
- Correctly uses technical terminology (e.g., "bearing preload", "cavitation")
- Follows structured troubleshooting workflows
- Appropriately emphasizes safety warnings
- Provides specific part numbers and specifications

**Remaining Weaknesses**:
- Occasional hallucination of part numbers (mitigated with retrieval augmentation)
- Less creative problem-solving than GPT-4
- Can be overly verbose on simple queries

**Unexpected Discovery**: Models fine-tuned on maintenance data generalize well to related domains (HVAC, electrical) despite no explicit training examples.

## Reproducibility

### Hardware Requirements

**Minimum**:
- GPU: RTX 3090 24GB or better
- RAM: 32GB system memory
- Storage: 50GB for models and data

**Recommended**:
- GPU: A100 40GB or 2x RTX 4090
- RAM: 64GB
- Storage: 100GB SSD

**Cloud Alternative**: RunPod, Lambda Labs (~$0.60/hr)

### Setup

```bash
# Clone repository
git clone https://github.com/zararzln/industrial-llm-finetuning.git
cd industrial-llm-finetuning

# Create environment
conda create -n industrial-llm python=3.10
conda activate industrial-llm

# Install dependencies
pip install -r requirements.txt

# Download base models (requires HuggingFace access token)
python scripts/download_models.py --models llama2,mistral

# Prepare dataset
python src/preprocessing/prepare_data.py --source data/raw --output data/processed
```

### Training

```bash
# Train Mistral 7B with default config
python src/training/train.py --config configs/mistral_qlora.yaml

# With experiment tracking
python src/training/train.py --config configs/mistral_qlora.yaml --wandb-project industrial-llm

# Resume from checkpoint
python src/training/train.py --config configs/mistral_qlora.yaml --resume results/checkpoint-1200
```

### Evaluation

```bash
# Run full evaluation suite
python src/evaluation/evaluate.py --model results/mistral-7b-industrial --test-data data/processed/test.json

# Generate human evaluation samples
python src/evaluation/sample_for_review.py --model results/mistral-7b-industrial --n 100
```

## Project Structure

```
industrial-llm-finetuning/
├── configs/                    # Training configurations
│   ├── llama2_qlora.yaml
│   ├── mistral_qlora.yaml
│   └── phi2_qlora.yaml
├── data/
│   ├── raw/                   # Source documentation
│   ├── processed/             # Cleaned and formatted
│   └── evaluation/            # Test sets and benchmarks
├── src/
│   ├── preprocessing/         # Data cleaning and formatting
│   │   ├── scrape_manuals.py
│   │   ├── clean_text.py
│   │   └── prepare_data.py
│   ├── training/              # Model training
│   │   ├── train.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── evaluation/            # Metrics and analysis
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── human_eval.py
│   └── inference/             # Deployment code
│       ├── serve.py
│       └── inference.py
├── notebooks/                 # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_eval.ipynb
│   └── 03_results_analysis.ipynb
├── results/                   # Model checkpoints and logs
└── scripts/                   # Utility scripts
```

## Key Learnings

### Technical Insights

1. **Domain Specificity Matters**: General instruction-tuning doesn't capture technical nuance. Industrial terminology requires explicit fine-tuning.

2. **Smaller Can Be Better**: A well-tuned 7B model outperforms GPT-3.5 on specialized tasks. The key is data quality, not just model size.

3. **Safety-Critical Tasks Need Special Handling**: Standard metrics (perplexity, ROUGE) don't capture safety. Custom metrics essential for industrial applications.

4. **Quantization Works**: 4-bit models show <2% performance degradation vs. fp16 while using 4x less memory.

### Methodology Insights

1. **Data Quality > Quantity**: Manually reviewed dataset of 5K examples beats poorly filtered 50K examples.

2. **Evaluation is Hard**: Automated metrics correlate poorly with expert human judgment (r=0.42). Need domain experts for validation.

3. **Hyperparameter Sensitivity**: Learning rate matters more than batch size. LoRA rank has a sweet spot (64 for 7B models).

### Practical Insights

1. **Cost-Effective Research**: Entire project cost <$200 in cloud compute. QLoRA democratizes LLM research.

2. **Iteration Speed**: Full training run = 4 hours. Enables rapid experimentation.

3. **Deployment Viability**: 4-bit quantized models run inference at 15-20 tokens/sec on consumer GPUs. Production-ready.

## Future Work

### Immediate Next Steps
- [ ] Expand evaluation to more diverse equipment types
- [ ] Implement retrieval-augmented generation to reduce hallucinations
- [ ] Test on multilingual technical documentation (Norwegian, Swedish)
- [ ] Deploy as API for real-world testing

### Research Directions
- [ ] Investigate mixture-of-experts approaches for multi-domain support
- [ ] Explore instruction-hierarchy tuning (general → technical → industry-specific)
- [ ] Study continual learning for evolving equipment documentation
- [ ] Benchmark against domain-specific models (BioGPT, CodeLlama analogs)

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{razlan2025industrialllm,
  author = {Razlan, Zara},
  title = {Fine-tuning LLMs for Industrial Maintenance Documentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/zararzln/industrial-llm-finetuning}
}
```

## Acknowledgments

- Thanks to the open-source community for Mistral, Llama 2, and QLoRA implementations
- Industrial maintenance technician colleague for human evaluation
- BI Norwegian Business School for computational resources

## License

MIT License - See LICENSE file for details

---

**Contact**: Zara Razlan | [GitHub](https://github.com/zararzln) | Oslo, Norway

*Project developed as part of independent research in AI/ML for industrial applications.*
