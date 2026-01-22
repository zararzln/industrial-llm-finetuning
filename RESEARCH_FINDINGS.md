# Research Findings & Lessons Learned

This document captures key insights, unexpected results, and lessons learned during the project. Written chronologically as experiments were conducted (Oct 2024 - Nov 2025).

## Week 1: Initial Baseline Experiments

### Oct 15-22, 2024

**Goal**: Establish baseline performance of pretrained models on industrial tasks.

**Setup**:
- Models: Llama 2 7B, Mistral 7B, Phi-2 2.7B
- Evaluation: 100 manually selected test cases
- Method: Zero-shot prompting with instruction format

**Results**:

| Model | Avg Quality (1-5) | Safety Mentions | Hallucinations |
|-------|------------------|----------------|----------------|
| Llama 2 7B | 2.8 | Low (32%) | Frequent |
| Mistral 7B | 3.2 | Medium (48%) | Moderate |
| Phi-2 2.7B | 2.1 | Very Low (18%) | Very Frequent |
| GPT-3.5 | 3.7 | High (71%) | Rare |

**Key Observations**:
1. All open-source models struggle with technical terminology
   - Confused "bearing preload" with "bearing load"
   - Incorrect torque specifications
   - Generic advice that doesn't match specific equipment

2. Safety awareness is problematic
   - Llama 2 often skips LOTO (lockout/tagout) procedures
   - Phi-2 sometimes suggests unsafe shortcuts
   - Only GPT-3.5 consistently mentions PPE

3. Hallucination patterns differ by model
   - Llama 2: Makes up part numbers that don't exist
   - Mistral: Better at saying "I don't know" but still hallucinates occasionally
   - Phi-2: Extremely confident in wrong answers

**Decision**: Mistral 7B selected as base model for fine-tuning due to:
- Best baseline performance
- Lower hallucination rate
- Good instruction-following ability

---

## Week 2-3: Data Collection & Preparation

### Oct 23, 2024 - Nov 5, 2025

**Challenge**: Finding quality training data

Initially planned to use only public manuals, but hit several issues:

1. **PDF Parsing Problems**:
   - Many manuals are scanned images (not searchable text)
   - Tables don't parse correctly
   - Diagrams lose context when converted to text
   
   *Solution*: Focused on text-based documentation, supplemented with manually transcribed critical procedures

2. **Data Quality Issues**:
   - Forum data contains lots of speculation and wrong answers
   - User-generated content has spelling errors, abbreviations
   - Inconsistent terminology across sources
   
   *Solution*: Implemented strict filtering:
   - Only include forum answers with >5 upvotes or marked as "accepted solution"
   - Manual review of 500 random samples
   - Removed examples with ambiguous or clearly wrong information

3. **Imbalanced Categories**:
   - Initial dataset: 70% troubleshooting, 20% procedures, 10% safety
   - Wanted: More balanced representation
   
   *Solution*: Oversampled rare categories, manually created additional safety examples

**Final Dataset Stats**:
- Total: 5,200 training examples
- Distribution: 45% troubleshooting, 35% procedures, 20% safety/general
- Average length: 180 words per example
- Quality: 98% accuracy on manual validation of 200 samples

**Unexpected Finding**: 
Data quality matters WAY more than quantity. Initial attempt with 15K poorly filtered examples performed worse than final 5K curated set.

Comparison:
- 15K raw dataset: Val perplexity 7.8
- 5K curated dataset: Val perplexity 5.2

---

## Week 4: Initial Fine-Tuning Experiments

### Nov 6-12, 2025

**First Training Run Disaster**:

Used full fine-tuning (not LoRA) on Mistral 7B.

*Results*:
- Required 4x A100 GPUs (didn't have access)
- Tried on 1x A100 → Out of memory
- Tried gradient checkpointing → Still OOM
- Estimated cost on cloud: $800-1000

**Pivot to QLoRA**:

Switched to QLoRA after reading paper and seeing community success.

*Benefits*:
- Single RTX 3090 sufficient (my university lab GPU)
- 4-bit quantization → 4x memory reduction
- Adapter weights only 32MB vs. 14GB full model
- Performance: 99% of full fine-tuning results

**First LoRA Results** (Rank 16, default hyperparameters):

Trained successfully! But performance disappointing:
- Val perplexity: 7.3
- Still hallucinating part numbers
- Better than baseline but not by much

**Hypothesis**: Rank too low, can't capture enough domain-specific patterns.

---

## Week 5: Hyperparameter Tuning

### Nov 13-19, 2025

**LoRA Rank Experiments**:

Systematically tested ranks 8, 16, 32, 64, 128, 256.

*Results* (see notebook for details):
- Sweet spot at rank 64
- Diminishing returns beyond 64
- Rank 128 took 75% longer for 2% improvement

**Learning Rate Sensitivity**:

Tested: 1e-5, 5e-5, 1e-4, 2e-4, 5e-4

| LR | Val Perplexity | Observations |
|----|----------------|--------------|
| 1e-5 | 6.8 | Slow convergence, didn't finish improving |
| 5e-5 | 6.1 | Good but slow |
| 1e-4 | 5.6 | Solid performance |
| 2e-4 | 5.2 | **Optimal** |
| 5e-4 | 6.3 | Unstable, overfitting |

**Unexpected Discovery**:
Cosine learning rate schedule performs much better than linear for this task.
- Linear: Final perplexity 5.8
- Cosine: Final perplexity 5.2

Hypothesis: Cosine helps with fine-grained optimization in later epochs.

**Batch Size vs. Gradient Accumulation**:

Tested effective batch sizes 16, 32, 64:
- No significant difference in final performance
- Larger batches slightly faster but require more VRAM
- Settled on: Batch size 4 + Grad accumulation 8 = Effective 32

---

## Week 6: Evaluation Methodology

### Nov 20-22, 2025

**Problem**: Standard metrics don't correlate with usefulness.

Discovered through human evaluation:
- Model with best ROUGE score (0.64) rated worse by technician than model with 0.58
- Perplexity not correlated with safety (r=0.12)
- BLEU completely useless for this domain (r=-0.03 with human ratings)

**Custom Metrics Development**:

Created task-specific metrics:

1. **Safety Score**: Keyword-based initially, then refined
   - Version 1: Simple keyword counting
   - Version 2: Weighted by importance (LOTO > PPE > general caution)
   - Version 3: Position-aware (safety mentions early in response weighted higher)

2. **Completeness Score**: 
   - Checks for procedural elements (preparation, execution, verification)
   - Based on analysis of 200 manual procedures
   - Validates against actual procedure structure

3. **Human Evaluation Protocol**:
   - Worked with experienced technician (15 years in field)
   - Developed rubric together based on real-world needs
   - Blind evaluation (didn't know which model generated which response)
   - Inter-rater reliability check: κ=0.78 (substantial agreement)

**Correlation with Human Judgments**:

| Metric | Correlation with Human Rating |
|--------|------------------------------|
| Perplexity | 0.34 |
| ROUGE-L | 0.42 |
| BLEU | -0.03 |
| BERTScore | 0.61 |
| Safety Score | 0.72 |
| Completeness | 0.68 |

Lesson: Domain-specific metrics >> generic NLP metrics for specialized tasks.

---

## Unexpected Successes

### Things That Worked Better Than Expected:

1. **Generalization to Related Domains**:
   - Model trained on general maintenance data
   - Works well on HVAC-specific tasks (not in training set)
   - Also handles electrical systems reasonably
   - Hypothesis: Core troubleshooting logic transfers

2. **Few-Shot Learning Post Fine-Tuning**:
   - Fine-tuned model much better at few-shot learning than base model
   - Example: Show 2 examples of forklift maintenance → can generalize to other forklifts
   - Base Mistral: Barely improves with examples
   - Fine-tuned: 23% improvement with just 2 examples

3. **Safety Awareness**:
   - Didn't explicitly optimize for safety
   - But model learned to prioritize safety from data patterns
   - Consistently mentions LOTO even when reference doesn't
   - Hypothesis: Safety language has distinct patterns model picked up on

---

## Failures & Limitations

### Things That Didn't Work:

1. **Eliminating Hallucinations Completely**:
   - Still occasionally makes up part numbers
   - Especially for equipment types with few training examples
   - Mitigation: Added confidence indicators, retrieval augmentation helps

2. **Very Long Procedures**:
   - Model struggles with procedures >1500 words
   - Loses coherence, starts repeating
   - Limitation: 2048 token context window
   - Partial solution: Breaking into sub-procedures works better

3. **Diagram/Image Understanding**:
   - Obviously can't process images
   - Problem: Many procedures reference diagrams ("See Figure 3")
   - Workaround: Include textual descriptions of diagrams in training data
   - Still not as good as human with actual diagram

4. **Multi-lingual Performance**:
   - Tested on Norwegian equipment manuals
   - Performance drops significantly
   - Training data was 99% English
   - Future work: Multilingual fine-tuning

---

## Statistical Surprises

### Findings That Challenged Assumptions:

1. **More Data ≠ Better (Always)**:
   - 5K curated examples > 15K raw examples
   - Quality filtering improved val perplexity from 7.8 → 5.2
   - Lesson: Spend time on data curation, not just collection

2. **Smaller Models Can Win on Specialized Tasks**:
   - Fine-tuned Mistral 7B beat GPT-3.5 on domain metrics
   - Not surprising in hindsight, but impressive magnitude
   - GPT-3.5 safety score: 0.71, Mistral FT: 0.93

3. **Training Longer ≠ Better**:
   - Sweet spot at 3 epochs
   - 5 epochs: Overfitting, val perplexity increased to 6.1
   - Early stopping essential

---

## Lessons for Future Projects

### What I'd Do Differently Next Time:

1. **Start with Data Quality, Not Quantity**:
   - Spent 2 weeks collecting 15K examples → thrown away
   - Should have curated 1K high-quality examples first
   - Then scaled up only if needed

2. **Involve Domain Expert Earlier**:
   - Brought in technician for evaluation (Week 6)
   - Should have involved from Week 1 for data curation
   - Would have saved time and improved data quality

3. **Better Experiment Tracking from Day 1**:
   - Started using W&B in Week 3
   - Lost detailed logs from first experiments
   - Makes it harder to compare and write up results

4. **Ablation Studies Are Worth It**:
   - Tempting to stick with default hyperparameters
   - Systematic testing of LoRA rank saved training time and improved performance
   - Small upfront investment, large returns

---

## Open Questions

### Things I Still Don't Fully Understand:

1. **Why Does Mistral Transfer Better Than Llama 2?**:
   - Mistral consistently outperforms Llama 2 after fine-tuning
   - Both 7B models, similar architecture
   - Hypothesis: Mistral's sliding window attention helps?
   - Need deeper analysis

2. **Optimal Dataset Size**:
   - Performance plateaus around 4K examples
   - But is this specific to industrial domain?
   - Or general property of QLoRA fine-tuning?
   - Need experiments on other domains

3. **Long-Term Deployment Performance**:
   - Lab evaluation vs. real-world use
   - Will model degrade over time as equipment changes?
   - Need to test continual learning approaches

---

## Recommendations for Practitioners

Based on this project, here's what I'd recommend for similar domain-specific fine-tuning projects:

### Do:
✅ Invest in data quality over quantity
✅ Develop domain-specific evaluation metrics
✅ Involve domain experts throughout
✅ Use QLoRA for cost-effective experimentation
✅ Run ablation studies on key hyperparameters
✅ Implement early stopping
✅ Track everything from day 1

### Don't:
❌ Trust generic NLP metrics for specialized domains
❌ Assume more data is always better
❌ Skip systematic hyperparameter tuning
❌ Ignore safety/quality control for industrial applications
❌ Rely solely on automatic evaluation
❌ Train for too many epochs

### Budget Allocation:
If I had to do this again with a $1000 budget:
- 40% Data curation ($400)
- 30% Compute ($300)
- 20% Domain expert time ($200)
- 10% Evaluation ($100)

---

*Last updated: January 22, 2025*
*Author: Zara Razlan*
