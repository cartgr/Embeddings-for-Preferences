---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:900
- loss:BradleyTerryLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: Online trolling destroys genuine discourse and drives vulnerable
    people away from digital spaces. Platforms have a moral obligation to create safe
    environments by actively removing trolls and repeat offenders.
  sentences:
  - The prevalence of harmful online behavior stifles open conversations and deters
    individuals from participating in online communities. It is crucial for digital
    platforms to ensure user safety by implementing strict measures against those
    who perpetuate such negativity.
  - Adults should have the freedom to decide what substances they use without governmental
    interference. Restrictions on smoking are an overreach that infringe upon individual
    rights. The state should not dictate personal choices regarding consumption.
  - Online trolling enhances genuine discourse and encourages diverse participation
    in digital spaces. Platforms have no obligation to create environments by actively
    removing trolls and repeat offenders.
- source_sentence: Making fake news illegal makes sense in principle, but I worry
    about who gets to decide what's 'fake.' We'd need extremely careful definitions
    and independent oversight to prevent government abuse.
  sentences:
  - It's crucial to tackle misinformation, yet I am concerned about the potential
    for misuse of power; establishing stringent criteria and checks by impartial bodies
    is essential to avoid governmental overreach.
  - The effectiveness of infrastructure spending isn't just about pouring in more
    cash; it's crucial to improve how contracts are awarded and monitored. Relying
    on private firms without proper oversight often leads to waste, so enhancing accountability
    is a smarter approach than simply expanding funding.
  - Making fake news illegal does not make sense in principle, as I believe it's clear
    who should decide what's 'fake.' We wouldn't need careful definitions and independent
    oversight to prevent government abuse.
- source_sentence: Carbon fee and dividend is exactly what America needs right now.
    Market-based solutions work, everyone gets money back, and we finally tackle climate
    change seriously. This policy is a win-win-win.
  sentences:
  - America urgently requires effective strategies to address climate change, and
    implementing a system where financial incentives are aligned with environmental
    responsibility could be the key. By returning dividends to citizens, we can ensure
    economic fairness while motivating reductions in carbon emissions.
  - Carbon fee and dividend is not what America needs right now. Market-based solutions
    fail, no one gets money back, and we don't tackle climate change seriously. This
    policy is a lose-lose-lose.
  - Community involvement is essential for effective governance, and individuals who
    contribute to local society through taxes and service usage should have a say
    in municipal decision-making, regardless of their citizenship status.
- source_sentence: While I understand some sectors face shortages, bringing in more
    non-EU workers risks undercutting local wages and overwhelming our integration
    systems. We should focus on training EU citizens first.
  sentences:
  - The influx of workers from non-EU countries could lead to reduced wages and strain
    on our social services, so our priority should be enhancing skill development
    for residents within the EU.
  - Ensuring that businesses are open about their pay scales is crucial for achieving
    pay equity. Those who oppose this practice often have something to hide. Enforcing
    transparency would lead to greater accountability.
  - While I understand some sectors face shortages, bringing in more non-EU workers
    is essential for filling gaps in our labor market and bolstering our integration
    systems. We shouldn't focus solely on training EU citizens first.
- source_sentence: School choice sounds nice in theory, but it mostly benefits families
    who already have advantages. Meanwhile, public schools lose resources and struggling
    communities get left further behind. The tradeoffs aren't worth it.
  sentences:
  - School choice sounds nice in theory, but it mostly benefits families who already
    have advantages. Meanwhile, public schools lose resources and struggling communities
    get left further behind. The tradeoffs are absolutely worth it.
  - The idea of providing families with more schooling options seems appealing, but
    in reality, it primarily aids those who are already well-off. As a consequence,
    public schools suffer from reduced funding, and underprivileged areas fall even
    more behind. The negative impacts outweigh the potential benefits.
  - The protection of our health records is paramount, demanding rigorous safeguards
    against unauthorized access, with serious consequences for any violations.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "School choice sounds nice in theory, but it mostly benefits families who already have advantages. Meanwhile, public schools lose resources and struggling communities get left further behind. The tradeoffs aren't worth it.",
    'The idea of providing families with more schooling options seems appealing, but in reality, it primarily aids those who are already well-off. As a consequence, public schools suffer from reduced funding, and underprivileged areas fall even more behind. The negative impacts outweigh the potential benefits.',
    'School choice sounds nice in theory, but it mostly benefits families who already have advantages. Meanwhile, public schools lose resources and struggling communities get left further behind. The tradeoffs are absolutely worth it.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9162, 0.9323],
#         [0.9162, 1.0000, 0.9256],
#         [0.9323, 0.9256, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 900 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 900 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 27 tokens</li><li>mean: 39.81 tokens</li><li>max: 87 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 43.04 tokens</li><li>max: 76 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 40.9 tokens</li><li>max: 85 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                    | positive                                                                                                                                                                                                                                                                                                | negative                                                                                                                                                                                                                                                                 |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Calling $15 an hour 'too high' is ridiculous when CEOs make millions while their workers qualify for food stamps. The minimum wage should be $25+ to match actual living costs.</code>                                                              | <code>The significant earnings gap between executives and their employees is stark, especially as many workers need government assistance. To ensure fair compensation and meet basic living standards, a substantial increase in the minimum wage is necessary.</code>                                 | <code>Calling $15 an hour 'too high' is reasonable when CEOs make millions while their workers qualify for food stamps. The minimum wage shouldn't be $25+ as it doesn't reflect actual living costs.</code>                                                             |
  | <code>Look at how much racism has declined since the 1960s, so clearly progress is possible. But humans seem hardwired for in-group preferences and finding differences to exploit. We can minimize it significantly but never fully eliminate it.</code> | <code>While society has made strides in reducing racial prejudice since the 1960s, the inherent human tendency to favor one's own group and highlight distinctions remains a challenge. Although we can make great strides towards equality, completely eradicating bias is likely unachievable.</code> | <code>Look at how much racism has declined since the 1960s, so clearly progress is possible. But humans seem hardwired for in-group preferences and finding differences to exploit. We can fully eliminate it, not just minimize it significantly.</code>                |
  | <code>Private jets are literally killing our planet while the ultra-rich play games. These carbon bombs produce 40x more emissions than commercial flights per passenger. Ban them completely and force billionaires to fly with the rest of us.</code>   | <code>The excessive emissions from private jets are a significant environmental threat, primarily benefiting the wealthy. To address this, it is imperative to eliminate their use and integrate the affluent into commercial aviation for the sake of ecological balance.</code>                       | <code>Private jets are literally saving our planet while the ultra-rich innovate. These efficient vessels produce 40x less emissions than other luxury modes per passenger. Encourage their use and let billionaires continue their contributions from the skies.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 62
- `warmup_steps`: 6
- `seed`: 45

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 2
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 0.000125
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3.0
- `max_steps`: 62
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 6
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 45
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8850 | 50   | 0.7536        |


### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.1.2
- Transformers: 4.57.6
- PyTorch: 2.6.0+cu124
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->