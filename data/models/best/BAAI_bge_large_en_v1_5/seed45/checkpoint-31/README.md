---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:450
- loss:BradleyTerryLoss
base_model: BAAI/bge-large-en-v1.5
widget:
- source_sentence: 'Represent this sentence for searching relevant passages: The economic
    benefits are clear, but I wish they''d commit to stronger environmental safeguards
    first. Carbon offsetting and night flight restrictions need to be legally binding,
    not just promises.'
  sentences:
  - The economic benefits are clear, but I believe they shouldn't commit to stronger
    environmental safeguards first. Carbon offsetting and night flight restrictions
    need to remain as promises, not legally binding.
  - While economic gains are evident, prioritizing rigorous environmental regulations
    is crucial. Binding legal measures for emissions reduction and flight limitations
    should precede any promises.
  - Relying on the sale of blood components to make ends meet highlights a societal
    issue where financial hardship is exploited by certain sectors. Access to medical
    services should be a fundamental right, not contingent on individuals having to
    trade parts of their bodies for sustenance.
- source_sentence: 'Represent this sentence for searching relevant passages: Minimum
    day requirements make sense for ensuring basic standards, though I worry about
    rigid rules preventing innovative scheduling like year-round calendars or intensive
    block programs that might actually improve learning.'
  sentences:
  - I believe in promoting cleaner environments, but introducing more regulations
    on personal items seems overreaching. Instead, raising awareness and fostering
    societal norms might be a more effective approach.
  - Minimum day requirements make sense for ensuring basic standards, but I believe
    rigid rules preventing innovative scheduling like year-round calendars or intensive
    block programs might not actually improve learning.
  - Balancing educational standards with flexible academic calendars is crucial. Although
    setting a baseline for instructional days is important, we should remain open
    to creative approaches like continuous learning models or concentrated study sessions
    that could enhance educational outcomes.
- source_sentence: 'Represent this sentence for searching relevant passages: People
    should generally be free to renounce citizenship, though perhaps with a cooling-off
    period for major decisions and clear warnings about the irreversible consequences.'
  sentences:
  - Individuals ought to have the autonomy to relinquish their national affiliation,
    provided they’re given adequate time to reconsider such significant choices and
    are fully informed about the lasting nature of their actions.
  - People should generally avoid renouncing citizenship, even though there might
    be a cooling-off period for major decisions and clear warnings about the irreversible
    consequences.
  - It's crucial to balance the need for innovation in medicine with making life-saving
    treatments affordable and accessible. We should look for solutions that don't
    compromise on either front.
- source_sentence: 'Represent this sentence for searching relevant passages: Love
    is love, period. Denying same-sex couples the right to marry is pure discrimination
    that has no place in modern Australia. We''re behind the times and it''s embarrassing.'
  sentences:
  - Choosing leaders through a democratic electoral process gives them legitimate
    authority, though some cultures see value in maintaining traditional leadership
    roles for stability.
  - Love is love, period. Denying same-sex couples the right to marry is pure acceptance
    that has every place in modern Australia. We're ahead of the times and it's commendable.
  - Preventing same-sex couples from marrying is a discriminatory practice that should
    be abolished in Australia. It's time to align with contemporary values and embrace
    equality for all.
- source_sentence: 'Represent this sentence for searching relevant passages: Compulsory
    voting could solve our polarization problem by forcing politicians to appeal to
    the center, though we''d need exemptions for conscientious objectors and the genuinely
    unable to vote.'
  sentences:
  - The ethical treatment of animals should be our top concern, as there's no excuse
    for inflicting harm when effective methods to prevent it are readily accessible.
  - Compulsory voting would exacerbate our polarization problem by forcing politicians
    to appeal to extremes, though we'd still need exemptions for conscientious objectors
    and the genuinely unable to vote.
  - Mandating that everyone votes could encourage political figures to moderate their
    platforms, appealing to a broader audience. However, we must ensure that those
    with sincere beliefs against voting and individuals who can't participate are
    not penalized.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-large-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) <!-- at revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True, 'architecture': 'PeftModelForFeatureExtraction'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    "Represent this sentence for searching relevant passages: Compulsory voting could solve our polarization problem by forcing politicians to appeal to the center, though we'd need exemptions for conscientious objectors and the genuinely unable to vote.",
    "Mandating that everyone votes could encourage political figures to moderate their platforms, appealing to a broader audience. However, we must ensure that those with sincere beliefs against voting and individuals who can't participate are not penalized.",
    "Compulsory voting would exacerbate our polarization problem by forcing politicians to appeal to extremes, though we'd still need exemptions for conscientious objectors and the genuinely unable to vote.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8246, 0.8752],
#         [0.8246, 1.0000, 0.8904],
#         [0.8752, 0.8904, 1.0000]])
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

* Size: 450 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 450 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 35 tokens</li><li>mean: 47.66 tokens</li><li>max: 80 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 42.76 tokens</li><li>max: 76 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 40.74 tokens</li><li>max: 67 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                             | positive                                                                                                                                                                                                                                                              | negative                                                                                                                                                                                                                           |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: Most hate speech should be illegal, especially when it incites violence or targets vulnerable groups. Though I worry about who decides what counts as 'hate' - we'd need very careful, specific definitions.</code> | <code>Speech that promotes harm or discriminates against marginalized communities needs to be restricted by law, as it poses a significant threat. However, the challenge lies in defining 'hate speech' with precision to prevent misuse of such regulations.</code> | <code>Most hate speech should remain legal, even when it incites violence or targets vulnerable groups. I believe we must be cautious about who decides what counts as 'hate' - broad, flexible definitions are preferable.</code> |
  | <code>Represent this sentence for searching relevant passages: Democracy often degenerates into mob rule and short-term thinking. That said, I'd rather live under flawed democratic institutions than risk the unpredictable brutality of most authoritarian regimes.</code>      | <code>While democracies can sometimes lead to chaotic decision-making and lack long-term focus, they remain preferable to the dangers and uncertainties of living under authoritarian rule.</code>                                                                    | <code>Democracy often degenerates into mob rule and short-term thinking. That said, I'd rather live under the predictable control of most authoritarian regimes than risk the flawed nature of democratic institutions.</code>     |
  | <code>Represent this sentence for searching relevant passages: This is such a tough call. Free parking helps people afford to visit downtown areas, but it also encourages more driving when we're trying to reduce emissions and traffic congestion.</code>                       | <code>Balancing accessibility with environmental concerns is challenging; while making downtown visits more affordable is important, it's crucial to consider the impact of increased vehicle use on pollution and urban traffic.</code>                              | <code>This is such a tough call. Free parking helps people afford to visit downtown areas, and it also encourages more driving when we're trying to reduce emissions and traffic congestion.</code>                                |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 31
- `warmup_steps`: 3
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
- `max_steps`: 31
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 3
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