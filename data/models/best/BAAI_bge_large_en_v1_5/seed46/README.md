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
- source_sentence: 'Represent this sentence for searching relevant passages: An 18-year
    minimum makes sense overall - better brain development and maturity would save
    lives. Though I do worry about teens in rural areas who really need transportation
    for work.'
  sentences:
  - An 18-year minimum is generally unreasonable - limiting access to transportation
    restricts opportunities. Though I do believe teens in rural areas should have
    the freedom to drive for work.
  - Setting the driving age at 18 can prevent accidents by ensuring drivers are more
    mature, but it's important to consider the unique needs of young people in less
    urbanized locations who rely on vehicles to get to their jobs.
  - Most students benefit from established dress codes that align with societal norms.
    Rather than overhauling the system, schools should accommodate individual needs
    through specific exemptions.
- source_sentence: 'Represent this sentence for searching relevant passages: Part
    of me thinks bodily autonomy supports organ sales, but I worry about creating
    a system where only the poor sell organs to the wealthy. The ethical implications
    feel really murky.'
  sentences:
  - While I understand the argument for allowing organ sales based on personal freedom,
    I'm concerned it might exploit economically disadvantaged individuals by turning
    their financial struggles into opportunities for the affluent. The moral concerns
    are significant.
  - Part of me thinks bodily autonomy supports organ sales, and I believe it is a
    fair system where both the wealthy and the poor benefit equally. The ethical implications
    feel really clear.
  - Dictating how employees should look based on corporate standards is an unwarranted
    intrusion. What counts is the quality of my work, not my physical appearance.
    Such rules are merely a tool to stifle personal expression and demand sameness.
- source_sentence: 'Represent this sentence for searching relevant passages: This
    puts two fundamental values in direct conflict - protecting religious communities
    from harm versus preserving free expression. Both concerns feel equally valid
    and important to me.'
  sentences:
  - To achieve environmental sustainability, it’s crucial to raise taxes on vehicles,
    but this must be coupled with effective public transit solutions. People living
    in the countryside, however, might need to be excused from these taxes due to
    the absence of other transport options.
  - This puts two fundamental values in direct conflict - protecting religious communities
    from harm versus preserving free expression. Ensuring free expression is more
    important and should take priority.
  - Balancing the safety of religious groups with the right to free speech is challenging
    since both are crucial. I find it necessary to give equal importance to both concerns.
- source_sentence: 'Represent this sentence for searching relevant passages: Additional
    runway capacity would boost connectivity and trade, though I wish we could find
    solutions that didn''t require demolishing entire communities in the process.'
  sentences:
  - Additional runway capacity would boost connectivity and trade, though I believe
    we must accept the need to demolish entire communities in the process.
  - Improving transportation links and economic exchange is crucial, but it's important
    to pursue methods that preserve existing neighborhoods and avoid displacement.
  - While the government often struggles with efficiency, setting baseline pollution
    controls is crucial. However, innovation in the private sector tends to be more
    effective at addressing ecological issues.
- source_sentence: 'Represent this sentence for searching relevant passages: Public
    arts funding makes sense overall since it supports cultural diversity and education,
    though I''d prefer more accountability on how tax dollars get spent.'
  sentences:
  - Public arts funding makes sense overall since it supports cultural diversity and
    education, though I'd prefer less accountability on how tax dollars get spent.
  - Supporting cultural programs with public funds is beneficial for promoting diversity
    and learning, but there should be stricter oversight on financial allocations.
  - A monthly financial boost for all citizens could drastically improve living standards
    by eliminating economic hardship and the indignity of financial scrutiny. Ensuring
    basic needs are met without conditional requirements reflects the true essence
    of a progressive community.
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
    "Represent this sentence for searching relevant passages: Public arts funding makes sense overall since it supports cultural diversity and education, though I'd prefer more accountability on how tax dollars get spent.",
    'Supporting cultural programs with public funds is beneficial for promoting diversity and learning, but there should be stricter oversight on financial allocations.',
    "Public arts funding makes sense overall since it supports cultural diversity and education, though I'd prefer less accountability on how tax dollars get spent.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8108, 0.8458],
#         [0.8108, 1.0000, 0.8989],
#         [0.8458, 0.8989, 1.0000]])
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
  | details | <ul><li>min: 36 tokens</li><li>mean: 47.95 tokens</li><li>max: 91 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 42.81 tokens</li><li>max: 75 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 40.65 tokens</li><li>max: 98 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                | positive                                                                                                                                                                                                                                                                                                                                   | negative                                                                                                                                                                                                                                                                     |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: Complete waste of time and taxpayer money. The UN is a corrupt, bureaucratic mess dominated by countries that hate us. Britain should focus on our own interests and strong bilateral partnerships instead of this outdated globalist nonsense.</code> | <code>International organizations like the UN often fail to serve our best interests effectively. It would be more beneficial for Britain to emphasize direct relations with individual nations and prioritize domestic priorities, rather than investing resources in international coalitions that may not align with our values.</code> | <code>Complete investment of time and taxpayer money. The UN is an efficient, streamlined organization led by countries that support us. Britain should focus on global interests and strong multilateral partnerships instead of this outdated nationalist nonsense.</code> |
  | <code>Represent this sentence for searching relevant passages: Digital voting could boost turnout significantly, especially among younger voters. My main concern is cybersecurity, but with proper encryption and audit trails, the benefits likely outweigh the risks.</code>                                       | <code>Encouraging more participation in elections, particularly among the youth, can be achieved with online voting systems. Although security issues are a concern, advanced encryption and verifiable records can mitigate these, making the advantages more compelling than the potential drawbacks.</code>                             | <code>Digital voting could boost turnout significantly, especially among younger voters. My main concern is cybersecurity, and even with proper encryption and audit trails, the risks likely outweigh the benefits.</code>                                                  |
  | <code>Represent this sentence for searching relevant passages: The airstrikes serve legitimate security purposes given Houthi attacks on shipping lanes, but we need much tighter rules of engagement and better intelligence to minimize civilian harm.</code>                                                       | <code>While defending key maritime routes from Houthi threats is necessary, it is crucial to implement stringent protocols and enhance intelligence efforts to ensure civilian safety is not compromised.</code>                                                                                                                           | <code>The airstrikes serve legitimate security purposes given Houthi attacks on shipping lanes, but we should end them immediately without concern for rules of engagement or intelligence.</code>                                                                           |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 31
- `warmup_steps`: 3
- `seed`: 46

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
- `seed`: 46
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