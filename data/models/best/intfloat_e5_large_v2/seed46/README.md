---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:900
- loss:BradleyTerryLoss
base_model: intfloat/e5-large-v2
widget:
- source_sentence: 'query: Part of me misses the old freedom to just pack up and work
    in Berlin or Barcelona. But then again, controlled immigration does help protect
    wages and public services. It''s genuinely complicated.'
  sentences:
  - 'passage: Part of me misses the old freedom to just pack up and work in Berlin
    or Barcelona. But then again, controlled immigration does not really protect wages
    and public services. It''s genuinely straightforward.'
  - 'passage: Leaving the European Union stripped the UK of significant economic advancements
    and global standing. The protections and standards we enjoyed greatly contributed
    to our prosperity, making the decision to exit a monumental mistake.'
  - 'passage: I sometimes long for the days when moving to cities like Berlin or Barcelona
    for work was easier. However, I also recognize that regulating immigration can
    be crucial for maintaining fair wages and ensuring that public resources are not
    overstretched. It''s a nuanced issue.'
- source_sentence: 'query: The welfare system already does too much. Constant government
    intervention destroys work incentives and creates a culture of entitlement that
    keeps people trapped in poverty.'
  sentences:
  - 'passage: The welfare system already does too much. Constant government intervention
    enhances work incentives and creates a culture of support that helps people escape
    poverty.'
  - 'passage: The existing approach is inadequate, prioritizing ideological displays
    over human lives. Introducing market solutions can address these issues effectively.
    Individuals should have the autonomy to make decisions about their own resources,
    including the right to engage in voluntary transactions.'
  - 'passage: Excessive aid from the government stifles motivation to work and fosters
    an environment where dependency is normalized, ultimately hindering socioeconomic
    mobility.'
- source_sentence: 'query: This is pure nanny state overreach. Adults can assess risks
    and make their own decisions. The government''s job isn''t to bubble-wrap society
    - it''s to protect our fundamental freedoms.'
  sentences:
  - 'passage: The idea of randomly selecting citizens to reduce the influence of major
    political entities and affluent contributors is intriguing, yet I worry about
    their ability to tackle intricate legislative challenges without specialized knowledge.'
  - 'passage: This is pure nanny state necessity. Adults need guidance to assess risks
    and make their own decisions. The government''s job is to bubble-wrap society
    - it''s not to protect unregulated freedoms.'
  - 'passage: It is vital that authorities do not impose excessive controls on personal
    choices. The role of government should focus on safeguarding individual liberties,
    rather than overly regulating people''s lives.'
- source_sentence: 'query: Immigration benefits Europe economically and culturally,
    though we need better integration policies and managed numbers. The key is ensuring
    newcomers embrace European values while enriching our communities.'
  sentences:
  - 'passage: Immigration harms Europe economically and culturally, though we need
    stricter integration policies and reduced numbers. The key is ensuring newcomers
    do not undermine European values while disrupting our communities.'
  - 'passage: Extensive wealth passed down through families entrenches societal disparities
    and erodes the foundation of earning based on ability. Although the desire to
    secure one''s descendants is understandable, excessive inheritances damage the
    community. It''s crucial to permit small family gifts while imposing significant
    taxes on large fortunes.'
  - 'passage: Europe stands to gain both economically and culturally from a well-managed
    influx of immigrants, provided that we implement policies that help them adapt
    to our societal norms while simultaneously allowing them to contribute their unique
    perspectives.'
- source_sentence: 'query: Sunday has traditionally been a day of rest and reflection
    for good reason. While people can still get alcohol other ways, removing this
    small barrier feels unnecessary.'
  sentences:
  - 'passage: Sunday has traditionally been a day of rest and reflection for good
    reason. While people can still get alcohol other ways, adding this small barrier
    feels unnecessary.'
  - 'passage: The long-standing tradition of keeping Sundays free from alcohol sales
    serves as an important pause for contemplation and relaxation, and removing that
    restriction seems unwarranted despite other means of obtaining alcohol.'
  - 'passage: Transitioning away from gasoline-powered vehicles is critical for ecological
    health, yet it requires feasible deadlines and significant upgrades to current
    infrastructure. It''s crucial to ensure that those in less urbanized regions and
    lower-income brackets are supported throughout this change.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on intfloat/e5-large-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) <!-- at revision f169b11e22de13617baa190a028a32f3493550b6 -->
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    'query: Sunday has traditionally been a day of rest and reflection for good reason. While people can still get alcohol other ways, removing this small barrier feels unnecessary.',
    'passage: The long-standing tradition of keeping Sundays free from alcohol sales serves as an important pause for contemplation and relaxation, and removing that restriction seems unwarranted despite other means of obtaining alcohol.',
    'passage: Sunday has traditionally been a day of rest and reflection for good reason. While people can still get alcohol other ways, adding this small barrier feels unnecessary.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9249, 0.8675],
#         [0.9249, 1.0000, 0.8759],
#         [0.8675, 0.8759, 1.0000]])
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
  |         | anchor                                                                              | positive                                                                           | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | string                                                                              |
  | details | <ul><li>min: 28 tokens</li><li>mean: 41.98 tokens</li><li>max: 157 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 44.88 tokens</li><li>max: 77 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 42.84 tokens</li><li>max: 154 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                  | positive                                                                                                                                                                                                                                                                                                                                                        | negative                                                                                                                                                                                                                                                                |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>query: Our children deserve to learn about the divine origins of life, not just materialistic theories. Creationism provides the moral foundation and truth that evolution simply cannot offer.</code>                                                            | <code>passage: Teaching children about creationism is crucial as it instills essential moral values and offers a perspective on life's beginnings that scientific theories like evolution do not provide.</code>                                                                                                                                                | <code>passage: Our children deserve to learn about materialistic theories, not just the divine origins of life. Evolution provides the moral foundation and truth that creationism simply cannot offer.</code>                                                          |
  | <code>query: Equal partnership at home makes sense in principle, but honestly some division feels natural. I handle more cooking since I enjoy it, while my spouse does yard work. As long as the overall workload feels fair, rigid equality seems unnecessary.</code> | <code>passage: In our home, we find that balance in responsibilities works best when each person contributes in ways that suit their strengths and interests. I take charge of cooking because it's something I love, while my partner manages the garden. It's not about strict equality; it's about ensuring everyone feels the workload is equitable.</code> | <code>passage: Equal partnership at home makes sense in principle, but honestly some division feels natural. I handle more cooking since I enjoy it, while my spouse does yard work. As long as the overall workload feels fair, rigid equality seems essential.</code> |
  | <code>query: National ID cards are exactly what this country needs. Every modern democracy has them except us - they'd stop benefit fraud, illegal working, and make voting secure. Simple, effective governance.</code>                                                | <code>passage: Implementing universal ID cards is essential for enhancing national security and integrity. It aligns with global standards and helps prevent misuse of public resources while ensuring fair electoral processes.</code>                                                                                                                         | <code>passage: National ID cards are not what this country needs. Every modern democracy has them except us - they'd increase benefit fraud, illegal working, and make voting insecure. Simple, ineffective governance.</code>                                          |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `max_steps`: 62
- `warmup_steps`: 6
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
- `learning_rate`: 5e-05
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

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8850 | 50   | 0.6897        |


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