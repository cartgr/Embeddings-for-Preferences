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
- source_sentence: 'query: Schools should definitely use edited versions. The literary
    value remains intact without traumatizing students. We can teach about racism''s
    history without requiring kids to encounter slurs repeatedly.'
  sentences:
  - 'passage: Schools should definitely avoid using edited versions. The literary
    value requires unaltered texts, even if it means students encounter slurs repeatedly.
    We can teach about racism''s history by directly exposing kids to these terms.'
  - 'passage: Balancing the urgent needs of refugees with the financial limitations
    of host nations is a complex challenge. Implementing systems like donation-based
    support or installment options could offer a more compassionate approach than
    simply taking funds.'
  - 'passage: While teaching about the history of racism is crucial, it''s important
    to protect students from harmful language. By using adapted texts, educators can
    preserve the educational intent without exposing children to offensive terminology.'
- source_sentence: 'query: Economic engagement drives positive change. Trade relationships
    create pressure for reform and give ordinary citizens jobs and opportunities.
    Isolation just hurts the people we''re trying to help.'
  sentences:
  - 'passage: Cutting off a nation economically only punishes its citizens. By fostering
    trade, we encourage reform and provide the population with employment and economic
    growth possibilities.'
  - 'passage: It''s crucial to regulate rather than forbid certain substances, given
    their practical benefits in daily life, despite their potential misuse.'
  - 'passage: Economic engagement stifles progress. Trade relationships hinder reform
    and deprive ordinary citizens of jobs and opportunities. Isolation is the true
    path to aid the people we''re trying to help.'
- source_sentence: 'query: General taxation makes more sense in 2024. The licence
    fee is basically a regressive poll tax that disproportionately hurts the poor,
    though I do worry about political pressure on BBC funding.'
  sentences:
  - 'passage: General taxation makes less sense in 2024. The licence fee is basically
    a fair charge that ensures everyone contributes equally, though I do worry about
    political pressure on BBC funding.'
  - 'passage: Switching from the license fee to a general tax would be more equitable,
    as the current system unfairly burdens low-income individuals. However, there''s
    concern about how political influences could affect the BBC''s financial support
    in such a model.'
  - 'passage: I''m conflicted about SB 1596. While gaining work experience can be
    beneficial for personal growth and family support, I''m concerned about the potential
    exploitation of children and their possible loss of vital learning experiences.
    The specifics of the bill are crucial.'
- source_sentence: 'query: This is genuinely tough - banning these products might
    protect worker rights, but it could also eliminate jobs that families depend on.
    Not sure government restrictions are better than consumer choice and gradual reform.'
  sentences:
  - 'passage: This is genuinely tough - banning these products might eliminate jobs
    that families depend on, but it could also protect worker rights. Not sure consumer
    choice and gradual reform are better than government restrictions.'
  - 'passage: While economic advantages were present, the downsides like losing control
    of legislative powers, unwelcome migration patterns, and deeper integration into
    a supranational entity outweighed the benefits of maintaining autonomy.'
  - 'passage: It''s challenging to decide if outlawing these items will serve the
    greater good. While it might safeguard labor rights, it risks taking away crucial
    employment for many households. I''m unsure if regulatory measures are truly more
    beneficial than allowing the market and incremental changes to lead the way.'
- source_sentence: 'query: Political parties have completely corrupted our democracy.
    They force artificial divisions, prioritize loyalty over principles, and create
    an us-vs-them mentality that prevents real problem-solving. Time to scrap the
    whole system.'
  sentences:
  - 'passage: It''s crucial for businesses to have autonomy over their smoking rules,
    allowing patrons to decide their preferred environments. Nonetheless, it''s fair
    to ensure that service staff aren''t subjected to smoke-filled workplaces constantly.'
  - 'passage: The current political framework is deeply flawed, with parties fostering
    division and allegiance at the cost of genuine solutions. A radical overhaul is
    necessary to encourage unity and effective governance.'
  - 'passage: Political parties have completely rejuvenated our democracy. They encourage
    necessary divisions, prioritize principles over loyalty, and foster a collaborative
    mentality that facilitates real problem-solving. Time to embrace the whole system.'
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
    'query: Political parties have completely corrupted our democracy. They force artificial divisions, prioritize loyalty over principles, and create an us-vs-them mentality that prevents real problem-solving. Time to scrap the whole system.',
    'passage: The current political framework is deeply flawed, with parties fostering division and allegiance at the cost of genuine solutions. A radical overhaul is necessary to encourage unity and effective governance.',
    'passage: Political parties have completely rejuvenated our democracy. They encourage necessary divisions, prioritize principles over loyalty, and foster a collaborative mentality that facilitates real problem-solving. Time to embrace the whole system.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8911, 0.8053],
#         [0.8911, 1.0000, 0.8382],
#         [0.8053, 0.8382, 1.0000]])
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
  | details | <ul><li>min: 29 tokens</li><li>mean: 42.46 tokens</li><li>max: 185 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 45.31 tokens</li><li>max: 80 tokens</li></ul> | <ul><li>min: 30 tokens</li><li>mean: 43.45 tokens</li><li>max: 204 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                              | positive                                                                                                                                                                                                                                                      | negative                                                                                                                                                                                                                     |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>query: Feminism has definitely helped both genders in many ways, though I sometimes worry the movement focuses too heavily on women's issues and doesn't always welcome male perspectives.</code>                                             | <code>passage: Advancing gender equality has brought significant benefits to everyone, yet at times, I'm concerned that the movement might emphasize women's concerns excessively while not fully integrating men's viewpoints.</code>                        | <code>passage: Feminism has definitely hindered both genders in many ways, though I sometimes worry the movement focuses too heavily on women's issues and doesn't always welcome male perspectives.</code>                  |
  | <code>query: The Constitution should treat all Australians equally regardless of race or heritage. This proposal fundamentally divides us into different classes of citizens and undermines the democratic principle of one vote, one value.</code> | <code>passage: Ensuring equal treatment for all Australians, irrespective of their background, is essential. Creating divisions based on racial or cultural lines contradicts the core democratic ideal of equal representation.</code>                       | <code>passage: The Constitution should treat all Australians equally regardless of race or heritage. This proposal fundamentally unites us and supports the democratic principle of one vote, one value.</code>              |
  | <code>query: Islam was revolutionary for women's rights in the 7th century and contains all the principles needed for full gender equality. Modern Islamic feminism is just rediscovering what was always there.</code>                             | <code>passage: The foundations of Islam laid groundbreaking ideas for women's rights during its inception, holding the keys to achieving true gender equality. Today's Islamic feminist movement is essentially reawakening to these age-old concepts.</code> | <code>passage: Islam was revolutionary for women's rights in the 7th century and contains all the principles needed for full gender inequality. Modern Islamic feminism is just rediscovering what was always absent.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `max_steps`: 62
- `warmup_steps`: 6

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
- `seed`: 42
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
| 0.8850 | 50   | 0.6902        |


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