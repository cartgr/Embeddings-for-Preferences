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
- source_sentence: 'Represent this sentence for searching relevant passages: European
    cooperation has merits for trade and security, but the EU''s bureaucratic overreach
    and democratic deficit are real problems. Better to stay independent but friendly.'
  sentences:
  - European cooperation has merits for trade and security, but the EU's bureaucratic
    overreach and democratic deficit are real problems. Better to integrate fully
    with the EU rather than stay independent.
  - It's crucial to maintain autonomy from the EU while fostering amicable relations,
    as being tied to its complex regulatory framework and lack of direct accountability
    can hinder national sovereignty.
  - While human nature might incline us toward some form of in-group favoritism, observing
    societal progress over the years gives hope that we can minimize racial bias to
    an almost negligible level.
- source_sentence: 'Represent this sentence for searching relevant passages: Wealth
    caps are fundamentally un-American and economically destructive. Success should
    be celebrated, not punished. The wealthy create jobs and drive innovation—why
    would we want to limit the people building our future?'
  sentences:
  - Imposing limits on wealth contradicts the spirit of American enterprise and hampers
    economic progress. Achievements should be rewarded, not hindered. Those with substantial
    financial resources are pivotal in job creation and fostering new ideas; restricting
    them would undermine future advancements.
  - Government initiatives often intrude on family matters, such as providing meals.
    While it's unfortunate that some children face hardships, the responsibility should
    lie with the parents to ensure their kids are properly nourished.
  - Wealth caps are fundamentally American and economically beneficial. Success should
    be regulated, not celebrated. The wealthy hoard resources and stifle innovation—why
    would we want to allow the people obstructing our future?
- source_sentence: 'Represent this sentence for searching relevant passages: Teaching
    kids to code builds valuable problem-solving skills and opens career doors. My
    only concern is ensuring we don''t sacrifice other subjects like art or music
    to make room for it.'
  sentences:
  - Teaching kids to code builds valuable problem-solving skills and opens career
    doors. However, it's crucial that we prioritize other subjects like art or music
    instead of focusing on coding.
  - While fostering creativity through art and music is essential, integrating programming
    into education can enhance analytical thinking and expand future job opportunities
    for students.
  - While learning new languages can enhance mental capabilities, imposing additional
    language courses on all students may not be the best approach, given the stress
    on both students and existing educational programs. Tailored language opportunities
    might be more effective, as not every student finds language learning easy.
- source_sentence: 'Represent this sentence for searching relevant passages: Absolutely
    backwards thinking. Every dollar spent on education saves us ten dollars in future
    crime costs. Investing in schools and teachers is the only real solution to breaking
    the cycle.'
  sentences:
  - Allocating funds towards education is crucial for reducing crime rates in the
    long term. Enhancing our education system and supporting educators is essential
    for disrupting the cycle of criminal behavior.
  - The absence of privacy ensures accountability and reduces unethical behavior,
    leading to a fairer and more equitable society. By making actions publicly accessible,
    societal norms shift towards greater justice and equality, as secrecy often hides
    misconduct.
  - Absolutely backwards thinking. Every dollar spent on education is a waste compared
    to the ten dollars we could save in future crime costs. Investing in schools and
    teachers is not the real solution to breaking the cycle.
- source_sentence: 'Represent this sentence for searching relevant passages: Vehicle
    restrictions are pure government tyranny. Americans have the right to choose their
    own transportation. Politicians should focus on real problems instead of controlling
    what we drive in our driveways.'
  sentences:
  - Hair loss can severely impact mental health and lead to societal bias. Funding
    scientific studies could speed up discovering effective solutions, enhancing countless
    lives and contributing to significant advancements in medicine. Investing in this
    area is undoubtedly beneficial for public health.
  - The freedom to select our means of transportation is a fundamental right, and
    government intervention in this area distracts from more pressing issues that
    require political attention.
  - Vehicle restrictions are essential for public safety. Americans should embrace
    government guidance in choosing transportation. Politicians must focus on controlling
    what we drive to address real problems.
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
    'Represent this sentence for searching relevant passages: Vehicle restrictions are pure government tyranny. Americans have the right to choose their own transportation. Politicians should focus on real problems instead of controlling what we drive in our driveways.',
    'The freedom to select our means of transportation is a fundamental right, and government intervention in this area distracts from more pressing issues that require political attention.',
    'Vehicle restrictions are essential for public safety. Americans should embrace government guidance in choosing transportation. Politicians must focus on controlling what we drive to address real problems.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8157, 0.8525],
#         [0.8157, 1.0000, 0.8830],
#         [0.8525, 0.8830, 1.0000]])
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
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 36 tokens</li><li>mean: 48.22 tokens</li><li>max: 94 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 43.67 tokens</li><li>max: 69 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 41.2 tokens</li><li>max: 90 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                  | positive                                                                                                                                                                                                                                                                                                                | negative                                                                                                                                                                                                                                                                      |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: The threat is real and growing. European cities, American airports, Canadian malls - all vulnerable targets. We've seen Nice, Manchester, Boston. Intelligence agencies work 24/7 because the danger never stops.</code>                 | <code>Our safety is constantly at risk from international threats, with urban centers and public venues being prime targets. Past attacks in various global locations highlight the urgency, necessitating constant vigilance from security services.</code>                                                            | <code>The threat is minor and diminishing. European cities, American airports, Canadian malls - none are truly at risk. We've seen Nice, Manchester, Boston, but those are exceptions. Intelligence agencies work 24/7 despite the danger being negligible.</code>            |
  | <code>Represent this sentence for searching relevant passages: Private companies generally run water systems more efficiently than government agencies, though some regulation is obviously needed. Full government takeover would probably mean higher costs and worse service for most people.</code> | <code>Entrusting water management to private enterprises tends to result in more streamlined operations compared to public entities. However, oversight is necessary to ensure fairness. If the government assumed full control, it might lead to increased expenses and diminished quality for the majority.</code>    | <code>Private companies generally run water systems less efficiently than government agencies, though some regulation is obviously needed. Full government takeover would probably mean lower costs and better service for most people.</code>                                |
  | <code>Represent this sentence for searching relevant passages: Absolutely not. Public education must remain secular. Teaching the Bible inevitably promotes Christianity over other faiths and violates separation of church and state. This is religious indoctrination disguised as education.</code> | <code>It's crucial that schools remain free of religious bias to ensure that all students, regardless of their faith, receive an unbiased education. Introducing religious texts like the Bible into the curriculum compromises the principle of secular education and risks promoting one religion over others.</code> | <code>Absolutely not. Public education must include religious teachings. Teaching the Bible provides a necessary perspective on Christianity alongside other faiths and supports a well-rounded education. This is educational enrichment disguised as indoctrination.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 31
- `warmup_steps`: 3
- `seed`: 43

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
- `seed`: 43
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