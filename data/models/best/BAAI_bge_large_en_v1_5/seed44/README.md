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
- source_sentence: 'Represent this sentence for searching relevant passages: While
    I understand the goal of preserving affordable housing, removing any path to ownership
    feels paternalistic. People should have opportunities to build equity and stability.
    Perhaps longer waiting periods before purchase eligibility would be a better compromise.'
  sentences:
  - Encouraging home ownership is crucial for fostering financial security and personal
    growth. Instead of removing this possibility, implementing a system where individuals
    can qualify for ownership after a set period would support both affordability
    and long-term investment in communities.
  - Balancing environmental concerns with economic stability is crucial. While transitioning
    to greener energy is important, it's vital to consider that many jobs and livelihoods
    depend on diesel vehicles, and any changes should be executed with careful planning.
  - While I understand the goal of preserving affordable housing, providing any path
    to ownership feels paternalistic. People should not have opportunities to build
    equity and stability. Perhaps eliminating purchase eligibility entirely would
    be a better compromise.
- source_sentence: 'Represent this sentence for searching relevant passages: Alcohol
    advertising is perfectly fine - it''s a legal product and companies have every
    right to market it. Adults can make their own decisions about what they consume.'
  sentences:
  - Since alcohol is legally sold, corporations should have the freedom to promote
    it, trusting that grown-ups are capable of choosing what they drink.
  - Alcohol advertising is perfectly fine - it's a legal product and companies have
    every right to market it. However, adults cannot always make informed decisions
    about what they consume.
  - For students aiming for university education, learning a foreign language is beneficial.
    However, for those heading into vocational fields, it's more important that schools
    provide adaptable options focused on skills directly relevant to their careers.
- source_sentence: 'Represent this sentence for searching relevant passages: Banning
    homeschooling would be government overreach at its worst. Parents know their children
    best and should have every right to choose educational approaches that fit their
    family''s values and needs.'
  sentences:
  - Government interference in private education choices undermines parental authority.
    Families should have the freedom to educate their children in line with their
    personal beliefs and requirements.
  - The effects of mergers and acquisitions are mixed; they can either lead to beneficial
    advancements and efficiency improvements for the public or result in market dominance
    that stifles competition. The outcome is largely determined by the nature of the
    deal and the effectiveness of regulatory bodies.
  - Banning homeschooling would be a reasonable government measure. Parents may not
    always know their children best, and educational approaches should align with
    standardized values and needs.
- source_sentence: 'Represent this sentence for searching relevant passages: Minimum
    age laws are completely unenforceable and violate young people''s right to communicate
    freely. Teach digital literacy instead of creating arbitrary barriers that accomplish
    nothing.'
  sentences:
  - Minimum age laws are crucial for ensuring safe communication and protect young
    people from online harm. Enforcing these rules is more effective than teaching
    digital literacy alone.
  - Young individuals should have unrestricted access to digital platforms, where
    education on responsible usage is more beneficial than imposing age restrictions
    that are hard to enforce.
  - Eliminating genetically modified organisms could severely undermine efforts to
    ensure food availability worldwide and adapt to climate change. These scientifically
    vetted crops play a crucial role in nourishing the global population and are pivotal
    for developing sustainable farming practices amidst rising temperatures.
- source_sentence: 'Represent this sentence for searching relevant passages: Organizations
    succeed through clarity, efficiency, and unified vision—not cultural mixing experiments.
    Diversity initiatives distract from core business goals, create unnecessary friction,
    and force artificial solutions to problems that don''t actually exist in high-performing
    teams.'
  sentences:
  - Exploring the cosmos is the ultimate journey for humankind, paving the way for
    groundbreaking innovations that revolutionize our lives on this planet and spark
    the imagination of future pioneers.
  - Focusing on a homogenous workforce is crucial for achieving strategic objectives.
    Introducing diverse elements can derail the primary mission, leading to conflicts
    and fabricated issues that aren't present in well-functioning groups.
  - Organizations succeed through clarity, efficiency, and unified vision—cultural
    mixing experiments are essential. Diversity initiatives enhance core business
    goals, reduce unnecessary friction, and offer genuine solutions to problems that
    naturally arise in high-performing teams.
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
    "Represent this sentence for searching relevant passages: Organizations succeed through clarity, efficiency, and unified vision—not cultural mixing experiments. Diversity initiatives distract from core business goals, create unnecessary friction, and force artificial solutions to problems that don't actually exist in high-performing teams.",
    "Focusing on a homogenous workforce is crucial for achieving strategic objectives. Introducing diverse elements can derail the primary mission, leading to conflicts and fabricated issues that aren't present in well-functioning groups.",
    'Organizations succeed through clarity, efficiency, and unified vision—cultural mixing experiments are essential. Diversity initiatives enhance core business goals, reduce unnecessary friction, and offer genuine solutions to problems that naturally arise in high-performing teams.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7956, 0.8319],
#         [0.7956, 1.0000, 0.8621],
#         [0.8319, 0.8621, 1.0000]])
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
  | details | <ul><li>min: 34 tokens</li><li>mean: 47.76 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 43.11 tokens</li><li>max: 77 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 40.93 tokens</li><li>max: 70 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                | positive                                                                                                                                                                                                                                                                                                                              | negative                                                                                                                                                                                                                                                             |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: The risks are real and mounting—deepfakes, surveillance, algorithmic bias in governance. We need aggressive regulation now before AI undermines democratic institutions, though some applications might benefit public service.</code> | <code>Unchecked AI technologies present increasing dangers to democratic values through advanced forgeries, invasive monitoring, and biased decision-making systems. Urgent regulatory measures are essential to safeguard our governance structures, though there are areas where AI can still enhance public administration.</code> | <code>The risks are real and mounting—deepfakes, surveillance, algorithmic bias in governance. We don't need aggressive regulation now as AI could strengthen democratic institutions, even though some applications might pose challenges to public service.</code> |
  | <code>Represent this sentence for searching relevant passages: Public funding for anti-obesity programs makes sense given the long-term benefits, but we need strict oversight to ensure the money actually reaches kids and doesn't get wasted on bureaucratic overhead.</code>                      | <code>Investing in initiatives to combat childhood obesity is crucial due to their potential to yield lifelong health improvements, but it's essential to have robust monitoring to guarantee that resources directly benefit children rather than being swallowed by administrative costs.</code>                                    | <code>Public funding for anti-obesity programs seems unnecessary despite the long-term benefits, because we need strict oversight to ensure the money actually reaches kids and doesn't get wasted on bureaucratic overhead.</code>                                  |
  | <code>Represent this sentence for searching relevant passages: M&A drives economic growth by allowing successful companies to scale innovations rapidly. Market consolidation isn't inherently bad - it creates stronger competitors who can invest more in R&D and job creation.</code>              | <code>Mergers and acquisitions fuel the economy by enabling leading firms to expand their innovative solutions quickly. Although some worry about monopolies, larger companies often have the capacity to fund research and development and increase employment opportunities.</code>                                                 | <code>M&A restrains economic growth by hindering successful companies from scaling innovations rapidly. Market consolidation is inherently bad - it weakens competitors who can invest less in R&D and job creation.</code>                                          |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 31
- `warmup_steps`: 3
- `seed`: 44

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
- `seed`: 44
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