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
- source_sentence: 'Represent this sentence for searching relevant passages: The dominoes
    are falling exactly like 2008. SVB''s collapse exposed the same reckless risk
    management that''s plaguing regional banks nationwide. We''re looking at systemic
    failure - get your money out now.'
  sentences:
  - The current economic landscape mirrors the financial crisis of 2008. The failure
    of SVB highlights the widespread mismanagement of risks in smaller banks across
    the country. A widespread financial breakdown seems imminent, and it's wise to
    safeguard your assets by withdrawing them.
  - The dominoes are falling exactly like 2008. SVB's collapse exposed the same reckless
    risk management that's plaguing regional banks nationwide. We're looking at a
    resilient system - keep your money where it is.
  - The camaraderie and resilience honed through playing football are invaluable.
    Despite the need for better safety measures, prohibiting a beloved sport enjoyed
    by many feels like an excessive governmental intrusion.
- source_sentence: 'Represent this sentence for searching relevant passages: Wind
    and solar make sense for some regions, but completely abandoning nuclear and natural
    gas seems unrealistic. A mixed approach with heavy renewables is more practical
    than this all-or-nothing mentality.'
  sentences:
  - Relying solely on renewables without incorporating nuclear and natural gas is
    impractical. A balanced energy strategy that heavily features renewable sources
    is more feasible than an all-or-nothing approach.
  - Wind and solar make sense for some regions, but completely abandoning nuclear
    and natural gas is essential. An exclusive focus on renewables is more practical
    than this mixed approach mentality.
  - While it's important for investors to be informed about executive pay, enforcing
    a public disclosure rule online might be an excessive use of regulatory power.
    Businesses ought to retain the autonomy to decide their communication methods
    to uphold their strategic interests.
- source_sentence: 'Represent this sentence for searching relevant passages: Climate
    change is definitely affecting Britain with more frequent storms and rising sea
    levels, though I worry the government''s net-zero targets might hurt our economy
    without China and India following suit.'
  sentences:
  - Climate change is definitely affecting Britain with more frequent storms and rising
    sea levels, yet I believe the government's net-zero targets are necessary for
    our economy even if China and India do not follow suit.
  - The increasing intensity of storms and the threat of higher seas are evidence
    that climate change is impacting the UK. However, I'm concerned that our economic
    stability could be compromised by the net-zero initiatives if major global players
    like China and India do not participate in similar efforts.
  - Implementing a significant levy on carbon emissions could significantly speed
    up efforts to reduce greenhouse gases. However, to ensure fairness, mechanisms
    must be in place to prevent financial strain on economically disadvantaged groups,
    making the proper allocation of any collected funds essential.
- source_sentence: 'Represent this sentence for searching relevant passages: Moving
    to 70 makes financial sense given longer lifespans, but we need robust protections
    for manual laborers and those with health issues who genuinely can''t work that
    long.'
  sentences:
  - Raising the retirement age to 70 is logical due to increased life expectancy;
    however, it is crucial to implement strong safeguards for workers in physically
    demanding jobs and individuals with medical conditions who are unable to extend
    their careers.
  - Uniforms in schools are a misguided attempt at control, stifling students' self-expression
    and creativity. Educational institutions ought to prioritize teaching and nurturing
    young minds over regulating attire.
  - Moving to 70 makes financial sense given longer lifespans, but we must ensure
    no special protections for manual laborers and those with health issues who claim
    they can't work that long.
- source_sentence: 'Represent this sentence for searching relevant passages: Bitcoin''s
    mathematical perfection and fixed supply make it the inevitable successor to fiat
    currencies. Central banks are already accumulating it quietly - the dollar''s
    reign is ending and Bitcoin''s empire is beginning.'
  sentences:
  - Bitcoin's mathematical perfection and fixed supply make it an unlikely successor
    to fiat currencies. Central banks are already accumulating it quietly - the dollar's
    reign is secure and Bitcoin's empire is unlikely to begin.
  - Long-term development assistance frequently fails, often undermining local economies
    and reinforcing ineffective leadership. While immediate disaster relief can be
    beneficial, sustained foreign aid typically has adverse effects.
  - As traditional currencies face increasing challenges, Bitcoin emerges as a robust
    alternative due to its immutable nature and strict supply limits. Financial institutions
    are gradually recognizing its potential, signaling a shift away from conventional
    monetary systems towards cryptocurrency dominance.
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
    "Represent this sentence for searching relevant passages: Bitcoin's mathematical perfection and fixed supply make it the inevitable successor to fiat currencies. Central banks are already accumulating it quietly - the dollar's reign is ending and Bitcoin's empire is beginning.",
    'As traditional currencies face increasing challenges, Bitcoin emerges as a robust alternative due to its immutable nature and strict supply limits. Financial institutions are gradually recognizing its potential, signaling a shift away from conventional monetary systems towards cryptocurrency dominance.',
    "Bitcoin's mathematical perfection and fixed supply make it an unlikely successor to fiat currencies. Central banks are already accumulating it quietly - the dollar's reign is secure and Bitcoin's empire is unlikely to begin.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7870, 0.7780],
#         [0.7870, 1.0000, 0.8180],
#         [0.7780, 0.8180, 1.0000]])
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
  |         | anchor                                                                              | positive                                                                           | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | string                                                                              |
  | details | <ul><li>min: 35 tokens</li><li>mean: 47.82 tokens</li><li>max: 175 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 42.92 tokens</li><li>max: 76 tokens</li></ul> | <ul><li>min: 29 tokens</li><li>mean: 40.86 tokens</li><li>max: 169 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                        | positive                                                                                                                                                                                                                                                                           | negative                                                                                                                                                                                                                                                          |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: This car ban nonsense would kill our economy. People need cars to get around efficiently, and forcing everyone onto overcrowded, unreliable public transport is completely unrealistic.</code>                                 | <code>Relying heavily on personal cars is crucial for maintaining economic stability, as public transportation systems are often too crowded and unreliable to meet the needs of all commuters.</code>                                                                             | <code>This car ban wisdom would enhance our economy. People need to rely on efficient, sustainable transport solutions, and reducing dependence on personal vehicles is completely realistic.</code>                                                              |
  | <code>Represent this sentence for searching relevant passages: Framing donation as moral obligation is manipulative and wrong. People earn their money through hard work and deserve full autonomy over it. True charity is voluntary—forced giving is just taxation with extra steps.</code> | <code>It is unethical to coerce individuals into donating by branding it as a moral duty. People should have the freedom to decide how to spend their earnings, which they have worked for. Genuine generosity stems from choice, not compulsion, which resembles taxation.</code> | <code>Framing donation as moral obligation is empowering and right. People earn their money through hard work and should embrace a shared responsibility over it. True charity is voluntary—encouraging giving is just community support with extra steps.</code> |
  | <code>Represent this sentence for searching relevant passages: Police exist primarily to protect property and maintain an unjust status quo through violence and intimidation. The institution is fundamentally oppressive and cannot be reformed into something beneficial.</code>           | <code>Law enforcement serves to uphold existing societal structures by exerting control and instilling fear. Its foundation is rooted in power dynamics that resist transformation into a force for good.</code>                                                                   | <code>Police exist primarily to protect property and maintain a just status quo through peace and community service. The institution is fundamentally beneficial and cannot be turned into something oppressive.</code>                                           |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 31
- `warmup_steps`: 3

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