---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:675
- loss:BradleyTerryLoss
base_model: sentence-transformers/sentence-t5-xl
widget:
- source_sentence: Wind and solar are expensive fantasies pushed by elites who've
    never worked a real job. Oil and gas are reliable, affordable, and have lifted
    billions out of poverty.
  sentences:
  - Traditional energy sources like oil and gas provide a stable and cost-effective
    solution that has significantly contributed to economic development and improved
    living standards for many across the globe.
  - The legitimacy of our electoral process is questionable due to unexpected voting
    patterns and modifications in voting procedures, particularly in key states. These
    issues warrant thorough examination, even if concrete evidence of large-scale
    fraud is elusive.
  - Wind and solar are reliable, affordable, and have lifted billions out of poverty.
    Oil and gas are expensive fantasies pushed by elites who've never worked a real
    job.
- source_sentence: Immigration benefits our economy and culture, though we need better
    coordination between agencies and more resources for border processing. The framework
    is sound but execution needs improvement.
  sentences:
  - Enhancing the economic and cultural landscape requires immigration, but the current
    system needs improved inter-agency cooperation and increased funding for border
    operations to be truly effective.
  - Immigration harms our economy and culture, though we need better coordination
    between agencies and more resources for border processing. The framework is flawed
    but execution needs improvement.
  - To safeguard our health and environment, it's crucial to halt the import of produce
    from nations that rely on harmful agricultural chemicals.
- source_sentence: We need robust outcome-focused policies to level the playing field,
    though I worry about potential innovation costs. Sometimes correcting historical
    inequities requires aggressive intervention in results.
  sentences:
  - Raising children within a community setting fosters healing and growth by providing
    them with various role models and a strong support system, essential for overcoming
    generational trauma.
  - We need robust outcome-focused policies to level the playing field, yet I believe
    innovation costs are too high to justify such measures. Sometimes correcting historical
    inequities requires avoiding aggressive intervention in results.
  - To address past injustices, we must implement forceful measures that focus on
    equalizing opportunities, even if it means risking some progress on innovation.
- source_sentence: Trickle-down economics is the foundation of prosperity. When businesses
    and job creators have more capital, they expand operations, hire workers, and
    drive innovation. Every major economic boom in history started with investment
    flowing from the top down.
  sentences:
  - To ensure meaningful engagement in civic duties, it's crucial to incorporate long-standing
    community members and young individuals nearing adulthood. However, implementing
    criteria such as a minimum residency period for immigrants and educational assessments
    for teenagers can help maintain informed decision-making.
  - Economic growth is driven by empowering businesses to invest more in their operations.
    When companies have the resources to expand, it leads to job creation and technological
    advancements, laying the groundwork for widespread prosperity.
  - Trickle-down economics is the root of inequality. When businesses and job creators
    have more capital, they hoard profits, reduce workforce, and stifle innovation.
    Every major economic crisis in history started with investment being hoarded at
    the top.
- source_sentence: Strategic sectors definitely need tighter controls, especially
    from non-allied nations. But we shouldn't throw the baby out with the bathwater
    - legitimate investment still creates jobs and brings expertise we need.
  sentences:
  - It's crucial to regulate critical industries more strictly, particularly those
    involving foreign entities with differing agendas. Nonetheless, it's important
    to recognize that foreign investments can drive employment and introduce valuable
    skills to our economy.
  - Strategic sectors definitely need looser controls, especially from non-allied
    nations. But we should throw the baby out with the bathwater - legitimate investment
    doesn't create jobs and brings expertise we don't need.
  - The future is uncertain. While the current challenges are significant, history
    shows resilience in overcoming even greater obstacles. Ultimately, much hinges
    on leadership's approach to addressing environmental issues and social disparity.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/sentence-t5-xl

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/sentence-t5-xl](https://huggingface.co/sentence-transformers/sentence-t5-xl). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/sentence-t5-xl](https://huggingface.co/sentence-transformers/sentence-t5-xl) <!-- at revision 92e07434e0b0e93b36dd780c47c9b48d32fbcdaa -->
- **Maximum Sequence Length:** 256 tokens
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 1024, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
  (3): Normalize()
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
    "Strategic sectors definitely need tighter controls, especially from non-allied nations. But we shouldn't throw the baby out with the bathwater - legitimate investment still creates jobs and brings expertise we need.",
    "It's crucial to regulate critical industries more strictly, particularly those involving foreign entities with differing agendas. Nonetheless, it's important to recognize that foreign investments can drive employment and introduce valuable skills to our economy.",
    "Strategic sectors definitely need looser controls, especially from non-allied nations. But we should throw the baby out with the bathwater - legitimate investment doesn't create jobs and brings expertise we don't need.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8949, 0.3436],
#         [0.8949, 1.0000, 0.3346],
#         [0.3436, 0.3346, 1.0000]])
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

* Size: 675 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 675 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 25 tokens</li><li>mean: 41.51 tokens</li><li>max: 87 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 45.42 tokens</li><li>max: 78 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 42.61 tokens</li><li>max: 87 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                              | positive                                                                                                                                                                                                                                                     | negative                                                                                                                                                                                                                                  |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Hard to say honestly. The system faces serious pressures but has survived worse crises before. Could go either way - depends on how leaders respond to climate change and inequality.</code>                                  | <code>The future is uncertain. While the current challenges are significant, history shows resilience in overcoming even greater obstacles. Ultimately, much hinges on leadership's approach to addressing environmental issues and social disparity.</code> | <code>Hard to say honestly. The system faces serious pressures but has not survived worse crises before. Could go either way - depends on how leaders respond to climate change and inequality, but it's likely to fail.</code>           |
  | <code>Moving fully cashless seems premature given how many people still rely on cash daily. Sure, digital payments are convenient, but we'd be creating barriers for vulnerable populations who aren't well-served by banks.</code> | <code>Transitioning to a completely digital payment system is premature because many individuals depend on cash for their everyday transactions. While online payments offer ease, it could marginalize groups who lack adequate banking services.</code>    | <code>Moving fully cashless is necessary despite how many people still rely on cash daily. Sure, digital payments are convenient, and we wouldn't be creating barriers for vulnerable populations who aren't well-served by banks.</code> |
  | <code>Part of me loves the simplicity of a clean slate, but then I think about families with disabled kids or people hit by natural disasters. Maybe the complexity exists for good reasons?</code>                                 | <code>Embracing a fresh start seems straightforward, yet the reality of supporting those with disabilities or disaster survivors reveals the necessity of intricate systems.</code>                                                                          | <code>Part of me finds the idea of a clean slate appealing, but then I consider families with disabled kids or individuals affected by natural disasters. The complexity might not be necessary after all.</code>                         |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 46
- `warmup_steps`: 4
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
- `max_steps`: 46
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 4
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