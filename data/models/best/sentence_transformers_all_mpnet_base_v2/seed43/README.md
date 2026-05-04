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
- source_sentence: Capitalism creates wealth but also massive inequality. Socialist
    countries often stagnate, but Nordic models show promise. Maybe the answer isn't
    choosing sides but finding the right balance between markets and government intervention.
  sentences:
  - The increasing number of immigrants, coupled with high birthrates and challenges
    in adopting secular norms, is leading to the development of distinct cultural
    enclaves in Europe. These changes, marked by separate legal frameworks and areas
    of restricted access, suggest a shift towards cultural displacement rather than
    harmonious integration.
  - Neither unregulated capitalism nor purely socialist systems provide a complete
    solution. A blend of free enterprise with strategic state oversight, as demonstrated
    by some Nordic nations, could bridge the gap between economic growth and social
    equity.
  - Capitalism creates wealth but also massive inequality. Socialist countries often
    stagnate, and while Nordic models show promise, maybe the answer is to choose
    sides and not seek a balance between markets and government intervention.
- source_sentence: Local manufacturing is the backbone of any resilient economy. Every
    dollar spent on imports is a dollar taken from our neighbors' pockets and our
    community's future.
  sentences:
  - Local manufacturing is the backbone of any resilient economy. Every dollar spent
    on imports is a dollar contributed to our neighbors' pockets and our community's
    future.
  - The broad shutdowns led to severe economic damage and hindered educational progress,
    which will have long-lasting negative effects. A targeted approach to safeguard
    those at higher risk would have been a more balanced strategy.
  - Sustaining a strong and independent economy requires supporting domestic production.
    Investing in foreign goods undermines local businesses and weakens our community's
    economic prospects.
- source_sentence: While I understand face coverings can help in some situations,
    making them compulsory everywhere indoors feels like government overreach. People
    should be able to assess their own risk and make personal choices.
  sentences:
  - The expansion of NATO has led to increased tensions, and incorporating Ukraine
    could escalate matters dangerously. It's crucial to address such disputes through
    negotiation to avoid the potential for catastrophic outcomes.
  - Mandating masks in all indoor spaces is an overstep by the authorities. Individuals
    ought to have the freedom to evaluate the risks for themselves and decide accordingly.
  - While I understand face coverings can help in some situations, making them compulsory
    everywhere indoors does not feel like government overreach. People should not
    be able to assess their own risk and make personal choices.
- source_sentence: Sport hunting is legalized animal cruelty disguised as tradition.
    These are sentient beings with complex social lives, not targets for human amusement.
    The conservation argument is just convenient justification.
  sentences:
  - Targeting animals for sport is an ethical issue, camouflaged as a cultural practice.
    Animals have intricate social structures and deserve compassion, not to be part
    of entertainment. Claims that it's beneficial for conservation are often misleading.
  - Sport hunting is a respected tradition rather than legalized animal cruelty. These
    animals serve as targets for human amusement, not sentient beings with complex
    social lives. The conservation argument is a valid justification.
  - Ensuring children's safety while cycling is crucial, which makes head protection
    non-negotiable. However, implementing compulsory helmet regulations could unintentionally
    lead to fewer families engaging in biking if they lack the financial means for
    the necessary equipment.
- source_sentence: Competition and private investment have improved many rail services,
    even if imperfectly. Renationalization risks bureaucratic stagnation and taxpayer
    burden, though I'll admit the current system needs major reforms to work properly.
  sentences:
  - The introduction of market-driven principles in railway management has led to
    noticeable enhancements, despite some flaws. Transitioning back to state ownership
    could result in inefficiencies and increased costs for the public, although significant
    changes are necessary for the current model to function effectively.
  - Competition and private investment have hindered many rail services, even if imperfectly.
    Renationalization promises bureaucratic efficiency and reduces taxpayer burden,
    though I'll admit the current system needs major reforms to work properly.
  - Policies targeting high-income earners as a solution to economic disparities are
    misguided. Individuals with substantial incomes are pivotal in job creation and
    innovation, and they already contribute significantly in taxes. Penalizing their
    success could stifle the economic progress that benefits society as a whole.
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
    "Competition and private investment have improved many rail services, even if imperfectly. Renationalization risks bureaucratic stagnation and taxpayer burden, though I'll admit the current system needs major reforms to work properly.",
    'The introduction of market-driven principles in railway management has led to noticeable enhancements, despite some flaws. Transitioning back to state ownership could result in inefficiencies and increased costs for the public, although significant changes are necessary for the current model to function effectively.',
    "Competition and private investment have hindered many rail services, even if imperfectly. Renationalization promises bureaucratic efficiency and reduces taxpayer burden, though I'll admit the current system needs major reforms to work properly.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8968, 0.9582],
#         [0.8968, 1.0000, 0.8565],
#         [0.9582, 0.8565, 1.0000]])
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
  | details | <ul><li>min: 26 tokens</li><li>mean: 40.11 tokens</li><li>max: 155 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 43.14 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 41.07 tokens</li><li>max: 152 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                               | positive                                                                                                                                                                                                              | negative                                                                                                                                                                                                                             |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Opt-out would definitely increase donation rates, but we need robust safeguards - easy opt-out procedures, family consultation rights, and public education campaigns about the change.</code> | <code>To boost organ donation, implementing an automatic enrollment system with clear exit options, ensuring family involvement, and informing the public about this shift is essential.</code>                       | <code>Opt-out might increase donation rates, but we should avoid robust safeguards - difficult opt-out procedures, limiting family consultation rights, and neglecting public education campaigns about the change.</code>           |
  | <code>Part of me thinks coding is super valuable for problem-solving skills, but then again we can't teach everything. Are we just following the latest tech trend or actually helping kids?</code>  | <code>While coding can enhance problem-solving abilities, it's important to question if we're merely riding a wave of technological enthusiasm rather than focusing on the core educational needs of children.</code> | <code>Part of me thinks coding is super valuable for problem-solving skills, but then again we can't teach everything. We should definitely make coding a priority for kids rather than just following the latest tech trend.</code> |
  | <code>Part of me wants to protect kids from predatory marketing, but another part thinks personal choice matters. Where exactly do we draw the line on government intervention?</code>               | <code>Balancing the safeguarding of children from aggressive advertising and respecting individual autonomy is challenging. We must carefully consider the extent to which authorities should step in.</code>         | <code>Part of me wants to protect kids from predatory marketing, but another part thinks personal choice does not matter. We should clearly draw the line on government intervention to ensure strict regulations.</code>            |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 62
- `warmup_steps`: 6
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

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8850 | 50   | 0.7531        |


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