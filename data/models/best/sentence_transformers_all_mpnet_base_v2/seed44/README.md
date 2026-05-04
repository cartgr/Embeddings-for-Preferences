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
- source_sentence: Most night shift premiums are just legacy policies from when it
    actually mattered. With 24/7 everything these days, working nights is often easier
    - less traffic, fewer interruptions, more autonomy.
  sentences:
  - These days, working during nighttime hours can be more convenient due to less
    congestion and fewer disruptions, making traditional night shift pay bonuses seem
    unnecessary.
  - Most night shift premiums are just outdated policies from a time when they were
    necessary. In today's 24/7 society, working nights is often more challenging -
    more traffic, constant interruptions, and reduced autonomy.
  - The debate over cryptocurrency is complex; while it has the potential to foster
    new technologies, its role in illicit activities and environmental harm cannot
    be ignored. Finding a regulatory approach that addresses these issues without
    stifling innovation might be more effective than banning it outright.
- source_sentence: Government funding would dramatically reduce corporate influence,
    which I support. My concern is whether the funding amounts would be adequate and
    how we'd handle independent expenditures that would still favor wealthy interests.
  sentences:
  - The right to migrate should be universally accepted, especially for those escaping
    dire circumstances. Restrictive immigration laws unjustly penalize individuals
    based on where they happen to be born, undermining their inherent right to seek
    a better life.
  - Government funding would dramatically reduce corporate influence, which I oppose.
    My concern is that the funding amounts would be excessive and how we'd handle
    independent expenditures that would still favor wealthy interests.
  - I believe that reducing the sway of corporations over politics is crucial, and
    state sponsorship could be a key solution. However, I'm worried about ensuring
    sufficient resources and addressing the issue of affluent groups continuing to
    shape outcomes through other financial means.
- source_sentence: Basic income makes sense in principle - simpler than our current
    welfare mess and gives people real freedom. Though £100 feels low and I worry
    about funding it without gutting other essential services we need.
  sentences:
  - The current welfare system is overly complex and burdensome, while a universal
    basic income could provide individuals with greater autonomy. Nonetheless, the
    proposed amount is insufficient, and I'm apprehensive about securing the necessary
    funds without compromising critical public services.
  - Basic income seems reasonable on the surface, appearing simpler than our current
    welfare system and seemingly offering people real freedom. However, £100 is actually
    too low, and I am concerned about funding it without cutting into other vital
    services we rely on.
  - To reduce the tragic consequences of impaired driving during New Year's celebrations,
    implementing restrictions on alcohol availability could be a crucial measure to
    protect families from preventable harm.
- source_sentence: There are compelling arguments on both sides here. Uniforms clearly
    help with equality and focus, but they also limit personal expression. Really
    depends on the specific school community and what they value most.
  sentences:
  - Deciding whether to use uniforms in schools hinges on the community's priorities.
    While uniforms can foster a sense of equality and enhance concentration, they
    might also restrict students' ability to showcase their individuality.
  - There are compelling arguments on both sides here. Uniforms clearly help with
    equality and focus, but they also limit personal expression. Really depends on
    the specific school community and what they value least.
  - Medical interventions for hearing loss should be approached with caution, as deaf
    individuals belong to thriving communities with their own languages and cultural
    identities that deserve respect and appreciation, not a presumption of needing
    correction.
- source_sentence: Opt-out would definitely increase donation rates, but we need robust
    safeguards - easy opt-out procedures, family consultation rights, and public education
    campaigns about the change.
  sentences:
  - Investing in nuclear energy is crucial for reliable power and achieving environmental
    targets, despite financial challenges seen in projects like Hinkley Point C. We
    must improve how these projects are managed and ensure financial plans are feasible.
  - Opt-out might increase donation rates, but we should avoid robust safeguards -
    difficult opt-out procedures, limiting family consultation rights, and neglecting
    public education campaigns about the change.
  - To boost organ donation, implementing an automatic enrollment system with clear
    exit options, ensuring family involvement, and informing the public about this
    shift is essential.
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
    'Opt-out would definitely increase donation rates, but we need robust safeguards - easy opt-out procedures, family consultation rights, and public education campaigns about the change.',
    'To boost organ donation, implementing an automatic enrollment system with clear exit options, ensuring family involvement, and informing the public about this shift is essential.',
    'Opt-out might increase donation rates, but we should avoid robust safeguards - difficult opt-out procedures, limiting family consultation rights, and neglecting public education campaigns about the change.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8406, 0.8029],
#         [0.8406, 1.0000, 0.6710],
#         [0.8029, 0.6710, 1.0000]])
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
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 26 tokens</li><li>mean: 40.02 tokens</li><li>max: 71 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 42.99 tokens</li><li>max: 77 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 41.13 tokens</li><li>max: 70 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                       | positive                                                                                                                                                                                                                                                                                                                  | negative                                                                                                                                                                                                                                                                                             |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Nuclear could work well for larger islands like mainland Britain, but smaller islands might struggle with the infrastructure costs and emergency planning requirements. We'd need careful feasibility studies for each location first.</code>          | <code>For expansive regions like Britain, nuclear energy could be a viable option, yet the financial and logistical challenges might be overwhelming for smaller islands. It's crucial to evaluate each site thoroughly before proceeding.</code>                                                                         | <code>Nuclear might not be suitable for larger islands like mainland Britain, but smaller islands could handle the infrastructure costs and emergency planning requirements more easily. We shouldn't require feasibility studies for each location first.</code>                                    |
  | <code>Complete waste of taxpayer money. These vanity projects line contractors' pockets while ordinary people struggle with cost of living. The government should cut spending, reduce taxes, and let the free market provide what's actually needed.</code> | <code>Government funds are frequently misused on extravagant projects that benefit a select few, while the average citizen faces financial hardship. It would be more beneficial for the government to scale back on unnecessary expenses, lower taxes, and allow market forces to address genuine societal needs.</code> | <code>Complete investment of taxpayer money. These essential projects enhance community growth while ordinary people benefit from improved infrastructure. The government should maintain spending, support essential services, and let the free market flourish alongside state initiatives.</code> |
  | <code>Animal testing remains absolutely essential for medical progress. Regulations ensure humane treatment while allowing us to save countless human lives through vaccines, cancer treatments, and surgical advances.</code>                               | <code>The use of animals in research is crucial for developing life-saving medical treatments, as ethical guidelines protect animal welfare while enabling breakthroughs in healthcare such as vaccines and cancer therapies.</code>                                                                                      | <code>Animal testing remains absolutely unnecessary for medical progress. Regulations ensure humane treatment while allowing us to avoid harming countless animals through alternative methods and technological advances.</code>                                                                    |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 62
- `warmup_steps`: 6
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

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8850 | 50   | 0.7503        |


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