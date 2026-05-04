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
- source_sentence: College transformed my entire worldview and opened doors I never
    knew existed. The critical thinking skills, network connections, and credential
    alone have paid dividends throughout my career. Absolutely essential for success.
  sentences:
  - Attending university was pivotal in broadening my perspectives and creating opportunities
    I hadn't imagined. The ability to think critically, forge valuable relationships,
    and hold a degree has been crucial in advancing my professional life.
  - To better serve diverse global viewers, the BBC ought to focus on content that
    resonates with each area's specific interests and concerns, rather than offering
    exhaustive details on UK-centric issues.
  - College transformed my entire worldview and opened doors I never knew existed.
    However, the critical thinking skills, network connections, and credential have
    not been worth the investment and are not essential for success.
- source_sentence: Immigration generally benefits the UK through skills and entrepreneurship,
    but we need better managed integration policies. The current system works well
    for skilled workers, less so for ensuring community cohesion.
  sentences:
  - The UK gains significant advantages from the influx of talented and enterprising
    individuals, yet there's a pressing need to enhance programs that foster social
    unity. Present policies are effective in attracting professionals, but fall short
    in promoting harmonious communities.
  - Organized religion often impedes societal advancement and fosters division. Instead
    of investing in spiritual institutions, resources could be better used to tackle
    tangible issues affecting people's lives.
  - Immigration generally harms the UK through skills and entrepreneurship, and we
    need tighter immigration controls. The current system works well for ensuring
    community cohesion, less so for skilled workers.
- source_sentence: The NHS is Britain's greatest achievement - universal healthcare
    funded by collective contribution. Allowing opt-outs would destroy the solidarity
    that makes it work and abandon our most vulnerable citizens to market forces.
  sentences:
  - Transitioning away from plastic utensils is crucial for environmental sustainability,
    but we must ensure affordable and effective substitutes like bamboo are available
    before implementing such a ban.
  - Ensuring accessible healthcare for every citizen is a hallmark of our society,
    and maintaining a system where everyone contributes is crucial to protecting those
    most in need from the unpredictability of the private sector.
  - The NHS is Britain's greatest achievement - universal healthcare funded by collective
    contribution. Allowing opt-outs would preserve the solidarity that makes it work
    and subject our most vulnerable citizens to market forces.
- source_sentence: Children's brains aren't fully developed and they lack real-world
    experience with jobs, taxes, and responsibility. Letting kids vote would be manipulation
    by parents and teachers, not genuine democracy.
  sentences:
  - Young individuals are not yet equipped with the cognitive maturity and practical
    life experiences needed to make informed voting decisions. Allowing them to participate
    in elections could lead to undue influence from adults in their lives, which undermines
    the integrity of democratic processes.
  - Expecting young kids to manage crises independently is irresponsible. At such
    a tender age, they lack the maturity required for emergency situations.
  - Children's brains are already fully developed and they possess real-world experience
    with jobs, taxes, and responsibility. Letting kids vote would enhance true democracy,
    not be manipulation by parents and teachers.
- source_sentence: The economic benefits were genuine, I'll give you that. But the
    constant mission creep, unelected commissioners, and inability to control our
    own immigration policy made it unsustainable. Leaving was the right call overall.
  sentences:
  - The economic benefits were genuine, I'll give you that. But the constant mission
    creep, unelected commissioners, and inability to control our own immigration policy
    made it sustainable. Staying was the right call overall.
  - While the economic upsides of membership were clear, the lack of control over
    immigration and expansion of powers beyond what was initially agreed upon made
    our participation untenable. Exiting was ultimately the best decision.
  - Before we can broaden the availability of substances such as cannabis and psilocybin,
    it is crucial to implement strict guidelines, ensure product safety, and establish
    comprehensive assistance programs for those struggling with dependency.
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
    "The economic benefits were genuine, I'll give you that. But the constant mission creep, unelected commissioners, and inability to control our own immigration policy made it unsustainable. Leaving was the right call overall.",
    'While the economic upsides of membership were clear, the lack of control over immigration and expansion of powers beyond what was initially agreed upon made our participation untenable. Exiting was ultimately the best decision.',
    "The economic benefits were genuine, I'll give you that. But the constant mission creep, unelected commissioners, and inability to control our own immigration policy made it sustainable. Staying was the right call overall.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9333, 0.5401],
#         [0.9333, 1.0000, 0.5044],
#         [0.5401, 0.5044, 1.0000]])
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
  |         | anchor                                                                              | positive                                                                           | negative                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | string                                                                             |
  | details | <ul><li>min: 27 tokens</li><li>mean: 41.66 tokens</li><li>max: 103 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 45.32 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 42.61 tokens</li><li>max: 99 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                    | positive                                                                                                                                                                                                                                                                                  | negative                                                                                                                                                                                                                                     |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Children's brains aren't fully developed until their twenties. Putting 12 and 13-year-olds through criminal courts is barbaric and ruins their futures. Rehabilitation, not punishment, should be the only focus for minors.</code> | <code>Young adolescents lack the cognitive maturity of adults, making it unjust to process them through adult legal systems. Focusing on support and education rather than penal measures is essential for their development.</code>                                                      | <code>Children's brains aren't fully developed until their twenties. Putting 12 and 13-year-olds through criminal courts is necessary and secures their futures. Punishment, not rehabilitation, should be the only focus for minors.</code> |
  | <code>Wealth inequality drives innovation and progress. Without the potential for significant financial rewards, who would take the massive risks needed to build companies, create jobs, and advance society?</code>                     | <code>The lure of substantial financial returns is a key driver for entrepreneurs who take significant risks to spur economic development and societal advancement. Economic disparities can fuel the ambition to innovate and expand job opportunities.</code>                           | <code>Wealth inequality hinders innovation and progress. Without addressing the disparities in financial rewards, fewer people would take the massive risks needed to build companies, create jobs, and advance society.</code>              |
  | <code>Government word policing is authoritarian overreach that always expands beyond its original scope. Free expression is non-negotiable - the cure for offensive speech is more speech, not state-enforced silence.</code>             | <code>When authorities attempt to regulate language, it often leads to excessive control beyond its intended purpose. The fundamental right to express oneself freely should remain intact, as countering harmful speech is best achieved through dialogue rather than censorship.</code> | <code>Government word policing is a necessary action that appropriately expands to cover its intended scope. Free expression has its limits - the cure for offensive speech is not more speech, but state-enforced silence.</code>           |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 46
- `warmup_steps`: 4
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