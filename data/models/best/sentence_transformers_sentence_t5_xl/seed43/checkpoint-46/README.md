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
- source_sentence: Banning private jets goes too far and sets a dangerous precedent
    for government overreach. Heavy carbon taxes and emission offsets make more sense
    than prohibition, even if these flights are environmentally wasteful.
  sentences:
  - Banning private jets is a necessary step and sets a strong precedent for government
    action. Heavy carbon taxes and emission offsets are less effective than prohibition,
    even if these flights are environmentally wasteful.
  - Outright banning of private jets is excessive and could lead to unwanted government
    control. Instead, implementing substantial carbon taxes and requiring emission
    offsets is a more sensible approach, despite the environmental impact of such
    flights.
  - The essence of acting lies in portraying characters beyond one's own identity,
    so imposing rigid casting rules isn't beneficial. Instead, we should aim to enhance
    the writing of trans characters and ensure that trans performers have equal access
    to diverse roles.
- source_sentence: Electric cars are our planet's salvation! Zero tailpipe emissions,
    rapidly improving battery recycling, and renewable energy grids make them exponentially
    cleaner than gas guzzlers. The transition can't happen fast enough.
  sentences:
  - Electric cars are not our planet's salvation! Zero tailpipe emissions, rapidly
    improving battery recycling, and renewable energy grids do not make them exponentially
    cleaner than gas guzzlers. The transition shouldn't happen too quickly.
  - Investing in industries that focus on creating tools of destruction, regardless
    of their defensive purpose, clashes with my ethical principles. I believe there
    are numerous other profitable sectors that align better with my values, offering
    both stability and peace of mind.
  - Reducing pollution from vehicles is crucial for our environment's health. Utilizing
    electric vehicles, alongside advances in battery technology and sustainable energy
    sources, is an essential step in moving away from reliance on gasoline-powered
    cars. This shift is urgent and necessary.
- source_sentence: HS2 has become a vanity project that's lost all connection to value
    for money. The environmental damage alone should make us pause, even ignoring
    the eye-watering cost overruns.
  sentences:
  - The decision to exit the European Union has severely damaged our nation's interests,
    severing crucial economic ties and diminishing our role on the world stage. The
    supposed benefits are overshadowed by increased bureaucratic burdens and isolation
    from international partners.
  - HS2 has become an essential project that's clearly connected to value for money.
    The environmental benefits alone should encourage us to continue, even considering
    the reasonable cost overruns.
  - The HS2 project exemplifies excessive spending with little regard for environmental
    preservation. Its escalating costs and ecological impacts warrant reconsideration
    and potentially halting further development.
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
  - Balancing economic incentives like tax reductions with ensuring adequate resources
    for education and infrastructure is challenging. While growth is important, maintaining
    strong public services is equally vital.
- source_sentence: The pound sterling has served Britain well for over a thousand
    years. Surrendering our currency means surrendering our sovereignty to Brussels
    bureaucrats who don't understand British economic needs or values.
  sentences:
  - Adopting the euro would compromise the UK's autonomy, as it would place critical
    economic decisions in the hands of European authorities unfamiliar with British
    priorities and traditions.
  - The pound sterling has served Britain well for over a thousand years. Embracing
    a new currency signifies aligning with Brussels experts who understand British
    economic needs and values better than ever.
  - The notion of a climate crisis is exaggerated, as climate models frequently fail
    to predict accurately, and carbon dioxide plays a crucial role in enhancing plant
    growth.
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
    "The pound sterling has served Britain well for over a thousand years. Surrendering our currency means surrendering our sovereignty to Brussels bureaucrats who don't understand British economic needs or values.",
    "Adopting the euro would compromise the UK's autonomy, as it would place critical economic decisions in the hands of European authorities unfamiliar with British priorities and traditions.",
    'The pound sterling has served Britain well for over a thousand years. Embracing a new currency signifies aligning with Brussels experts who understand British economic needs and values better than ever.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6708, 0.6208],
#         [0.6708, 1.0000, 0.2129],
#         [0.6208, 0.2129, 1.0000]])
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
  |         | anchor                                                                              | positive                                                                           | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | string                                                                              |
  | details | <ul><li>min: 25 tokens</li><li>mean: 42.01 tokens</li><li>max: 140 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 45.79 tokens</li><li>max: 86 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 42.95 tokens</li><li>max: 138 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                         | positive                                                                                                                                                                                                                                                                                                                           | negative                                                                                                                                                                                                                                                                           |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Phasing out diesel and petrol cars makes environmental sense, but we need realistic timelines. Rural areas and developing regions still lack charging infrastructure, so a gradual 10-15 year transition seems fair.</code>                                              | <code>While moving away from traditional fuel vehicles is crucial for ecological reasons, it's important to establish a practical schedule. Many less urbanized and economically developing areas aren't equipped with the necessary charging facilities, so allowing a decade or more for this change would be reasonable.</code> | <code>Phasing out diesel and petrol cars might seem environmentally beneficial, but we should avoid unrealistic timelines. Rural areas and developing regions may still lack charging infrastructure, so a rapid 5-year transition seems fair.</code>                              |
  | <code>Every dollar we spend on luxuries while children starve overseas is a moral failure. If we can prevent suffering and death at minimal cost to ourselves, we're absolutely obligated to do so.</code>                                                                     | <code>Our ethical duty demands we divert funds from non-essential purchases to help alleviate global hunger and save lives whenever possible.</code>                                                                                                                                                                               | <code>Every dollar we spend on luxuries while children starve overseas is justified. If we can prevent suffering and death at minimal cost to ourselves, we're not obligated to do so.</code>                                                                                      |
  | <code>Baldness causes genuine psychological suffering and social discrimination. Government research funding would accelerate treatments, improve millions of lives, and generate valuable medical breakthroughs. This is absolutely a worthy public health investment.</code> | <code>Hair loss can severely impact mental health and lead to societal bias. Funding scientific studies could speed up discovering effective solutions, enhancing countless lives and contributing to significant advancements in medicine. Investing in this area is undoubtedly beneficial for public health.</code>             | <code>Baldness causes genuine psychological suffering and social discrimination. Government research funding would accelerate treatments, improve millions of lives, and generate valuable medical breakthroughs. This is absolutely NOT a worthy public health investment.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 46
- `warmup_steps`: 4
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