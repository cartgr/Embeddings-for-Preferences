---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1800
- loss:BradleyTerryLoss
base_model: sentence-transformers/sentence-t5-xl
widget:
- source_sentence: These programs help people immediately but don't solve underlying
    economic problems. Sometimes temporary work becomes a crutch that delays real
    solutions. The benefits and drawbacks seem pretty evenly matched.
  sentences:
  - Addressing core economic challenges is far more crucial than temporary aid. Long-term
    solutions should take precedence over short-term fixes that only mask issues without
    truly resolving them.
  - These initiatives provide immediate assistance but often fail to address deeper
    economic issues. At times, temporary employment becomes a dependency that postpones
    lasting solutions. The pros and cons appear fairly balanced.
  - Physical libraries are outdated institutions that hinder our progress towards
    true information equality. By fully digitizing resources, we could remove geographical
    limitations, significantly cut costs, and offer everyone immediate access to the
    world's accumulated knowledge, regardless of their location or ability to travel.
- source_sentence: While I appreciate the environmental concerns, banning bottled
    water feels like government overreach. Better to focus on recycling programs and
    maybe tax plastic bottles rather than eliminate consumer choice entirely.
  sentences:
  - Prohibiting the sale of bottled water is a necessary step to address pressing
    environmental issues. It's essential to restrict consumer options to drive a shift
    towards sustainable practices.
  - This is mere political spectacle aimed at undermining Israel's right to self-defense.
    South Africa overlooks Hamas's charter promoting genocide and their tactic of
    using civilians as shields. Such misuse of international law threatens the system's
    integrity.
  - Although I understand the concerns for the environment, prohibiting bottled water
    seems like excessive government intervention. A better approach would be to enhance
    recycling initiatives and consider taxing plastic bottles instead of completely
    removing consumer choice.
- source_sentence: America is still our most natural partner despite recent political
    turbulence. The relationship needs careful management though - we can't just follow
    blindly on every foreign policy adventure like we did in Iraq.
  sentences:
  - The United States remains our closest ally even with the recent political challenges.
    However, we must manage the relationship carefully and not repeat the mistake
    of blindly supporting every foreign policy decision, such as the one in Iraq.
  - Identity politics has divided our society into clashing groups that struggle to
    engage in basic dialogue. We've forsaken mutual values and commonalities for never-ending
    disputes that achieve nothing.
  - Given the current political climate, aligning with the United States may not be
    in our best interest. It's crucial to critically evaluate foreign policy decisions,
    as past actions like the Iraq conflict have shown us the pitfalls of such alliances.
- source_sentence: Families are the backbone of society and deserve every bit of support
    we can give them. Tax credits for children would strengthen communities, boost
    birth rates, and help parents invest in their kids' futures.
  sentences:
  - Blocking should definitely be a user right, not restricted to admins. As adults,
    we have the ability to choose who we wish to engage with. Limiting blocking to
    admins is just another way platforms exert control over our digital freedom.
  - Families form the foundation of society and merit all the support we can offer.
    Child tax credits would empower communities, raise birth rates, and enable parents
    to invest in their children's futures.
  - Individuals should prioritize personal responsibility over relying on government
    support. Tax credits are an unnecessary burden on the economy and do not effectively
    encourage family growth.
- source_sentence: Space exploration represents humanity's greatest achievement and
    our only path to long-term survival. Every dollar spent advancing our cosmic reach
    is an investment in preventing extinction and unlocking infinite resources.
  sentences:
  - Employers should have the right to review social media for obvious warning signs
    or illegal conduct, but continuous monitoring seems excessive. It's crucial to
    establish clear guidelines regarding what they seek and the reasons behind it.
  - Space exploration stands as one of humanity's finest accomplishments and is crucial
    for our survival in the long run. Investing in our ability to explore the cosmos
    is a step toward avoiding extinction and accessing limitless resources.
  - Focusing on space exploration diverts essential resources from pressing issues
    on Earth. Instead of spending on cosmic endeavors, we should invest in solving
    immediate problems like poverty and climate change.
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
    "Space exploration represents humanity's greatest achievement and our only path to long-term survival. Every dollar spent advancing our cosmic reach is an investment in preventing extinction and unlocking infinite resources.",
    "Space exploration stands as one of humanity's finest accomplishments and is crucial for our survival in the long run. Investing in our ability to explore the cosmos is a step toward avoiding extinction and accessing limitless resources.",
    'Focusing on space exploration diverts essential resources from pressing issues on Earth. Instead of spending on cosmic endeavors, we should invest in solving immediate problems like poverty and climate change.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9462, 0.6319],
#         [0.9462, 1.0000, 0.6341],
#         [0.6319, 0.6341, 1.0000]])
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

* Size: 1,800 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                            | negative                                                                           |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | string                                                                             |
  | details | <ul><li>min: 17 tokens</li><li>mean: 41.51 tokens</li><li>max: 152 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 46.02 tokens</li><li>max: 155 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 39.48 tokens</li><li>max: 172 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                  | positive                                                                                                                                                                                                                                                                                                                               | negative                                                                                                                                                                                                                                                                                           |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Devolution works brilliantly because each nation has unique needs and cultures. Scotland's education system, Wales' language policies, and Northern Ireland's peace process all prove that tailored governance delivers better outcomes than one-size-fits-all approaches.</code> | <code>Devolution is highly effective because it caters to the distinct needs and cultures of each nation. The educational system in Scotland, the language initiatives in Wales, and the peace efforts in Northern Ireland all demonstrate that customized governance yields superior results compared to a universal approach.</code> | <code>Centralized government is more efficient, as it ensures consistency and equality across regions. A uniform education system, common language policies, and a collective peace strategy provide stability and cohesion that surpasses the fragmented outcomes of localized governance.</code> |
  | <code>Schools have a responsibility to create inclusive environments where all students see themselves reflected. Comprehensive education about gender and sexuality helps reduce bullying, builds empathy, and prepares kids for the real world they'll navigate as adults.</code>     | <code>Schools should be responsible for fostering inclusive settings where every student can see themselves represented. Teaching thoroughly about gender and sexuality can diminish bullying, enhance empathy, and equip children for the adult world they will face.</code>                                                          | <code>Focusing on traditional education topics is more beneficial for students, as introducing complex subjects like gender and sexuality at a young age could lead to confusion rather than preparing them for adulthood.</code>                                                                  |
  | <code>Climate change is an existential crisis and we have the technology right now to solve it. Every day we delay renewable energy transition, we're literally destroying our planet's future.</code>                                                                                  | <code>Climate change represents a critical threat, and we possess the technology today to address it. Delaying the shift to renewable energy is essentially jeopardizing the future of our planet.</code>                                                                                                                              | <code>The urgency of climate change is overstated, and our current technology is insufficient for meaningful impact. Investing in renewable energy transition now is an unnecessary expense with little immediate benefit.</code>                                                                  |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 1.25e-05
- `max_steps`: 125
- `warmup_steps`: 12

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 4
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1.25e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3.0
- `max_steps`: 125
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 12
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
| 0.4444 | 50   | 0.6212        |
| 0.8889 | 100  | 0.5954        |


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