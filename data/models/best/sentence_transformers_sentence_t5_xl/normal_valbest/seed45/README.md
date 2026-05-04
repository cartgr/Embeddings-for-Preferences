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
- source_sentence: Publishers invest heavily in quality control and distribution infrastructure
    that researchers rely on. Completely free access might sound nice but could undermine
    scientific rigor.
  sentences:
  - Publishers allocate substantial resources to quality assurance and distribution
    networks crucial for researchers. While completely free access seems appealing,
    it could compromise scientific excellence.
  - Zero-emission buildings are no longer just an option; they're crucial for our
    existence. The government must enforce regulations immediately to prevent developers
    from profiting at the expense of our planet's future.
  - Free access to research is essential for fostering innovation and ensuring scientific
    integrity, as it removes barriers to information and encourages collaboration.
- source_sentence: While children's education matters tremendously, cutting all adult
    programs would be shortsighted. Community colleges and workforce development generate
    serious economic returns that benefit everyone, including future students.
  sentences:
  - Focusing solely on children's education is key since adult programs like community
    colleges and job training are much less effective and offer minimal returns compared
    to investing in the youth.
  - Although the importance of children's education cannot be understated, eliminating
    all adult programs would be an oversight. Community colleges and workforce training
    offer significant economic benefits that help everyone, including the next generation
    of students.
  - International students elevate universities to global leaders. Instituting a 20%
    minimum would guarantee that each UK institution gains from varied viewpoints,
    cultural interactions, and the creativity sparked by gathering exceptional talents
    from around the globe.
- source_sentence: Physical punishment usually does more harm than good and often
    escalates beyond what parents intend. There are proven alternatives that teach
    better lessons without the emotional damage.
  sentences:
  - Strict physical discipline is essential for instilling discipline and respect
    in children, as alternative methods often fail to achieve the desired behavioral
    outcomes.
  - Physical discipline often causes more harm than benefit and can escalate beyond
    parental intentions. Proven alternatives exist that impart valuable lessons without
    causing emotional harm.
  - This topic truly divides me. I recognize the benefits of affirmative action for
    marginalized communities, yet I also acknowledge the worries about equity. Perhaps
    the emphasis should shift towards socioeconomic considerations instead?
- source_sentence: Legal systems should be colorblind and apply identical standards
    to everyone regardless of background. Special accommodations based on minority
    status fundamentally violate the principle of equal justice under law.
  sentences:
  - Justice systems should be blind to race and maintain the same criteria for all
    individuals, irrespective of their backgrounds. Making exceptions based on minority
    status disrupts the core principle of equal justice under the law.
  - Cannabis is certainly less harmful than alcohol since you can't overdose fatally
    on it, and it isn't physically addictive. However, inhaling smoke harms your lungs,
    and driving while high remains risky.
  - Recognizing and addressing the unique challenges faced by minorities is essential
    for ensuring true fairness in legal systems. Tailoring standards to account for
    diverse backgrounds helps achieve genuine justice.
- source_sentence: The NHS is criminally underfunded and we all know it. I'd gladly
    pay an extra 2p on income tax if it means proper staffing and equipment. Our health
    is worth investing in properly.
  sentences:
  - The NHS is inefficient and doesn't need more funding. Instead of raising taxes,
    we should focus on cutting waste and improving management to ensure better healthcare
    outcomes.
  - Education should be considered a fundamental right and not treated as a product.
    Charging tuition fees creates obstacles that reinforce class differences and leave
    students with long-term debt, which ultimately hinders personal and societal advancement.
  - The NHS is woefully lacking in funds, and it's a fact we're all aware of. I'd
    willingly contribute an extra 2p on my income tax if it ensured adequate staffing
    and equipment. Investing properly in our health is crucial.
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
    "The NHS is criminally underfunded and we all know it. I'd gladly pay an extra 2p on income tax if it means proper staffing and equipment. Our health is worth investing in properly.",
    "The NHS is woefully lacking in funds, and it's a fact we're all aware of. I'd willingly contribute an extra 2p on my income tax if it ensured adequate staffing and equipment. Investing properly in our health is crucial.",
    "The NHS is inefficient and doesn't need more funding. Instead of raising taxes, we should focus on cutting waste and improving management to ensure better healthcare outcomes.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9610, 0.6895],
#         [0.9610, 1.0000, 0.6697],
#         [0.6895, 0.6697, 1.0000]])
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
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 17 tokens</li><li>mean: 41.54 tokens</li><li>max: 90 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 46.13 tokens</li><li>max: 90 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 39.33 tokens</li><li>max: 85 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                       | positive                                                                                                                                                                                                                                                                                                         | negative                                                                                                                                                                                                                                        |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Most military spending feels like Cold War thinking when our real challenges are economic inequality and climate change. Some basic defense makes sense, but nothing like current levels.</code>                                                       | <code>Most military expenditure seems like Cold War mentality when our true issues are economic disparity and environmental change. Some fundamental defense is reasonable, but not at the current scale.</code>                                                                                                 | <code>Our primary focus should be on maintaining a strong military presence to ensure national security, rather than diverting resources to address economic or environmental concerns.</code>                                                  |
  | <code>Registration isn't that difficult currently, and requiring some minimal effort helps ensure voters are genuinely engaged. Automatic systems could create bloated, inaccurate voter rolls that complicate election administration unnecessarily.</code> | <code>Currently, the process of registration isn't overly challenging, and requiring a small amount of effort helps ensure that voters are truly committed. Implementing automatic systems might lead to inflated and inaccurate voter lists, making election management more complicated than necessary.</code> | <code>Simplifying voter registration through automatic systems would enhance accessibility and ensure broader participation, reducing the risk of disenfranchisement and making elections smoother to manage.</code>                            |
  | <code>Personal responsibility means paying for your own choices. Free birth control is just another government handout that taxpayers shouldn't fund. People need to take accountability for their decisions.</code>                                         | <code>Taking responsibility means covering the costs of your own actions. Providing free birth control is merely another government benefit that taxpayers shouldn't have to support. Individuals should be accountable for their life choices.</code>                                                           | <code>Access to free birth control is essential for empowering individuals with the ability to make informed choices about their reproductive health, and it is a wise investment of taxpayer resources that promotes public well-being.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 1.25e-05
- `max_steps`: 125
- `warmup_steps`: 12
- `seed`: 45

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

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4444 | 50   | 0.6209        |
| 0.8889 | 100  | 0.5959        |


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