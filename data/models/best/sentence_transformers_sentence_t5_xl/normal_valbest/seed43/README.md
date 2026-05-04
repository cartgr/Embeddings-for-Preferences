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
- source_sentence: Part of me thinks people should have easier access to procedures
    that make them happy, but I also worry we're just feeding into impossible beauty
    standards. It's hard to know if more accessibility helps or hurts society overall.
  sentences:
  - Restricting access to cosmetic procedures is crucial because it prevents the perpetuation
    of harmful beauty norms. Society should focus on accepting natural appearances
    rather than making it easier to alter them.
  - Offering financial rewards for organ donors could annually save many lives. It
    is only fair that individuals receive compensation for assuming medical risks,
    and market-driven strategies often achieve superior results compared to purely
    charitable efforts.
  - I believe that making procedures more accessible to those who find happiness in
    them is important, yet I fear it may contribute to unrealistic beauty ideals.
    It's challenging to determine whether increased accessibility benefits or harms
    society as a whole.
- source_sentence: Climate has always changed naturally and while human activity likely
    contributes, declaring an 'emergency' seems like fear-mongering designed to justify
    massive government overreach and economic disruption.
  sentences:
  - A national health insurance scheme is economically sensible and would benefit
    millions, although I'm concerned about the challenges of implementation and whether
    we can sustain quality care while effectively managing costs.
  - The climate crisis is a genuine emergency necessitating immediate action to curb
    human activities that are accelerating environmental harm, and government intervention
    is crucial to prevent further economic and ecological damage.
  - Climate changes naturally over time, and although human actions likely play a
    role, labeling the situation as an 'emergency' seems like fear-inducing rhetoric
    meant to justify excessive governmental control and economic upheaval.
- source_sentence: Independent journalism deserves support, but government funding
    creates concerning dependencies. Better to establish strict firewalls and transparent
    processes than watch local news disappear entirely due to market failures.
  sentences:
  - Government funding is essential to sustain independent journalism, as market forces
    alone cannot ensure its survival. Without these funds, the risk of losing diverse
    local news sources is too high.
  - Roadways are catastrophic for the environment, as they fragment ecosystems, elevate
    pollution, and foster reliance on vehicles. Instead of expanding asphalt surfaces,
    we should transform them into green areas.
  - Supporting independent journalism is crucial, but relying on government funding
    can lead to problematic dependencies. It is preferable to create robust firewalls
    and transparent procedures rather than allowing local news to vanish completely
    due to market failures.
- source_sentence: Schools already get plenty of funding for literacy instruction.
    Parents and communities should take more responsibility here instead of expecting
    government to solve every educational challenge that comes up.
  sentences:
  - Schools receive ample funding for literacy programs. Instead of relying on the
    government to address every educational issue, parents and communities should
    take on more responsibility.
  - Government intervention is crucial to address deficiencies in literacy education.
    Relying solely on parents and communities without adequate funding exacerbates
    the problem.
  - Privilege discussions are merely a form of performative activism, allowing individuals
    to feel superior while achieving nothing substantial. True transformation arises
    from policies and actions, not endless arguments about who has more advantages.
- source_sentence: Austerity is economic malpractice that punishes ordinary people
    for elite failures. It consistently deepens recessions, increases unemployment,
    and transfers wealth upward while solving nothing. Pure ideological nonsense.
  sentences:
  - Genetically altering animals is an affront to the natural order, disrespecting
    our boundaries and rights to modify creatures for our purposes. Such arrogance
    could result in unexpected outcomes and permanent harm to ecosystems and biodiversity.
  - Fiscal discipline is a necessary strategy to address economic challenges. It stabilizes
    the economy, creates jobs, and ensures that resources are distributed fairly,
    providing real solutions to financial problems.
  - Austerity represents an economic misjudgment that harms regular citizens due to
    the shortcomings of the elite. It frequently exacerbates economic downturns, raises
    unemployment rates, and shifts wealth upwards without addressing underlying issues.
    It's simply ideological absurdity.
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
    'Austerity is economic malpractice that punishes ordinary people for elite failures. It consistently deepens recessions, increases unemployment, and transfers wealth upward while solving nothing. Pure ideological nonsense.',
    "Austerity represents an economic misjudgment that harms regular citizens due to the shortcomings of the elite. It frequently exacerbates economic downturns, raises unemployment rates, and shifts wealth upwards without addressing underlying issues. It's simply ideological absurdity.",
    'Fiscal discipline is a necessary strategy to address economic challenges. It stabilizes the economy, creates jobs, and ensures that resources are distributed fairly, providing real solutions to financial problems.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9175, 0.5187],
#         [0.9175, 1.0000, 0.5528],
#         [0.5187, 0.5528, 1.0000]])
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
  |         | anchor                                                                             | positive                                                                            | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                            |
  | details | <ul><li>min: 27 tokens</li><li>mean: 41.56 tokens</li><li>max: 90 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 46.14 tokens</li><li>max: 102 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 39.4 tokens</li><li>max: 97 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                          | positive                                                                                                                                                                                                                                                                                               | negative                                                                                                                                                                                                                                      |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Stricter controls make sense for repeat DUI offenders and those with documented addiction issues. Background checks could help, though I worry about implementation costs and potential discrimination against recovering addicts.</code> | <code>Tighter regulations are sensible for those with repeated DUI offenses and individuals with known addiction problems. Background checks might assist, although I'm concerned about the expenses of implementation and the risk of unfair treatment toward those recovering from addiction.</code> | <code>Loosening restrictions for those with DUI records and addiction issues could be beneficial. Eliminating background checks might reduce costs and prevent unfair targeting of individuals who are in recovery.</code>                    |
  | <code>Part of me thinks we obviously need healthier, better-educated kids. But part of me wonders if more government involvement just creates dependency and crowds out family responsibility.</code>                                           | <code>A part of me believes we clearly need to focus on having healthier and better-educated children. Yet, another part questions if increased government intervention merely fosters reliance and diminishes family accountability.</code>                                                           | <code>Government support is crucial for fostering children's health and education, as it enhances opportunities and alleviates pressures on families trying to meet these responsibilities alone.</code>                                      |
  | <code>Housing is a human right, not a commodity for speculation. Maximum prices would finally give working families a chance at homeownership and stop wealthy investors from hoarding properties.</code>                                       | <code>Housing should be considered a human right rather than a vehicle for financial speculation. Implementing price caps could allow working families to achieve homeownership and deter wealthy investors from accumulating properties.</code>                                                       | <code>The housing market thrives on investment opportunities, and regulating prices could stifle economic growth. Encouraging investors to acquire properties can lead to increased development and housing availability for everyone.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 1.25e-05
- `max_steps`: 125
- `warmup_steps`: 12
- `seed`: 43

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
| 0.4444 | 50   | 0.6204        |
| 0.8889 | 100  | 0.5979        |


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