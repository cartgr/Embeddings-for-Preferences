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
- source_sentence: While preventing cancer sounds great in theory, taxpayers shouldn't
    fund every medical intervention. People can pay for their own vaccines - we already
    spend too much on healthcare programs.
  sentences:
  - Public funding for vaccines is crucial as it ensures everyone has access to life-saving
    treatments and ultimately reduces long-term healthcare costs by preventing diseases
    like cancer.
  - At times, seeing crowded urban areas makes me feel pessimistic, while reading
    about lower birth rates makes me question if there will be enough labor in the
    future. Ultimately, it seems to hinge on our ability to manage resources effectively.
  - Although cancer prevention seems ideal in concept, taxpayer money shouldn't cover
    all medical procedures. Individuals should finance their own vaccinations since
    we already allocate too much to healthcare initiatives.
- source_sentence: Hard to say honestly. Bitcoin has speculative elements for sure,
    but also real technological innovation and growing acceptance. Could go either
    way long-term.
  sentences:
  - It's difficult to determine for sure. While Bitcoin has speculative aspects, it
    also represents genuine technological progress and increasing mainstream acceptance.
    The long-term outcome remains uncertain.
  - Relying on Bitcoin is risky due to its speculative nature, and its so-called innovation
    and acceptance are overhyped. In the long run, it seems likely to falter.
  - Gender disparity significantly influences homophobia, particularly through the
    fear of men being perceived as 'feminine.' I also believe that religious and cultural
    influences are crucial in forming these views.
- source_sentence: This whole panic about non-readers is pure snobbery. Audiobooks,
    podcasts, interactive media - we have more ways to learn than ever before. Gatekeeping
    knowledge behind printed pages is outdated and frankly ridiculous.
  sentences:
  - The decline in reading physical books is concerning, as it signifies a departure
    from deep engagement with text. Emphasizing printed material is essential for
    maintaining a rigorous and thorough understanding of complex topics.
  - The fuss over people not reading traditional books is just elitism. With audiobooks,
    podcasts, and interactive media, there are more ways to absorb information than
    ever. Restricting knowledge to printed texts is outdated and frankly absurd.
  - Comprehensive sex education is essential for saving lives. Teaching kids about
    consent, protection, and healthy relationships leads to wiser choices. Any school
    district that rejects this curriculum is letting down their students and communities.
- source_sentence: Disposable plastic is destroying our planet and poisoning our bodies.
    Every piece ever made still exists somewhere, choking wildlife and filling our
    bloodstreams with microplastics. We must ban it completely NOW.
  sentences:
  - Single-use plastic is harming our environment and contaminating our bodies. Every
    piece ever produced still lingers, suffocating animals and infiltrating our systems
    with microplastics. It's imperative that we implement a total ban immediately.
  - The convenience and practicality of plastic products greatly enhance our daily
    lives, and imposing a total ban would be impractical and unnecessary. Wildlife
    and human health are not significantly threatened by plastics.
  - The government shouldn't allocate funds to arts programs at all. Let the free
    market determine what's worthwhile. My taxes should support infrastructure, defense,
    and necessary services, not poetry classes.
- source_sentence: Part of me thinks this makes sense for supporting families, but
    I also worry about putting too much burden on small businesses that are already
    struggling financially.
  sentences:
  - I partially agree that this approach could benefit families, though I'm also concerned
    about potentially overburdening small businesses that are facing financial challenges.
  - Supporting families shouldn't come at the cost of ignoring the crucial needs of
    small enterprises, which would benefit from reduced financial pressure.
  - This perspective is regressive. Capitalism fosters economic liberty and disperses
    authority, which are essential for democracy. The wealthiest and most democratic
    nations are capitalist, while socialism has historically undermined democratic
    systems.
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
    'Part of me thinks this makes sense for supporting families, but I also worry about putting too much burden on small businesses that are already struggling financially.',
    "I partially agree that this approach could benefit families, though I'm also concerned about potentially overburdening small businesses that are facing financial challenges.",
    "Supporting families shouldn't come at the cost of ignoring the crucial needs of small enterprises, which would benefit from reduced financial pressure.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9290, 0.7883],
#         [0.9290, 1.0000, 0.7830],
#         [0.7883, 0.7830, 1.0000]])
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
  | details | <ul><li>min: 17 tokens</li><li>mean: 41.55 tokens</li><li>max: 137 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 46.05 tokens</li><li>max: 163 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 39.14 tokens</li><li>max: 129 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                    | positive                                                                                                                                                                                                                                                                  | negative                                                                                                                                                                                                                                                                                                               |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Unlicensed drugs exist for good reason - they're unproven and potentially harmful. Allowing imports would create a dangerous black market preying on desperate families with false promises.</code>                                                 | <code>Unlicensed medications exist for legitimate reasons - they are untested and potentially dangerous. Permitting imports would lead to a hazardous underground market exploiting vulnerable families with misleading assurances.</code>                                | <code>Importing experimental treatments could provide desperate families with new hope, expanding access to potentially life-saving therapies that have not yet been approved.</code>                                                                                                                                  |
  | <code>Impfungen haben definitiv viele Krankheiten besiegt, das ist unbestritten. Gleichzeitig verstehe ich die Sorgen mancher Leute bezüglich Nebenwirkungen. Jeder sollte das Recht haben, selbst zu entscheiden, was mit seinem Körper passiert.</code> | <code>Impfstoffe haben ohne Zweifel viele Krankheiten besiegt, das steht außer Frage. Gleichzeitig kann ich die Bedenken bezüglich möglicher Nebenwirkungen nachvollziehen. Jeder sollte die Freiheit haben, selbst zu bestimmen, was mit seinem Körper geschieht.</code> | <code>Es ist wichtig, dass Impfungen nicht übertrieben werden, da die Risiken und Nebenwirkungen oft unterschätzt werden. Der Schutz vor Krankheiten kann auch durch natürliche Immunität erreicht werden, und niemand sollte gezwungen werden, medizinische Behandlungen zu akzeptieren, die sie nicht wollen.</code> |
  | <code>Roads are environmental disasters that destroy habitats, increase emissions, and promote car dependency. We should be removing lanes and converting them to green space, not paving over more of our planet.</code>                                 | <code>Roadways are catastrophic for the environment, as they fragment ecosystems, elevate pollution, and foster reliance on vehicles. Instead of expanding asphalt surfaces, we should transform them into green areas.</code>                                            | <code>Expanding road networks is essential for economic growth, providing crucial infrastructure that supports jobs and connects communities. We should focus on adding more lanes to ease traffic congestion.</code>                                                                                                  |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 1.25e-05
- `max_steps`: 125
- `warmup_steps`: 12
- `seed`: 46

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
- `seed`: 46
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
| 0.4444 | 50   | 0.6206        |
| 0.8889 | 100  | 0.5984        |


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