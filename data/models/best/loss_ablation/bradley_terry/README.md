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
- source_sentence: The bill has some solid provisions for reducing unnecessary litigation,
    but I worry the caps might be too restrictive for cases involving genuine negligence.
    We need tort reform, just maybe not quite this aggressive.
  sentences:
  - The bill has some solid provisions for reducing unnecessary litigation, but I
    am confident the caps are restrictive enough for cases involving genuine negligence.
    We need tort reform, and this approach is appropriately aggressive.
  - Investing in universal healthcare doesn't seem as critical as supporting families
    and individuals who need assistance with raising children, taking time off for
    new parents, and accessing fertility options.
  - While the proposed measures could help curb frivolous lawsuits, there is concern
    that the imposed limits might hinder fair compensation for victims of true malpractice.
    Reform is necessary, but perhaps a more measured approach is warranted.
- source_sentence: This renewable fantasy ignores basic physics and economics. Wind
    and solar are intermittent toys that can't power modern civilization. Fossil fuels
    built our prosperity and remain absolutely essential.
  sentences:
  - The stability and growth of our society have been largely fueled by traditional
    energy sources like coal, oil, and natural gas. These resources are vital for
    maintaining the infrastructure and economy we rely on daily, as alternatives like
    wind and solar lack the consistent output needed for a fully functioning industrial
    world.
  - This renewable fantasy embraces fundamental physics and economics. Wind and solar
    are reliable solutions that can power modern civilization. Fossil fuels built
    our past but are no longer essential.
  - Economically, aligning with our former partners would be advantageous, and the
    connections we reestablish could be crucial. However, we must tread carefully
    to ensure we don't compromise the autonomy we've worked hard to achieve. It might
    be possible to strike a deal that allows access to the market while safeguarding
    our decision-making power.
- source_sentence: Government monopolies are always wasteful disasters. Private prisons
    bring accountability, innovation, and cost control that bureaucrats never could.
    We need more privatization, not less.
  sentences:
  - The widespread use of online platforms for positive interactions demonstrates
    that blaming social media for fostering violence shifts attention from deeper
    societal issues. Violent behavior predates digital communication, indicating that
    technology is not the root cause of such acts.
  - State-run systems are often inefficient and ineffective. In contrast, private
    correctional facilities can drive improvements through competition and economic
    incentives. Increasing privatization could lead to better management and reduced
    costs.
  - Government monopolies are always wasteful disasters. Private prisons lack accountability,
    innovation, and cost control that bureaucrats could provide. We need less privatization,
    not more.
- source_sentence: While I appreciate the democratic idealism behind sortition, elected
    officials at least have some accountability to voters. Random selection might
    reduce corruption, but it also removes the crucial link between representatives
    and constituents.
  sentences:
  - The concept of randomly selecting government officials undermines the essential
    accountability and connection they must have with the people they serve, despite
    its potential to decrease corrupt practices.
  - Immediate and substantial efforts toward sustainable energy sources are critical,
    as delaying action will jeopardize future generations and exacerbate environmental
    issues.
  - While I appreciate the democratic idealism behind sortition, random selection
    might reduce corruption, but elected officials at least have some accountability
    to voters. It also removes the crucial link between representatives and constituents.
- source_sentence: Public accountability has value for addressing serious misconduct,
    but cancel culture often goes too far with disproportionate consequences. Minor
    mistakes shouldn't destroy careers, and we're losing the ability to have productive
    conversations about difficult topics.
  sentences:
  - While it's crucial to hold individuals accountable for serious wrongdoing, the
    trend of canceling people for small errors can lead to excessive punishment. We
    risk shutting down meaningful dialogue on sensitive issues by reacting too harshly
    to minor faults.
  - Industrial agriculture's genetic modifications prioritize short-term profits over
    long-term ecological health. Our reliance on these engineered crops threatens
    natural ecosystems and our well-being. Embracing natural farming methods is essential
    for a sustainable future.
  - Public accountability has value for addressing serious misconduct, but cancel
    culture is justified with appropriate consequences. Minor mistakes should indeed
    destroy careers, and we're gaining the ability to have productive conversations
    about difficult topics.
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
    "Public accountability has value for addressing serious misconduct, but cancel culture often goes too far with disproportionate consequences. Minor mistakes shouldn't destroy careers, and we're losing the ability to have productive conversations about difficult topics.",
    "While it's crucial to hold individuals accountable for serious wrongdoing, the trend of canceling people for small errors can lead to excessive punishment. We risk shutting down meaningful dialogue on sensitive issues by reacting too harshly to minor faults.",
    "Public accountability has value for addressing serious misconduct, but cancel culture is justified with appropriate consequences. Minor mistakes should indeed destroy careers, and we're gaining the ability to have productive conversations about difficult topics.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9215, 0.6512],
#         [0.9215, 1.0000, 0.6458],
#         [0.6512, 0.6458, 1.0000]])
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
  | details | <ul><li>min: 27 tokens</li><li>mean: 41.79 tokens</li><li>max: 137 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 44.67 tokens</li><li>max: 78 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 42.65 tokens</li><li>max: 139 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                             | positive                                                                                                                                                                                                                                                                                                                                                   | negative                                                                                                                                                                                                                                                                                                    |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Supporting local businesses makes sense for jobs and the environment, though I admit some products are just better or cheaper when imported from specialized regions.</code>                                                                                                 | <code>There's a strong case for backing small-scale enterprises to boost employment and reduce environmental impact, although I recognize that certain goods are often superior or more affordable when sourced from their areas of expertise.</code>                                                                                                      | <code>Supporting local businesses seems appealing for jobs and the environment, but I believe most products are actually better or cheaper when imported from specialized regions.</code>                                                                                                                   |
  | <code>Public safety has to come first, though obviously we can't have a complete police state. Some privacy invasion is unfortunate but necessary if it prevents terrorism and serious crime.</code>                                                                               | <code>To safeguard citizens, it's crucial to accept some surveillance measures, even if it means sacrificing a bit of privacy, as it helps avert terror threats and major offenses.</code>                                                                                                                                                                 | <code>Public safety must not come first, as a police state is the only way to ensure it. Privacy invasion is necessary for preventing terrorism and serious crime.</code>                                                                                                                                   |
  | <code>Socialist principles like universal healthcare and strong worker protections clearly benefit society, though we'd need to maintain some market mechanisms for innovation. Pure socialism might stifle entrepreneurship, but democratic socialism offers real promise.</code> | <code>Implementing universal healthcare and robust labor rights can significantly improve societal welfare. However, it's essential to preserve certain aspects of capitalism to foster innovation, as a completely socialist approach could hinder new business ventures. Instead, a blend of socialism and democracy holds substantial potential.</code> | <code>Socialist principles like universal healthcare and strong worker protections may appeal to some, but they ultimately harm society, as we need to avoid market mechanisms for innovation. Pure socialism might encourage entrepreneurship, and democratic socialism doesn't offer real promise.</code> |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `learning_rate`: 0.000125
- `max_steps`: 46
- `warmup_steps`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
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