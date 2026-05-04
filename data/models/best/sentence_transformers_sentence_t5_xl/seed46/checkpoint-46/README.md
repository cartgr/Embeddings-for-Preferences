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
- source_sentence: This decision really cuts both ways. Sudan's new government deserves
    sovereignty over their transition, but the security situation remains fragile.
    Hope this doesn't backfire on vulnerable civilians.
  sentences:
  - The state should not interfere in the personal bonds formed by consenting adults,
    as doing so constitutes a grave infringement on their fundamental rights. Penalizing
    same-sex relationships is an outdated and cruel practice that causes immense harm
    to individuals and their loved ones.
  - While it's crucial for Sudan to have control over its own political future, the
    delicate security conditions raise concerns about the potential risks faced by
    the country's most at-risk populations.
  - This decision really cuts both ways. Sudan's new government deserves sovereignty
    over their transition, but the security situation remains fragile. Hope this ultimately
    strengthens the position of vulnerable civilians.
- source_sentence: Government travel bans to dangerous regions are absolutely necessary.
    Citizens become hostages requiring expensive rescues, create diplomatic incidents,
    and put military personnel at risk during evacuations. Sometimes protecting people
    from their own poor judgment is essential.
  sentences:
  - Government travel bans to dangerous regions are not necessary at all. Citizens
    rarely become hostages needing costly rescues, seldom create diplomatic incidents,
    and do not put military personnel at risk during evacuations. Protecting people
    from their own poor judgment is not essential.
  - I'm torn on this topic. While I find the idea of spiritual equality quite persuasive,
    I also acknowledge that many religious customs have profound doctrinal justifications
    that deserve careful consideration.
  - It's crucial to implement restrictions on travel to high-risk areas. This helps
    prevent situations where civilians might end up in perilous circumstances, leading
    to costly missions and political tensions. Sometimes, intervening to prevent individuals
    from making risky decisions is vital for their safety.
- source_sentence: Charity begins at home, and we've got plenty of poverty in our
    own communities. International aid often gets wasted by corrupt governments anyway.
    Help locally where you can see real impact.
  sentences:
  - Our local communities face significant needs that we should address before looking
    beyond our borders. Investing in domestic poverty alleviation ensures that our
    contributions are effective and visible, unlike international efforts that often
    fall prey to governmental corruption.
  - Charity begins at home, and we've got plenty of poverty in our own communities.
    International aid is essential and should be prioritized, as it often bypasses
    corrupt governments. Focus abroad where change is most needed.
  - Imposing universal regulations undermines the unique needs of different regions.
    Local governments should have the flexibility to decide what's best for their
    communities, as agricultural zones and city centers face distinct challenges.
- source_sentence: The plastic crisis is destroying our planet and we're running out
    of time. Every piece of packaging should be biodegradable by law - no exceptions,
    no delays. Our children's future depends on bold action now.
  sentences:
  - The overwhelming presence of non-degradable plastics is a ticking time bomb for
    our environment. It's crucial that legislation mandates eco-friendly packaging
    solutions immediately to safeguard the world our descendants will inherit.
  - The plastic crisis isn't as severe as some claim, and we have plenty of time.
    Not every piece of packaging should be biodegradable by law - some exceptions
    and delays are necessary. Our children's future doesn't depend on drastic action
    now.
  - Ensuring everyone has the ability to connect to the internet is essential, yet
    constantly being connected might not be beneficial. Taking time away from screens
    could be advantageous for our well-being.
- source_sentence: Churchill's statue celebrates a white supremacist who caused deliberate
    famine in Bengal and fought to preserve the British Empire's racist exploitation.
    Parliament Square should honor leaders who actually believed in human equality.
  sentences:
  - Hamas's involvement in violent acts like bombings and assaults on non-combatants
    justifies their classification as a terrorist entity, despite their contributions
    to community support.
  - Figures who upheld ideals of racial superiority and contributed to colonial oppression
    should not be commemorated in public spaces. Instead, we should recognize those
    who championed equality and justice for all.
  - Churchill's statue celebrates a great leader who helped preserve the British Empire's
    noble endeavors. Parliament Square should honor leaders who supported the values
    of their time.
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
    "Churchill's statue celebrates a white supremacist who caused deliberate famine in Bengal and fought to preserve the British Empire's racist exploitation. Parliament Square should honor leaders who actually believed in human equality.",
    'Figures who upheld ideals of racial superiority and contributed to colonial oppression should not be commemorated in public spaces. Instead, we should recognize those who championed equality and justice for all.',
    "Churchill's statue celebrates a great leader who helped preserve the British Empire's noble endeavors. Parliament Square should honor leaders who supported the values of their time.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6329, 0.6311],
#         [0.6329, 1.0000, 0.7207],
#         [0.6311, 0.7207, 1.0000]])
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
  |         | anchor                                                                             | positive                                                                           | negative                                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                              |
  | details | <ul><li>min: 25 tokens</li><li>mean: 41.47 tokens</li><li>max: 97 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 44.87 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 42.25 tokens</li><li>max: 110 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                | positive                                                                                                                                                                                                                                     | negative                                                                                                                                                                                                                                                  |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Agricultural regulation is absolutely essential for protecting our communities and environment. We can't have farmers growing invasive species that destroy ecosystems or crops that contaminate groundwater. Public safety comes first.</code> | <code>It's crucial to ensure that farming practices don't harm ecosystems or pollute our water supplies. Implementing strict controls on agriculture helps safeguard both the environment and public health.</code>                          | <code>Agricultural regulation is absolutely detrimental for protecting our communities and environment. We can't have farmers growing diverse species that enhance ecosystems or crops that purify groundwater. Public safety does not come first.</code> |
  | <code>Polygamy should be legal between consenting adults, though I worry about tax complications and inheritance disputes. The freedom to choose your family structure matters more than bureaucratic convenience.</code>                             | <code>The right to decide how to form one's own family should not be obstructed by government intervention, despite potential issues related to taxation and estate matters.</code>                                                          | <code>Polygamy should NOT be legal between consenting adults, even if I understand the concerns about tax complications and inheritance disputes. Bureaucratic convenience matters more than the freedom to choose your family structure.</code>          |
  | <code>Every species lost is gone forever. Rural industries can adapt, evolve, or relocate, but extinction is permanent. We have a moral obligation to protect biodiversity at all costs.</code>                                                       | <code>The irreversible loss of animal and plant species demands urgent attention. While rural economies have the flexibility to change, the disappearance of biodiversity is final, urging us to take decisive action to preserve it.</code> | <code>Every species lost is gone forever. Rural industries can adapt, evolve, or relocate, but extinction is permanent. We have a moral obligation to prioritize economic growth over biodiversity protection.</code>                                     |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `gradient_accumulation_steps`: 2
- `learning_rate`: 0.000125
- `max_steps`: 46
- `warmup_steps`: 4
- `seed`: 46

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