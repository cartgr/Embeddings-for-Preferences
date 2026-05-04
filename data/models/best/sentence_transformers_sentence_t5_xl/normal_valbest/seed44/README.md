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
- source_sentence: Generally people know their own situations best. Adding barriers
    might help some couples reconcile, but it could also force people to stay in harmful
    relationships longer. The potential for increased abuse and trauma outweighs the
    benefits.
  sentences:
  - Trading corporate data seems to dive too deeply into the realm of surveillance
    capitalism. While targeted advertisements can occasionally be beneficial, the
    invasion of privacy outweighs these benefits. We urgently require much more stringent
    regulations on what data can be exchanged.
  - Increased relationship barriers are essential as they encourage couples to work
    through their issues, reducing the likelihood of separation and providing stability,
    which ultimately outweighs any potential downsides.
  - People usually understand their own circumstances best. While implementing obstacles
    might help some partners mend their relationships, it could also result in individuals
    remaining in damaging relationships for longer periods. The risk of heightened
    abuse and trauma surpasses any potential benefits.
- source_sentence: This is fascinating but impossible to define universally. Nordic
    welfare states work for some cultures, while others thrive with American-style
    capitalism. Geography, history, and values all shape what works.
  sentences:
  - This topic is intriguing yet universally defining it is challenging. Nordic welfare
    systems suit certain cultures, whereas others flourish with the capitalism seen
    in America. The effectiveness depends on geography, history, and societal values.
  - American-style capitalism should be adopted globally, as it provides a blueprint
    for success and prosperity, unlike the Nordic welfare approach which often leads
    to economic stagnation.
  - Government redistribution discourages achievement and encourages mediocrity. Inequality
    is a result of varied levels of effort, talent, and personal decisions. The state
    should safeguard property rights rather than seize wealth from hardworking individuals.
- source_sentence: Parental school choice is destroying public education and creating
    a two-tiered system based on wealth. Education is a public good that requires
    collective investment, not a private commodity for individual consumption.
  sentences:
  - A part of me appreciates the concept of all children dining together without any
    embarrassment or obstacles, yet another part questions whether those funds could
    benefit more kids by providing superior teachers or reducing class sizes.
  - Allowing parents to choose schools is undermining public education and creating
    a wealth-based divide. Education should be seen as a collective investment and
    public asset, not treated as a commodity for individual use.
  - Allowing families to select schools promotes competition and innovation, which
    improves education overall. Treating education as a private choice empowers families
    and leads to better outcomes for everyone.
- source_sentence: Rooney's boycott sends an important message about Palestinian rights,
    though I wish she'd found a way to reach Israeli readers who might actually challenge
    their government's policies from within.
  sentences:
  - A part of me believes that investing in education should be prioritized, yet I
    also understand that the actual expenses of lending need to be covered. Perhaps
    income-based interest rates might be a viable solution?
  - Rooney's refusal to engage is a significant statement regarding Palestinian rights,
    although I hope she could have found a method to communicate with Israeli audiences
    who might initiate change within their own government.
  - Rooney's boycott is counterproductive, as it alienates Israeli readers who are
    key to promoting change and advancing Palestinian rights through dialogue and
    understanding.
- source_sentence: The two-state concept sounds neat on paper but ignores too many
    realities on the ground. Security concerns, water rights, Jerusalem - these issues
    might be better addressed through confederation or autonomy arrangements.
  sentences:
  - The idea of two states looks appealing on paper but overlooks numerous ground
    realities. Issues like security, water rights, and Jerusalem might be better handled
    with confederation or autonomy solutions.
  - Tax havens are ethically corrupt strategies that allow the affluent to shirk their
    civic duties while others fund infrastructure like roads, schools, and hospitals.
    These loopholes need to be completely eradicated without any exceptions.
  - Establishing two separate states is crucial for addressing the complex issues
    of security, water distribution, and the status of Jerusalem, ensuring long-term
    peace and stability.
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
    'The two-state concept sounds neat on paper but ignores too many realities on the ground. Security concerns, water rights, Jerusalem - these issues might be better addressed through confederation or autonomy arrangements.',
    'The idea of two states looks appealing on paper but overlooks numerous ground realities. Issues like security, water rights, and Jerusalem might be better handled with confederation or autonomy solutions.',
    'Establishing two separate states is crucial for addressing the complex issues of security, water distribution, and the status of Jerusalem, ensuring long-term peace and stability.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9757, 0.6945],
#         [0.9757, 1.0000, 0.6972],
#         [0.6945, 0.6972, 1.0000]])
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
  |         | anchor                                                                            | positive                                                                            | negative                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | string                                                                            |
  | details | <ul><li>min: 17 tokens</li><li>mean: 41.4 tokens</li><li>max: 90 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 46.13 tokens</li><li>max: 102 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 39.53 tokens</li><li>max: 97 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                     | positive                                                                                                                                                                                                                                                                                                                       | negative                                                                                                                                                                                                                                                                                             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Scapegoating immigrants is classic fear-mongering. They revitalize dying towns, start businesses, pay taxes, and do jobs that keep our economy running. The real problem is politicians manufacturing crises instead of fixing our broken immigration system.</code> | <code>Blaming immigrants is typical fear-mongering. They breathe new life into struggling communities, create businesses, contribute taxes, and perform essential jobs that sustain our economy. The true issue lies with politicians who generate fake crises instead of addressing our flawed immigration system.</code>     | <code>Politicians have no choice but to address the chaos caused by unchecked immigration. Instead of improving communities, immigrants burden public services and take away jobs, worsening the economy. The immigration system needs stricter enforcement, not liberal reforms.</code>             |
  | <code>Higher earners should contribute more to the NHS, though £200k seems arbitrary. We need sustainable funding, but this should be part of broader tax reform rather than a quick fix that might drive talent abroad.</code>                                            | <code>Those with higher incomes should give more to support the NHS, although setting the threshold at £200k seems random. We need enduring funding solutions, but this should be incorporated into a comprehensive tax overhaul rather than a hasty measure that might cause skilled individuals to leave the country.</code> | <code>Flat contributions to the NHS are fairer and prevent targeting high earners, ensuring talent remains in the country. Instead of overhauling tax systems, maintaining a balanced approach without drastic changes secures consistent funding without risking an exodus of professionals.</code> |
  | <code>Demonstrations without proper permits and oversight are just chaos waiting to happen. Public safety trumps everything else - we need strict licensing, time limits, designated zones, and immediate shutdown authority for authorities.</code>                       | <code>Protests without the necessary permits and supervision are simply disasters in the making. Ensuring public safety is the highest priority, which necessitates rigorous licensing, specific time frames, assigned areas, and the ability for authorities to shut down events immediately.</code>                          | <code>Spontaneous demonstrations are essential expressions of freedom, and imposing licensing, time restrictions, or specific zones stifles that. Public safety can be maintained without excessive control, allowing people to voice their opinions freely and openly.</code>                       |
* Loss: <code>src.embedding.trainer.BradleyTerryLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 1.25e-05
- `max_steps`: 125
- `warmup_steps`: 12
- `seed`: 44

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
| 0.4444 | 50   | 0.6206        |
| 0.8889 | 100  | 0.5985        |


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