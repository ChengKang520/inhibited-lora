---
library_name: peft
language:
- en
license: apache-2.0
base_model: bert-large-cased
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
model-index:
- name: InA90
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: GLUE RTE
      type: glue
      args: rte
    metrics:
    - type: accuracy
      value: 0.5487364620938628
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# InA90

This model is a fine-tuned version of [bert-large-cased](https://huggingface.co/bert-large-cased) on the GLUE RTE dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6912
- Accuracy: 0.5487

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.53.0
- Pytorch 2.7.1+cu126
- Datasets 4.0.0
- Tokenizers 0.21.4