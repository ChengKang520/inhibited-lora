# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train QLoRA with PEFT

# %% [markdown]
# ## Imports

# %%
# Basic Imports
import time
import pathlib
import pickle

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
# Basic Model Imports
import peft
import bitsandbytes as bnb
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from tqdm.auto import tqdm

# Import Own Modules
import lora_utils
from glue_squad_utils import *
from typing import Optional


# %% [markdown]
# ## Settings


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    batch_size: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_inhibition: Optional[float] = field(default=0.3)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    num_warmup_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./Output_PEFT",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    resume_from_checkpoint: Optional[str] = field(default=None)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# %%
output_path = pathlib.Path(script_args.output_dir)
output_path.mkdir(exist_ok=True, parents=False)

model_id = script_args.model_name  # "roberta-base" # "roberta-base" or "roberta-large"

output_path = output_path / model_id
output_path.mkdir(exist_ok=True, parents=False)

squad_v2_na_boost_factor = 2.1  # reduce weight of NA loss

# Config to load the quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.use_4bit,
    bnb_4bit_use_double_quant=script_args.use_nested_quant,
    bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=['classifier', 'qa_outputs'],
    # classifier for GLUE or qa_outputs for SQuAD, not to be quantized as these are new matrices
)

# Config for the LoRA Injection via PEFT
peft_config = peft.LoraConfig(
    r=script_args.lora_r,  # dimension of the LoRA injected matrices
    lora_alpha=script_args.lora_alpha,
    # parameter for scaling, use 8 here to make it comparable with our own implementation
    lora_inhibition=script_args.lora_inhibition,
    lora_dropout=script_args.lora_dropout,  # dropout probability for layers
    # target_modules=['query', 'key', 'value', 'intermediate.dense', 'output.dense'], # be precise about dense because classifier has dense too
    target_modules=['query', 'key', 'value'],
    modules_to_save=["LayerNorm", "classifier", "qa_outputs"],
    # Retrain the layer norm; classifier is the fine-tune head; qa_outputs is for SQuAD
    bias="none",  # none, all, or lora_only
)

# %% [markdown]
# ## Setup & Test QLoRA RoBERTa-Model

# %% [markdown]
# First we will load the base model to show the parameter names

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# When importing the quanitzed model version this number is actually wrong, as all the quantized weights no longer show
parameter_count_orig = lora_utils.count_parameters(model)
print(f"Parameter count, before LoRA adjustments: {lora_utils.format_with_underscore(parameter_count_orig)}")
print("")

lora_utils.parameter_count_table(model, output_file_path=(output_path / "pre_lora_parameters.txt"), output_print=False,
                                 add_dtypes=True)

# %%
# Import the model with quantization
model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype="auto", quantization_config=bnb_config)

# %%
# Check that the model was indeed loaded in 4 bits
# print(f"The following print should show Linear4bit elements in the attention:")
# print(model.roberta.encoder.layer[4].attention)

# print(f"This here should show a uint8 value:")
# print(model.roberta.encoder.layer[4].attention.self.query.weight.dtype)

# %%
lora_utils.parameter_count_table(model, output_file_path=(output_path / "pre_lora_parameters_quantized.txt"),
                                 output_print=False, add_dtypes=True, show_nograd_paras=True)

# %%
# Test if the quantized model works
try:
    x = torch.randint(low=1, high=30_000, size=(16, 21))
    y = model(x)
    print("The quantized model works")
except Exception as e:
    print("The quantized model doesn't work yet")
    print(e)

# %%
# Inject the LoRA into the parameters
model = peft.get_peft_model(model, peft_config)

# When importing the quanitzed model version this number is actually wrong, as all the quantized weights no longer show
parameter_count_lora = lora_utils.count_parameters(model)
print(f"Parameter count, after LoRA injection: {lora_utils.format_with_underscore(parameter_count_lora)}")
print(f"Percentage of parameters to be trained compared to orig: {(parameter_count_lora / parameter_count_orig):.4%}")
print("")

lora_utils.parameter_count_table(model, output_file_path=(output_path / "after_qlora_parameters.txt"),
                                 output_print=False, add_dtypes=True)
lora_utils.parameter_count_table(model, output_file_path=(output_path / "after_qlora_parameters_all.txt"),
                                 output_print=False, add_dtypes=True, show_nograd_paras=True)

# %%
# Test if the quantized model works
try:
    x = torch.randint(low=1, high=30_000, size=(16, 21))
    y = model(x)
    print("The full QLoRA model works")
except Exception as e:
    print("The full QLoRA model doesn't work yet")
    print(e)


# %% [markdown]
# ## Define Model Wrapper for Fine-Tuning

# %%
# Helper functions for the training to initialize model

def initialize_model(task, model_id, nr_classes):
    if task in glue_task_list:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=nr_classes,
                                                                   quantization_config=bnb_config)
        model = peft.get_peft_model(model, peft_config)
        return model
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_id, quantization_config=bnb_config)
        model = peft.get_peft_model(model, peft_config)
        return model


# %%
# Show the model components after setup
# Test GLUE
model = initialize_model(model_id=model_id, task='cola', nr_classes=2)

try:
    x = torch.randint(low=1, high=30_000, size=(16, 21))
    y = model(x)
    print("The full QLoRA model with finetune-head works for GLUE")
except Exception as e:
    print("The full QLoRA model with finetune-head doesn't work yet for GLUE")
    print(e)

print('TABLE FOR GLUE')
lora_utils.parameter_count_table(model, output_file_path=(output_path / "final_check_init_function_glue_all.txt"),
                                 output_print=False, add_dtypes=True, show_nograd_paras=True)
lora_utils.parameter_count_table(model, output_file_path=(output_path / "final_check_init_function_glue.txt"),
                                 output_print=True, add_dtypes=True, show_nograd_paras=False)

# Test SQuAD
model = initialize_model(model_id=model_id, task='squad_v1', nr_classes=2)

try:
    x = torch.randint(low=1, high=30_000, size=(16, 21))
    y = model(x)
    print("The full QLoRA model with finetune-head works for SQuAD")
except Exception as e:
    print("The full QLoRA model with finetune-head doesn't work yet for SQuAD")
    print(e)

print('TABLE FOR SQUAD')
lora_utils.parameter_count_table(model, output_file_path=(output_path / "final_check_init_function_squad_all.txt"),
                                 output_print=False, add_dtypes=True, show_nograd_paras=True)
lora_utils.parameter_count_table(model, output_file_path=(output_path / "final_check_init_function_squad.txt"),
                                 output_print=True, add_dtypes=True, show_nograd_paras=False)

# %%
del x, y

# %% [markdown]
# ## GLUE and SQUAD Training

# %%
train_task_list_loop = train_task_list

# %%
# Keep tokenizer around, such that we can initialize the task loader first and get the number of classes, before initializing the model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

# %%
# Training loop
for task in ['squad_v2']:  # train_task_list_loop:
    print(f"\n\nRunning for Task: {task}")
    print("==============================\n")

    # 0. Setup Task Loader and Training objects
    task_loader = GlueSquadTaskLoader(tokenizer=tokenizer, task=task, batch_size=script_args.batch_size)
    nr_batches = task_loader.get_nr_batches()
    print(f"Nr batches: {nr_batches}")
    nr_classes = task_loader.get_nr_classes()
    print(f"Nr classes: {nr_classes}")

    # 1. Initialize model
    del model
    model = initialize_model(task, model_id, nr_classes)
    model = model.to(device)

    nr_total_epochs = (script_args.num_warmup_epochs + script_args.num_train_epochs)

    val_test_keys = [s for s in task_loader.dataset.keys() if
                     'val' in s or 'test' in s]  # -> test is always empty for the GLUE tasks

    # Define Optimizer and Schedules
    # For the quantized model we have to use a special version of the AdamW from bits and bytes
    adamW_parameters_finetune = {
        "lr": 4e-5,
        "betas": (0.9, 0.98),
        "eps": 1e-06,
        "weight_decay": 0.1,
    }

    optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], **adamW_parameters_finetune)

    loss_list = []

    warm_up_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.000000001, end_factor=1.0,
                                                         total_iters=(nr_batches * script_args.num_warmup_epochs),
                                                         last_epoch=-1, verbose=False)
    main_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                      total_iters=(nr_batches * script_args.num_train_epochs),
                                                      last_epoch=-1, verbose=False)

    ####################################################
    ## Training                                       ##
    ####################################################

    # dict to save results
    res_dict = {
        'epoch': []
    }

    model.to(device)

    if task == "stsb":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    print("Starting model training")

    for epoch in range(nr_total_epochs):
        print(f"Running for epoch {epoch}")
        res_dict['epoch'].append(epoch)

        # Train loop
        model.train()

        predictions = []  # also calc the in-training metric
        if task in ["squad", "squad_v1", "squad_v2"]:
            predictions = {'id': [], 'prediction_text': []}
            if task == "squad_v2":
                predictions['no_answer_probability'] = []
        tqdm_looper = tqdm(task_loader.epoch_iterator(), total=nr_batches)

        for raw_batch in tqdm_looper:

            optimizer.zero_grad()

            input_tensor = raw_batch['input_ids'].clone().detach().to(device).long()
            attention_mask = raw_batch['attention_mask'].clone().detach().to(device).long()

            if task in ["squad", "squad_v1", "squad_v2"]:
                start_positions = raw_batch['start_positions'].long().clone().detach().to(device)
                end_positions = raw_batch['end_positions'].long().clone().detach().to(device)
                if task == "squad_v2":
                    non_answerable = raw_batch['non_answerable'].float().to(device)
            elif task == "stsb":
                target_labels = raw_batch['label'].clone().detach().to(torch.float16).to(device)
            else:  # glue, non stsb
                target_labels = raw_batch['label'].clone().detach().to(device).long()

            if task in ["squad", "squad_v1", "squad_v2"]:
                output_tensor = model(input_tensor, attention_mask=attention_mask)
                # Extract the outputs
                start_logits = output_tensor['start_logits']
                end_logits = output_tensor['end_logits']
                if task == "squad_v2":
                    na_prob_logits = output_tensor['start_logits'][:, 0] + output_tensor['end_logits'][:, 0]

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                if task == "squad_v2":
                    loss_fct_cls = torch.nn.BCEWithLogitsLoss()
                    cls_loss = loss_fct_cls(na_prob_logits, non_answerable)
                    total_loss += cls_loss * squad_v2_na_boost_factor
                loss = total_loss
            else:
                output_tensor = model(input_tensor, attention_mask=attention_mask)['logits']
                loss = criterion(output_tensor, target_labels)

            loss.backward()
            optimizer.step()

            # Steps for schedule
            if epoch < script_args.num_warmup_epochs:
                warm_up_schedule.step()
            else:
                main_schedule.step()

            with torch.no_grad():
                if task == "stsb":
                    predictions += output_tensor.clone().detach().squeeze().tolist()
                elif task in ["squad", "squad_v1", "squad_v2"]:
                    start_positions = start_logits.clone().detach().argmax(-1).tolist()
                    end_positions = end_logits.clone().detach().argmax(-1).tolist()
                    # If end position is before the start position set it to the start position to extract an empty string
                    end_positions = [e if e >= s else s for s, e in zip(start_positions, end_positions)]

                    # Reconstruct answer and check
                    recon_answer_tokens = [ts[start:end] for ts, start, end in
                                           zip(raw_batch['input_ids'], start_positions, end_positions)]
                    recon_answer = task_loader.tokenizer.batch_decode(recon_answer_tokens, skip_special_tokens=True,
                                                                      clean_up_tokenization_spaces=True)
                    predictions['id'] += raw_batch['id']
                    predictions['prediction_text'] += recon_answer
                    if task == "squad_v2":
                        na_pred_probabilities = torch.sigmoid(na_prob_logits.clone().detach()).tolist()
                        predictions['no_answer_probability'] += na_pred_probabilities
                else:
                    predictions += output_tensor.clone().detach().argmax(-1).tolist()

            loss_list.append(loss.item())
            tqdm_looper.set_description(
                f"loss: {np.mean(loss_list[-100:]):.6f}; lr: {lora_utils.get_lr(optimizer):.8f}")

        # Evaluate with Metrics for Train
        k = "train"
        if task in ["squad", "squad_v1", "squad_v2"]:
            references_dict = [
                {'id': id, 'answers': answer}  # , 'no_answer_threshold': 0.5
                for id, answer in zip(task_loader.shuffled_dataset['id'], task_loader.shuffled_dataset['answers'])
            ]
            if task == "squad_v2":
                predictions_dict = [
                    {'id': id, 'prediction_text': prediction_text, 'no_answer_probability': no_answer_probability}
                    for id, prediction_text, no_answer_probability in
                    zip(predictions['id'], predictions['prediction_text'], predictions['no_answer_probability'])
                ]
            else:
                predictions_dict = [
                    {'id': id, 'prediction_text': prediction_text}
                    for id, prediction_text in zip(predictions['id'], predictions['prediction_text'])
                ]
            res_score = task_loader.metric.compute(predictions=predictions_dict, references=references_dict)
        else:
            res_score = task_loader.metric.compute(predictions=predictions,
                                                   references=task_loader.shuffled_dataset['label'])
        print(f"score for {k}: {res_score}")
        for score_key, score_value in res_score.items():
            k_add = f"{k}_{score_key}"
            if k_add not in res_dict:
                res_dict[k_add] = []
            res_dict[k_add].append(score_value)

        ####################################################
        ## Validation & Test Loop                         ##
        ####################################################
        with torch.no_grad():
            for k in val_test_keys:
                model.eval()
                predictions = []  # also calc the in-training metric
                if task in ["squad", "squad_v1", "squad_v2"]:
                    predictions = {'id': [], 'prediction_text': []}
                    if task == "squad_v2":
                        predictions['no_answer_probability'] = []

                tqdm_looper = tqdm(task_loader.epoch_iterator(split_type=k),
                                   total=task_loader.get_nr_batches(split_type=k))

                for raw_batch in tqdm_looper:

                    # Eval load input data
                    input_tensor = raw_batch['input_ids'].clone().detach().to(device).long()
                    attention_mask = raw_batch['attention_mask'].clone().detach().to(device).long()

                    if task in ["squad", "squad_v1", "squad_v2"]:
                        start_positions = raw_batch['start_positions'].long().clone().detach().to(device)
                        end_positions = raw_batch['end_positions'].long().clone().detach().to(device)
                        if task == "squad_v2":
                            non_answerable = raw_batch['non_answerable'].float().to(device)
                    elif task == "stsb":
                        target_labels = raw_batch['label'].clone().detach().to(device).float()
                    else:  # glue, non stsb
                        target_labels = raw_batch['label'].clone().detach().to(device).long()

                    # Eval model predictions
                    if task in ["squad", "squad_v1", "squad_v2"]:
                        output_tensor = model(input_tensor, attention_mask=attention_mask)
                        # Extract the outputs
                        start_logits = output_tensor['start_logits']
                        end_logits = output_tensor['end_logits']
                        if task == "squad_v2":
                            na_prob_logits = output_tensor['start_logits'][:, 0] + output_tensor['end_logits'][:, 0]
                    else:
                        output_tensor = model(input_tensor, attention_mask=attention_mask)['logits']

                    # Eval save model predictions
                    if task == "stsb":
                        predictions += output_tensor.clone().detach().squeeze().tolist()
                    elif task in ["squad", "squad_v1", "squad_v2"]:
                        start_positions = start_logits.clone().detach().argmax(-1).tolist()
                        end_positions = end_logits.clone().detach().argmax(-1).tolist()
                        # If end position is before the start position set it to the start position to extract an empty string
                        end_positions = [e if e >= s else s for s, e in zip(start_positions, end_positions)]

                        # Reconstruct answer and check
                        recon_answer_tokens = [ts[start:end] for ts, start, end in
                                               zip(raw_batch['input_ids'], start_positions, end_positions)]
                        recon_answer = task_loader.tokenizer.batch_decode(recon_answer_tokens, skip_special_tokens=True,
                                                                          clean_up_tokenization_spaces=True)
                        predictions['id'] += raw_batch['id']
                        predictions['prediction_text'] += recon_answer
                        if task == "squad_v2":
                            na_pred_probabilities = torch.sigmoid(na_prob_logits.clone().detach()).tolist()
                            predictions['no_answer_probability'] += na_pred_probabilities
                    else:
                        predictions += output_tensor.clone().detach().argmax(-1).tolist()

                # Eval apply metric
                if task in ["squad", "squad_v1", "squad_v2"]:
                    references_dict = [
                        {'id': id, 'answers': answer}  # , 'no_answer_threshold': 0.5
                        for id, answer in
                        zip(task_loader.shuffled_dataset['id'], task_loader.shuffled_dataset['answers'])
                    ]
                    if task == "squad_v2":
                        predictions_dict = [
                            {'id': id, 'prediction_text': prediction_text,
                             'no_answer_probability': no_answer_probability}
                            for id, prediction_text, no_answer_probability in
                            zip(predictions['id'], predictions['prediction_text'], predictions['no_answer_probability'])
                        ]
                    else:
                        predictions_dict = [
                            {'id': id, 'prediction_text': prediction_text}
                            for id, prediction_text in zip(predictions['id'], predictions['prediction_text'])
                        ]
                    res_score = task_loader.metric.compute(predictions=predictions_dict, references=references_dict)
                elif "test" not in k:
                    # Don't run this for GLUE problems with test, as it is not defined
                    res_score = task_loader.metric.compute(predictions=predictions,
                                                           references=task_loader.shuffled_dataset['label'])
                else:
                    res_score = {"not_calculated": np.nan}
                print(f"score for {k}: {res_score}")
                for score_key, score_value in res_score.items():
                    k_add = f"{k}_{score_key}"
                    if k_add not in res_dict:
                        res_dict[k_add] = []
                    res_dict[k_add].append(score_value)

    df_res = pd.DataFrame(res_dict)
    df_res.to_csv(output_path / f"results_{task}.csv")

    # save model
    model.save_pretrained((output_path / f"model_{task}.pth"))

    # save loss list
    with open((output_path / f"model-{task}-loss_list.pkl"), 'wb') as handle:
        pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nBest Results:")
    print(df_res.max())
    print('-' * 42)

