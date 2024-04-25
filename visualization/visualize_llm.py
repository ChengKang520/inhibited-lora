
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
import os
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import json5
from threading import Thread
from typing import Iterator, Optional

from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TextIteratorStreamer,
)

logger = logging.getLogger()
transformers.logging.set_verbosity_error()

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default="/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/Llama-2-7b-chat-hf/final_checkpoints",
    )
    quantize: Optional[bool] = field(default=False)
    task: Optional[str] = field(
        default="squad_v2",
    )
    output_csv_file: Optional[str] = field(default="/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/Llama-2-7b-chat-hf/results.csv")


def get_model_and_tokenizer(
    model_name: str,
    adapter_name: str,
    tokenizer_name: Optional[str] = None,
    quantize: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    if adapter_name is not None:
        model = PeftModel.from_pretrained(model, adapter_name, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name
    )

    return model, tokenizer

# def get_input_token_length(
#     tokenizer: AutoTokenizer,
#     message: str,
#     chat_history: list[tuple[str, str]],
#     system_prompt: str,
# ) -> int:
#     prompt = get_prompt(message, chat_history, system_prompt)
#     input_ids = tokenizer([prompt], return_tensors="np", add_special_tokens=False)[
#         "input_ids"
#     ]
#     return input_ids.shape[-1]



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)

model, tokenizer = get_model_and_tokenizer(
    model_name=script_args.model_name,
    adapter_name=script_args.adapter_name,
    tokenizer_name=script_args.tokenizer_name,
    quantize=script_args.quantize,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

input_text = "I put my red bag in the black bag ."
output_text = "What is the colour of my bag ?"
query = f"""
        Use the following context to answer the question. Think step by step and explain your reasoning.
        Context:
        \"\"\"
        {input_text}
        \"\"\"
        
        Question:
        \"\"\"
        {output_text}
        \"\"\"
        """
# print("#############################################")
# print(query)
# print("#############################################")


inputs = tokenizer(query, return_tensors="pt").to(device)

# print('#####################################')
# print(inputs['input_ids'][0])
# print('#####################################')

select_mode = ["inhibition_no"]
task = script_args.task  #  "squad_v2"

visualize_file = "/home/kangchen/inhibited_lora/visualization/" + task + "/" + script_args.model_name + "/"

for plot_mode in range(len(select_mode)):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # print("#############################################")
    # print(outputs)
    # print("#############################################")

    attention_scores_draw = outputs['attentions']  # Retrieve attention from model outputs

    layers_num = len(attention_scores_draw)

    attention_scores_plot = torch.zeros(inputs['input_ids'].size()[1], inputs['input_ids'].size()[1], layers_num)

    for plot_layer in range(len(attention_scores_draw)):
        attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
        ## Plot Attention Scores
        attention_scores_plot[:, :, plot_layer] = torch.squeeze(torch.mean(attention_heads, 0))

    ##  *************************************************************
    # print('#####################################')
    attention_scores_plot = torch.squeeze(attention_scores_plot).detach().numpy()
    attention_scores_size = attention_scores_plot.shape
    # print(attention_scores_size)

    for plot_layer in range(layers_num):
        ## Plot Attention Scores
        # for plot_head in range(heads_num):

        plot_head = "average"
        file_name = "layer_" + str(plot_layer) + "_head_" + str(plot_head)
        path = os.path.join(visualize_file, select_mode[plot_mode])

        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

        attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
        attention_heads_size = attention_heads.size()

        text_plot = []
        for i_token in range(len(inputs['input_ids'][0])):
            text_plot.append(tokenizer.decode(inputs['input_ids'][0][i_token]))

        plot_0 = plt
        fig = plot_0.figure()
        imgplot = plot_0.imshow(attention_scores_plot[:, :, plot_layer], cmap='BuGn')  # , vmin=-1.0, vmax=5.0
        plot_0.xticks(np.arange(0, len(text_plot)), text_plot, rotation='vertical')
        plot_0.yticks(np.arange(0, len(text_plot)), text_plot, rotation='horizontal')
        plot_0.colorbar(orientation='vertical')
        plot_0.show()
        save_file = path + '/' + file_name + '.png'
        # print(save_file)
        plot_0.savefig(save_file)
        plot_0.close()
        # print('#####################################')

    generate_ids = model.generate(inputs.input_ids, max_length=512)
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # answer_start_index = outputs.start_logits.argmax()
    # answer_end_index = outputs.end_logits.argmax()
    # predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    # output_text = tokenizer.decode(predict_answer_tokens)
    print(output_text)


