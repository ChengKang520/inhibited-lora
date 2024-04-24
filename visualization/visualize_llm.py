



import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser

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


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default="results/final_checkpoints",
    )
    quantize: Optional[bool] = field(default=False)
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )
    output_csv_file: Optional[str] = field(default="results/results.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=1)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


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


def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


def get_input_token_length(
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ]
    return input_ids.shape[-1]


def run(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


def extract_answer(text):
    text = text[text.find("{") :]
    text = text[: text.find("}") + 1]
    try:
        # JSON5 is a little less picky than JSON
        answer = json5.loads(text)["answer"]
    except:
        answer = None
    return answer

def get_answer(prompt, pipeline):
    response = ""
    while True:
        instruction = prompt.find("[/INST] ")
        if instruction == -1:
            break
        instruction += len("[/INST] ")
        current_prompt = response.strip()
        current_prompt += prompt[:instruction] + "</s>"
        logger.debug("Instruction: %s", prompt[:instruction])
        prompt = prompt[instruction:]
        prompt = prompt[prompt.find("<s>") :]
        response = pipeline(
            current_prompt,
            do_sample=False,
            num_beams=script_args.num_beams,
            num_return_sequences=1,
            max_new_tokens=512,
        )[0]["generated_text"]
        logger.debug("Response: %s", response[len(current_prompt) :].strip())

    response = response[len(current_prompt) :].strip()
    return extract_answer(response), response



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


with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Context",
            "Question",
            "Correct answers",
            "Model answer",
            "Full response",
            "Exact match",
        ]
    )

    dataset = load_from_disk(script_args.dataset)["test"]
    if script_args.shuffle:
        dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.num_samples is not None:
        dataset = dataset.select(range(script_args.num_samples))

    for text in tqdm(dataset["text"]):
        answer_start = text.rfind("```json")
        prompt = text[:answer_start]
        answers = extract_answer(text[answer_start:])
        context = prompt[prompt.find("Context: ") + 9 : prompt.find("Question: ") - 1]
        logger.debug("Context: %s", context)
        question = prompt[prompt.find("Question: ") + 10 : prompt.find("[/INST] ")]
        question = question[: question.find("[/INST]")]
        logger.debug("Question: %s", question)
        logger.debug("Correct answers: %s", answers)
        model_answer, full_response = get_answer(prompt, pipeline)
        logger.debug("Model answer: %s", model_answer)
        exact_match = model_answer is not None and model_answer in answers

        writer.writerow(
            [
                context,
                question,
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()

