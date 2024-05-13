#  InA: Inhibition Adaption on Pre-Trained Language Models



This repository contains the code necessary to reproduce Shunting Inhibition on LoRA, the shunting inhibition mechanism that controls passed information from previous layers introduced in the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4551993):



## Installation

1. Install requirements

```bash
$ pip install -r requirements.txt
$ pip install -e peft-0.10.0/
```



## Visualization of Average Attention Heatmap 

Firstly, we should finetune the RoBERTa-large on SquAD v2.0 using different inhibition levels. 


- `RoBERTa-large`
    ```bash
    $ cd LoRA-LM/
    $ sbatch squad_RoBERTa.batch
    ```
<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_0_head_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_1_head_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_2_head_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_3_head_average.png"  align=center > |
|21| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer20_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_20_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_20_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_20_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_20_head_average.png"  align=center > |
|22| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer21_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_21_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_21_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_21_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_21_head_average.png"  align=center > |
|23| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer22_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_22_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_22_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_22_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_22_head_average.png"  align=center > |
|24| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer23_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_00/layer_23_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_10/layer_23_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_30/layer_23_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/FacebookAI/roberta-large/squad_v2/Inhibition_90/layer_23_head_average.png"  align=center > |







- `Llama2-7B` needs 2 GPUs
    ```bash
    $ cd LoRA-LM/
    $ sbatch squad_llama-2.batch
    ```
<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_0_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_0_head_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_1_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_1_head_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_2_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_2_head_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_3_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_3_head_average.png"  align=center > |
|29| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer28_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_28_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_28_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_28_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_28_head_average.png"  align=center > |
|30| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_29_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_29_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_29_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_29_head_average.png"  align=center > |
|31| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_30_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_30_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_30_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_30_head_average.png"  align=center > |
|32| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_00/layer_31_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_10/layer_31_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_30/layer_31_head_average.png"  align=center > |  <img style="height:100px" src="./Output_PEFT/Llama-2-7b-chat-hf/squad_v2/Inhibition_90/layer_31_head_average.png"  align=center > |





