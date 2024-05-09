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



- `BERT-large`
    ```bash
    $ cd LoRA-LM/
    $ sbatch squad_BERT.batch
    ```

<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer0_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer1_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer2_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer3_average.png"  align=center > |
|21| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer20_average.png"  align=center > |
|22| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer21_average.png"  align=center > |
|23| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer22_average.png"  align=center > |
|24| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer23_average.png"  align=center > |






- `RoBERTa-large`
    ```bash
    $ cd LoRA-LM/
    $ sbatch squad_RoBERTa.batch
    ```
<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer0_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer1_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer2_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer3_average.png"  align=center > |
|21| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer20_average.png"  align=center > |
|22| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer21_average.png"  align=center > |
|23| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer22_average.png"  align=center > |
|24| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer23_average.png"  align=center > |







- `Llama2-7B` needs 2 GPUs
    ```bash
    $ cd LoRA-LM/
    $ sbatch squad_llama-2.batch
    ```
<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer0_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer1_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer2_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer3_average.png"  align=center > |
|21| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer20_average.png"  align=center > |
|22| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer21_average.png"  align=center > |
|23| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer22_average.png"  align=center > |
|24| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer23_average.png"  align=center > |






