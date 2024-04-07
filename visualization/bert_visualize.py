
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# select_mode = ["inhibition_no", "inhibition_00", "inhibition_10", "inhibition_30", "inhibition_90", "inhibition_99"]
select_mode = ["inhibition_no"]

parent_dir = '/home/kangchen/Quantized_Adaptor/Figures/'
layers_num = 24
heads_num = 16

for plot_mode in range(len(select_mode)):

    model_file = "/home/kangchen/Quantized_Adaptor/visualize_result/BERT/" + select_mode[plot_mode] + "/mrpc/"
    tokenizer = BertTokenizer.from_pretrained(model_file)
    model = BertForQuestionAnswering.from_pretrained(model_file)

    input_text = ["Shares of Xoma fell 16 percent in early trade , while shares of Genentech , a much larger company with several products on the market , were up 2 percent ."]
    output_text = ["Shares of Genentech , a much larger company with several products on the market , rose more than 2 percent ."]




    input_text_split = input_text[0].split()
    output_text_split = output_text[0].split()
    print('#####################################')
    print(input_text_split)
    print(len(input_text_split))
    print(len(output_text_split))
    print(len(input_text_split) + len(output_text_split))
    print('#####################################')
    # input_text = input_text.split()

    input_length = len(input_text)
    output_length = len(output_text)
    attention_scores_plot = torch.zeros(input_length, output_length, layers_num, heads_num)

    for i_inputs in range(input_length):
        for i_outputs in range(output_length):

            # question, text = "I put my red bag in the black bag.", "What is the colour of my bag?"
            question, text = input_text[i_inputs], output_text[i_outputs]
            inputs = tokenizer(question, text, return_tensors="pt")

            # print('#####################################')
            # print(inputs['input_ids'].size())
            # print('#####################################')



            # print(attention_scores_plot.size())

            with torch.no_grad():
                attention_scores_draw, outputs = model(**inputs)

    #         for plot_layer in range(layers_num):
    #             attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
    #             attention_heads_size = attention_heads.size()
    #
    #             ## Plot Attention Scores
    #             for plot_head in range(heads_num):
    #                 # print('#####################################')
    #                 # print(torch.squeeze(attention_heads[plot_head, :, :]).size())
    #                 # print('#####################################')
    #
    #                 # print(torch.squeeze(attention_heads[plot_head, :, :]).size())
    #                 attention_scores_plot[i_inputs, i_outputs, plot_layer, plot_head] = torch.mean(torch.squeeze(attention_heads[plot_head, :, :]))
    #
    # ##  *************************************************************
    # # print('#####################################')
    # attention_scores_plot = torch.squeeze(attention_scores_plot).detach().numpy()
    # attention_scores_size = attention_scores_plot.shape
    # # print(attention_scores_size)
    #
    # for plot_layer in range(layers_num):
    #
    #     ## Plot Attention Scores
    #     for plot_head in range(heads_num):
    #
    #         file_name = "layer_" + str(plot_layer) + "_head_" + str(plot_head)
    #         path = os.path.join(parent_dir, select_mode[plot_mode])
    #
    #         isExist = os.path.exists(path)
    #         if not isExist:
    #             # Create a new directory because it does not exist
    #             os.makedirs(path)
    #             print("The new directory is created!")
    #
    #         attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
    #         attention_heads_size = attention_heads.size()
    #
    #         plot_0 = plt
    #         fig = plot_0.figure()
    #         imgplot = plot_0.imshow(attention_scores_plot[:, :, plot_layer, plot_head], cmap='RdBu')  #  , vmin=-1.0, vmax=5.0
    #         plot_0.xticks(np.arange(0, len(output_text)), output_text, rotation='vertical')
    #         plot_0.yticks(np.arange(0, len(input_text)), input_text, rotation='horizontal')
    #         plot_0.colorbar(orientation='vertical')
    #         plot_0.show()
    #         save_file = path + '/' + file_name + '.png'
    #         # print(save_file)
    #         plot_0.savefig(save_file)
    #         plot_0.close()
    #     # print('#####################################')




        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        output_text = tokenizer.decode(predict_answer_tokens)
        print(output_text)

        ## *********************************************
        ## Plot Loss
        loss = outputs.loss
        print(loss)





