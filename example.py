import torch
from transformers import BertTokenizer,BertModel

from transformers import BertForMaskedLM
from IPython.display import display
import sys
from bertviz import head_view
PRETRAINED_MODEL_NAME = "bert-base-chinese"
model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME, output_attentions=True)
#取得预训练此模型使用的tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
#clear_output()
print("PyTorch版本： ", torch.__version__)

#拿出tokenize里面的字典
vocab = tokenizer.vocab
#看看字典大小
print("字典大小：" , len(vocab))

"""
这段代码带入已经训练好的masked语言模型，
对有[MASK]的句子做预测
"""




def WhatIsNextMask(ids):
    # 除了tokens意外我们还需要辨别句子的segment ids

    tokens_tensor = torch.tensor([ids])  # （1,seq_len）
    segments_tensors = torch.zeros_like(tokens_tensor)
    maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
    #clear_output()

    #使用masked LM预测[MASK] 位置所代表的实际token
    maskedLM_model.eval()
    with torch.no_grad():
        outputs = maskedLM_model(tokens_tensor,segments_tensors)
        predictions = outputs[0]
    del maskedLM_model

    #将[MASK]位置的分布概率取top k最有可能的tokens出来
    masked_index = 5
    k = 3
    probs,indices = torch.topk(torch.softmax(predictions[0,masked_index],-1),k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

    #显示top k可能的字， 一般就是给top 1 当做预测值
    print("输入tokens: ", tokens[:10])
    print('-'*50)
    for i, (t,p) in enumerate(zip(predicted_tokens,probs),1):
        tokens[masked_index] = t
        print("Top {} ({:2}%) :{}".format(i,int(p.item()*100),tokens[:10]),'...')


def call_html():
    import IPython
    display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
                  jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
                },
              });
            </script>
            '''))

def getAttnVis(s1,s2):
    inputs = tokenizer.encode_plus(s1,s2,return_tensors='pt',
                                   add_special_tokens=True
                                   )
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    attention = model(input_ids,token_type_ids= token_type_ids)[-1]
    input_id_list = input_ids[0].tolist() #batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    call_html()

    #交给BertViz视觉化, 这个head有指代消解能力，能正确的关注“他”所指代的对线
    head_view(attention,tokens)


if __name__ == '__main__':
    text = "[CLS]等到潮水[MASK]了，就知道谁没穿裤子。"
    tokens = tokenizer.tokenize(text)
    print(tokens, '...')
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(text)
    print(ids[:10],'...')
    WhatIsNextMask(ids)

    #Top 1 (73%) :['[CLS]', '等', '到', '潮', '水', '来', '了', '，', '就', '知'] ...
    #Top 2 ( 3%) :['[CLS]', '等', '到', '潮', '水', '到', '了', '，', '就', '知'] ...
    #Top 3 ( 3%) :['[CLS]', '等', '到', '潮', '水', '干', '了', '，', '就', '知'] ...
    #Bert 通过潮与水 从2w多个字的可能性中跳出来“来”作为这个情景下[MASK] token的预测值

    ######################################################################



    #不同情景的句子
    sentence_a = "胖虎叫大雄去买漫画，"
    sentence_b = "回来慢了就打他。"

    #得到tokens丢入BERT取得attention
    getAttnVis(sentence_a,sentence_b)