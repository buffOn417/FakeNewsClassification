from torch.utils.data import DataLoader
from data2bert import FakeNewsDataset,create_mini_batch,processData
from model import get_predictions,build_model
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
import torch
from train import output_dir,device
def load_model():
    """
    Load and quantize the fine-tuned BERT model with PyTorch
    :return: Your own model
    """
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    
    model.to(device)
    return model,tokenizer

def load_bert_params(PATH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_model = torch.load(PATH,map_location='cpu')
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 3
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    config = BertConfig.from_pretrained("bert-base-chinese",num_labels = NUM_LABELS)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME,config = config)
    model.load_state_dict(torch_model)
    model.to(device)
    return model,tokenizer
if __name__ == '__main__':
    #processData()
    model, tokenizer = load_bert_params("./model/finetune/bert_params.pth")

    #testset = FakeNewsDataset("test", tokenizer=tokenizer)
    #testloader = DataLoader(testset,batch_size = 8, collate_fn = create_mini_batch)
    #print("testloader type",type(testloader))
    #print("testloader",testloader)


    
    predictions = get_predictions(model,"萨拉赫人气爆棚!埃及总统大选未参选获百万选票 现任总统压力山大	辟谣！里昂官方否认费基尔加盟利物浦，难道是价格没谈拢？	321187")
    print("predictions",predictions)
    
    #index_map = {v : k for k, v in testset.label_map.items()}

    
    #df = pd.DataFrame({"Category":predictions.tolist()})
    #df['Category'] = df.Category.apply(lambda x:index_map[x])
    #df_pred = pd.concat([testset.df.loc[:,["Id"]],
    #                     df.loc[:,'Category']],axis = 1)
    #df_pred.to_csv('bert_1_prec_training_samples.csv',index = False)
    #df_pred.head()



