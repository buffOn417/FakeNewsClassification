#载入一个可以做中文多分类任务的模型, n_class = 3

from transformers import BertForSequenceClassification
import torch
from bp.data2bert import FakeNewsDataset, create_mini_batch,tokenizer
#from pytorch_transformers.modeling_bert import BertPreTrainedModel
from torch.utils.data import DataLoader
PRETRAINED_MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model():
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS
    )
    # high level 显示此模型里的modules
    print("""
        name            module
        ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))
            # 使用Adam Optimizer 更新整个分类模型的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # lr = learning_rate
    model = model.to(device)
    return model,optimizer

def get_predictions(model,dataloader,compute_acc = False):
    predictions = None
    correct = 0
    total = 0
    with torch.no_grad():
        #遍巡整个资料集
        for data in dataloader:
            #将所有tensors移动到gpu上面
            print("model.parameters().is_cuda? ： ", next(model.parameters()).is_cuda)
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

                #别忘记前三个tensors分别为tokens，segements以及masks
                #且强烈建议在这些tensors丢入model时指定对应的参数名称
            tokens_tensors, segments_tensors,masks_tensors = data[:3]
            print("tokens_tensors: ",tokens_tensors)
            print("segments_tensors: ",segments_tensors)
            print("masks_tensors: ",masks_tensors)
            outputs = model(input_ids= tokens_tensors,
                                token_type_ids = segments_tensors,
                                attention_mask = masks_tensors)
            logits = outputs[0]
            _,pred = torch.max(logits.data,1)
            print("data: ",data)

            #用来计算训练集的分类准确率
            if compute_acc :
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                print("labels: ",data[3])
                print("total: ",total)
                print("correct: ",correct)

                #将当前batch记录下来
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions,pred))
            print("predictions: ",predictions)
        if compute_acc:
            acc = correct/total
            print("compute_acc: ",compute_acc)
            print("total: ",total)
            return predictions, acc
        print("compute_acc: ",compute_acc)

        return predictions
#计算整个分类模型以及里面的简单分类器有多少个参数：
def get_learnable_params(module):
        return [p for p in module.paramters() if p.requires_grad]

if __name__ == '__main__':
    #让模型跑起来（但是没有gpu在本地）并取得训练集的分类准确率
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)
    model = build_model()
    trainset = FakeNewsDataset("train", tokenizer=tokenizer)
    print("Trainset :", trainset)
    BATCH_SIZE = 64
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             collate_fn=create_mini_batch)
    print("Train Loader: ",trainloader)
    _,acc = get_predictions(model,trainloader,compute_acc= True)
    print("classification acc: ",acc)

    model_params = get_learnable_params(model)
    clf_params = get_learnable_params(model.classifier)
    print(f"""
    整个分类模型的参数量：{sum(p.numel() for p in model_params)}
    线性分类器的参数量： {sum(p.numel() for p in clf_params)}
    """)


