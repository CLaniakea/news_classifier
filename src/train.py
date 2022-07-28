import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import news_classifier
import torch.utils.data as Data
import pdb
from tqdm import tqdm
from sklearn.metrics import f1_score
torch.manual_seed(42)
torch.cuda.manual_seed(42)
ls_fun=nn.CrossEntropyLoss()
pad=torch.Tensor([7550]).cuda()
def load_data(file_name):
    return torch.load(file_name)

def test():
    test_data=load_data('../data_sets/test_a.pt')
    vocab_size=7551 # 7550作为padding 
    batch_size=128
    in_dim=512
    num_cls=14
    num_layers=2
    num_heads=8
    # model = news_classifier(vocab_size, in_dim, in_dim, num_cls, num_layers, num_heads)
    model=torch.load('./models.pt')
    model=model.cuda()
    test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    preds=[]
    for input_data in tqdm(test_loader):
        input_data=input_data.cuda()
        out=model(input_data, input_data==pad)
        preds.append(torch.argmax(out, 1).detach().cpu())
    preds=torch.cat(preds)
    preds=list(preds)
    df=pd.DataFrame(preds)
    df.to_csv('./test_a.csv', index=False,header=['label'])


def eval(model,eval_loader):
    eval_loss=[]
    preds=[]
    labels=[]
    model.eval()

    bestf1=-1.0
    bestauc=-1.0
    for i,eval_data in enumerate(tqdm(eval_loader)):
        input_data, label=eval_data[:,:-1], eval_data[:, -1]
        input_data=input_data.cuda()
        label=label.cuda()

        # pdb.set_trace()
        out=model(input_data, input_data==pad)
        # print(label.shape)
        loss=ls_fun(out, label)
        eval_loss.append(loss.detach().cpu().item())
        preds+=[torch.argmax(out, 1).detach().cpu()]
        labels+=[label.detach().cpu()]
        # print(preds,eval_loss,labels)
        # time.sleep(1)
    # pdb.set_trace()
    preds=torch.cat(preds)
    labels=torch.cat(labels)

    f1=f1_score(labels, preds, average='macro')
    if f1>bestf1:
        bestf1=f1
        torch.save(model, './models.pt')
    model.train()
    return eval_loss, f1

def train(file_name):
    lr=1e-4
    vocab_size=7551 # 7550作为padding
    epochs = 50
    batch_size=128
    max_len=512
    in_dim=512
    num_cls=14
    num_layers=2
    num_heads=8

    train_data, eval_data= load_data(file_name+'/train_set.pt'), load_data(file_name+'/eval_set.pt')
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = Data.DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=0)

    model=news_classifier(vocab_size, in_dim, in_dim, num_cls, num_layers, num_heads)
    model=model.cuda()
    opt=torch.optim.Adam(model.parameters(), lr)
    print('start')
    loss=0.0
    for epoch in range(epochs):
        for i,train_data in enumerate(tqdm(train_loader)):
            input_data, label=train_data[:,:-1], train_data[:, -1]
            
            input_data=torch.LongTensor(input_data).cuda()
            label=label.cuda()
            out=model(input_data, input_data==pad)
            #print(out.shape)
            loss=ls_fun(out, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
        eval_loss,f1=eval(model, eval_loader)
        print('epoch: {}, loss {:.5f}, f1 {:.5f}'.format(epoch, np.mean(eval_loss), f1))
        
    # model = news_classifier(vocab_size, in_dim, in_dim, num_cls, num_layers, num_heads)
# train('../data_sets/')
test()
