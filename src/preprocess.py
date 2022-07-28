import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import pdb

def load_data(file_name):
    data = pd.read_csv(file_name, sep='\t').values.tolist()
    ret=[]
    l=[]
    for d in data:
        # d[1]=[int(_) for _ in d[1].split()]
        # l.append(len(d[1]))
        # if len(d[1])>=512: d[1]=d[1][:512]
        # else:d[1]+=[7550]*(512-len(d[1]))
        # d=d[1]+[d[0]]
        # ret.append(d)

        d=[int(_) for _ in d[0].split()]
        l.append(len(d))
        if len(d)>=512: d=d[:512]
        else:d+=[7550]*(512-len(d))
        ret.append(d)
    print(np.mean(np.array(l)))
    return ret

def data_split(data,test_ratio):
    np.random.seed(42)
    np.random.shuffle(data)
    test_set_size = int(len(data) * test_ratio)
    return data[test_set_size:],data[:test_set_size]

def save2disk(data, file_name):
    # pdb.set_trace()
    data=np.array(data, dtype=np.int)
    a=torch.from_numpy(data)
    torch.save(a, file_name)

# data=load_data('../data_sets/train_set.csv')
# train_data,test_data=data_split(data, 0.1)
# save2disk(train_data, '../data_sets/train_set.pt')
# save2disk(test_data, '../data_sets/eval_set.pt')

data=load_data('../data_sets/test_a.csv')
save2disk(data, '../data_sets/test_a.pt')


# pdb.set_trace()

# save2disk(test_data)

