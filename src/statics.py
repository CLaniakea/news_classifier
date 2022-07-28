"""
统计最大最小字符号
"""
import pandas as pd
import pdb

def read_data(file_path):
    df = pd.read_csv(file_path, sep='\t').values.tolist()
    char_set=set()
    ml=0
    mi=float('inf')
    for i,(_,char) in enumerate(df):
        if i%1000==0: print(i)
        # pdb.set_trace()
        cl = char.split()
        ml=max(ml,len(cl))
        mi=min(mi, len(cl))
        for ch in cl:
            char_set.add(int(ch))
    print(max(char_set), min(char_set), len(char_set),ml,mi)

read_data('../data_sets/train_set.csv')
# 7549 0 6869
# 7549 0 6203