
import pandas as pd
import networkx as nx
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    return df.user, df.item, df.label

def read_network(filename, relations=None):
    df = pd.read_csv(filename)
    label_types = list(pd.unique(df['label']))
    if 0 not in label_types:
        label_types.insert(0,0) # 默认 0 为没有关系
    label_mapping = dict(zip(label_types, range(len(relations))))
    # save mapping
    f = open("data/relation_mapping","w")
    f.write(str(label_mapping))
    f.close()
    # pd.DataFrame(np.array(label_mapping),index=np.arange(len(relations))).to_csv("data/relation_mapping.csv",index=None)
    df['label'].apply(lambda x: label_mapping[x])
    if relations is not None:
        df = df[df['label'].isin(relations)]
    graph = nx.from_pandas_edgelist(df, source='source', 
                                        target='target', edge_attr='label')
                                   
    return graph

read_network('data/network.txt', [0,1,3])