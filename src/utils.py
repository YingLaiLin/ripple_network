
import pandas as pd
import networkx as nx
import numpy as np
import config
def load_data(filename, relations=None):
    df = pd.read_csv(filename)
    if relations is not None:
        df = df[df['label'].isin(relations)]
    return df.user, df.item, df.label

def read_network(filename, relations=None):
    df = pd.read_csv(filename)
    if relations is not None:
        # relation discrete
        df = df[df['label'].isin(relations)]
    
    label_types = list(pd.unique(df['label']))    
    if 0 not in relations:
        relations.insert(0,0)
    if 0 not in label_types:
        label_types.insert(0,0) # 默认 0 为没有关系
    
    label_mapping = dict(zip(label_types, range(len(relations))))
    df['label'] = df['label'].apply(lambda x: label_mapping[x])
    # save mapping
    f = open("data/relation_mapping.txt","w")
    f.write(str(label_mapping))
    f.close()
    
    graph = nx.from_pandas_edgelist(df, source='source', 
                                        target='target', edge_attr='label')
    graph.add_nodes_from(np.arange(config.num_nodes))      
    return graph,label_mapping


    
