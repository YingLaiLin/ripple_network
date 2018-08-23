
import pandas as pd
import networkx as nx

def load_data(filename):
    df = pd.read_csv(filename)
    return df.user, df.item, df.label

def read_network(filename, relations=None):
    df = pd.read_csv(filename)
    if relations is not None:
        df = df[df['label'].isin(relations)]
    graph = nx.from_pandas_edgelist(df, source='source', 
                                        target='target', edge_attr='label')
    return graph


if __name__ == '__main__':
    graph = read_network("data/network.txt",[1,2])
    print(graph)
    
    
    