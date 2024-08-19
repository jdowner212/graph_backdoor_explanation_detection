from   utils.config import *
import networkx as nx
import pickle
import torch
from   torch_geometric.data import Data
import torch_geometric.datasets as datasets
from   torch_geometric.transforms import BaseTransform
from   torch_geometric.utils import degree


class GraphObject(object):
    def __init__(self, nx_graph, label, node_features=[],edge_features=[],edge_index=[],is_backdoored=False, original_index=None, triggered_edges=[]):
        '''
            nx_graph: a networkx graph
            pyg_graph: a torch_geometric graph
        '''
        self.nx_graph = nx_graph
        self.pyg_graph = Data(x=node_features, edge_attr=edge_features, edge_index=edge_index, y=label, is_backdoored=is_backdoored, original_index=original_index, triggered_edges=triggered_edges)


class DeduplicateEdges(BaseTransform):
    def __call__(self, data):
        edge_tuples = set(tuple(sorted(edge)) for edge in data.edge_index.t().tolist())
        unique_edges = torch.tensor(list(edge_tuples), dtype=torch.long).t()
        data.edge_index = unique_edges
        return data
    

def read_pyg_data(dataset,cleaned=False,use_edge_attr=False):
    g_list = []
    nx_graph_list = []
    label_dict = {}
    if dataset=='DBLP':
        with open(f'{src_dir}/DBLP_data_list_random_5000.pkl','rb') as f:
            pyg_dataset = pickle.load(f)
    elif dataset != 'BGS':
        pyg_dataset = datasets.TUDataset(root=f'/tmp/{dataset}', name=dataset,cleaned=cleaned,use_edge_attr=use_edge_attr)
    n_g = len(pyg_dataset)
    for i in range(n_g):
        pyg_graph = pyg_dataset[i]
        edge_index = pyg_graph.edge_index.cpu().numpy()
        g = nx.Graph()
        g.add_nodes_from(range(pyg_graph.num_nodes))  # This ensures all nodes are added, including isolated ones
        for i in range(edge_index.shape[1]):
            g.add_edge(edge_index[0, i], edge_index[1, i])
        nx_graph_list.append(g)
    ''' cap degree at chosen maximum (currently all equal to true max degree value)'''
    all_degrees = [degree(data.edge_index[0], data.num_nodes) for data in pyg_dataset]
    max_degree = data_shape_dict[dataset]['max_degree']
    for i,g in enumerate(nx_graph_list):
        pyg_graph   = pyg_dataset[i]
        degrees     = torch.as_tensor(all_degrees[i], dtype=torch.int64)
        exceeds_max = degrees>max_degree
        degrees[exceeds_max] = max_degree 
        n_rows, n_cols = len(degrees), max_degree+1
        x = torch.zeros(n_rows, n_cols)
        x.scatter_(1, degrees.unsqueeze(1),1)
        pyg_graph.x = x
        try:
            pyg_graph.y = pyg_graph.y.item()
        except:
            pass
        edge_index = torch.tensor(list(g.edges())).T
        l = pyg_graph.y
        assert pyg_graph.y is not None
        if not l in label_dict:
            mapped = len(label_dict)
            label_dict[l] = mapped
        try:
            assert edge_index.max() <= len(pyg_graph.x)
        except:
            print('issue with graph',i)
        this_g = GraphObject(nx_graph=g, label=l, node_features=pyg_graph.x, edge_index=edge_index, original_index=i, triggered_edges=[])
        g_list.append(this_g)
    return g_list, label_dict


def load_data(dataset, 
              data_type='pyg',
              cleaned=False,
              use_edge_attr=False):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    assert data_type == 'text' or data_type=='pyg', "dataset must be type 'text' or 'pyg'"
    if data_type=='pyg':
        g_list, label_dict = read_pyg_data(dataset,cleaned=cleaned,use_edge_attr=use_edge_attr)
        max_degree = g_list[0].pyg_graph.x.shape[1]
        tag2index = {i: i for i in range(max_degree)}
    return g_list, tag2index


def is_fully_connected(graph):
    if graph.is_directed():
        raise ValueError("The graph should be undirected.")
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    max_edges = num_nodes * (num_nodes - 1) / 2
    return num_edges == max_edges


def get_all_possible_edges(data):
    num_nodes = data.x.size(0)
    all_edges = torch.combinations(torch.arange(num_nodes), r=2)
    all_edges = [tuple(edge.numpy()) for edge in all_edges]
    all_edges = [(i,j) for (i,j) in all_edges if i!=j]
    return all_edges