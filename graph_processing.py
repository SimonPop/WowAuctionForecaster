import numpy as np
import json
from random import randint
from tqdm import tqdm
import os.path
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import networkx as nx

def get_item_dict_from_xml():
    """
    Creates a dictionary id -> raw_xml for each item in Reagent directory.
    """
    item_dict = {}
    item_files = [f for f in listdir('./data/Reagents') if isfile(join('./data/Reagents', f))] 
    for item_path in item_files:
        item_id = item_path[5:-4]
        item_dict[item_id] = _load_item(item_path)
    return item_dict

def _get_reagents_from_xml(raw_xml, item_id):
    """
    Get all reageants for all recipes of one item.
    """
    soup = BeautifulSoup(raw_xml, "lxml")
    level = int(soup.find('level').text)
    # Get item name
    name = soup.find('b', class_ = 'q1')
    if name is None:
        name = soup.find('b', class_ = 'q2')
    if name is None:
        name = soup.find('b', class_ = 'q3')
    name = name.text
    # Get each recipe (spell)
    createdby = soup.find('createdby')
    if createdby is None:
        return  (name, [], level)
    spells = createdby.find_all('spell')
    # Get each reagent in each recipe
    reagent_lists = [spell.find_all('reagent') for spell in spells]
    return (name, reagent_lists, level)

def create_graph_from_dict(item_dict, min_level = 100):
    """
    Create a networkx graph from an item dict.
    """
    G = nx.Graph()
    to_remove = []
    keys = list(item_dict.keys())
    for item_id in keys:
        raw_xml = item_dict[item_id]
        name, reagent_lists, level = _get_reagents_from_xml(raw_xml, item_id)
        if level > min_level:
            G.add_node(item_id, name = name, level = level, item_id = item_id)
            for reagent_list in reagent_lists:
                for reagent in reagent_list:
                    reagent_id = reagent['id']
                    if reagent_id in keys:
                        G.add_edge(item_id, reagent_id)
    for item_id in to_remove:
        G.remove_node(item_id)
    return G

def _load_item(item_path):
    """
    Load an item from local or from WowHead API.
    """
    file_name = './data/Reagents/{}'.format(item_path)
    f = open(file_name, "r")
    raw_xml = f.read()
    f.close()
    return raw_xml

def _get_dimensions_and_indexes(df):
    """
    Determines how many dimension (number of different kind of values in the df) and the list of indexes.
    """
    df_indexes = [col.split('_')[-1] for col in df.columns]
    unique, count = np.unique(df_indexes, return_counts=True)
    dimension = count[0]
    all_equals = np.all([c == dimension for c in count])
    return unique, dimension

def create_data_sequences(df, graph, tw):
    from data_preprocessing import create_inout_sequences
    import torch
    from torch_geometric.data import Data
    df_indexes, dimension = _get_dimensions_and_indexes(df)
    graph_indexes = list(graph.nodes)
    cuda = torch.device('cuda')
    # Keep only nodes that are in DF.
    for node in graph_indexes:
        if not node in df_indexes:
            print('delete node ' + node)
            graph.remove_node(node)
    node_dict = dict([(id, node) for id, node in enumerate(df_indexes)])
    inverse_node_dict = dict([(node, id) for id, node in enumerate(df_indexes)])
    # Create data edges
    starts = []
    ends = []
    for pair in graph.edges:
        start = inverse_node_dict[pair[0]]
        end = inverse_node_dict[pair[1]]
        starts.append(start)
        ends.append(end)
        starts.append(end)
        ends.append(start)
    # Values TODO : check order node
    xs = create_inout_sequences(df.values, tw)
    # TODO : create list of data and then dataset
    edge_index = torch.tensor([starts, ends], dtype=torch.long, device=cuda)
    dataset = []
    for x_features, x_labels in xs:
        # TODO check dimensions
        x_features = torch.tensor(x_features, dtype=torch.float, device=cuda).view(len(graph.nodes), -1, tw)
        # Convert label
        labels = torch.tensor(x_labels, dtype=torch.float, device = cuda).view(-1, dimension)
        data = Data(x=x_features, labels=labels, edge_index=edge_index)
        dataset.append(data)
    return dataset
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from undermine_query import get_all_df_from_graph

class UndermineDataset(InMemoryDataset):
    def __init__(self, root, graph, tw=13, transform=None, pre_transform=None):
        self.tw = tw
        self.graph = graph
        self.realms = ['outland', 'draenor', 'silvermoon', 'kazzak', 'ravencrest', 'ragnaros', 'stormscale', 'sylvanas']
        super(UndermineDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return [realm + '.csv' for realm in self.realms]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # List of realms : https://www.wowprogress.com/realms/rank/eu/lang.en

        # Get Data from all train realms
        for realm in tqdm(self.realms):
            individual_dfs = get_all_df_from_graph(graph, slug = '"{}"'.format(realm), region = '"EU"')
            merged_df = multi_timeseries_uniformization(individual_dfs, ['pricestart', 'quantityavg'])
            merged_df.to_csv(self.root + 'raw/{}.csv'.format(realm))

    def process(self):
        raw_dfs = [pd.read_csv(self.root + '/raw/' + realm) for realm in self.raw_file_names]
        indexes = [pd.to_datetime(raw_df['when']) for raw_df in raw_dfs]
        for raw_df, index in zip(raw_dfs, indexes):
            raw_df.index = index
        processed_dfs = [multi_timeseries_preprocessing(raw_df.drop('when', axis=1)) for raw_df in raw_dfs]
        
        data_list = []

        for processed_df in processed_dfs:
            data_list.extend(create_data_sequences(processed_df, self.graph, self.tw))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])