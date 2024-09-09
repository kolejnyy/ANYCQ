import numpy as np
import networkx as nx
import os
import json
import re
from copy import deepcopy


def load_encodings(path):
    with open(os.path.join(path, 'kgindex.json')) as f:
        map_dict = json.load(f)

    encode_r = {}
    encode_e = {}

    for key in map_dict['r'].keys():
        encode_r[map_dict['r'][key]] = key
    for key in map_dict['e'].keys():
        encode_e[map_dict['e'][key]] = key

    return encode_r, encode_e


def baseline_nx(cq_type):
    # Initialise the graph with vertices representing variables
    graph = nx.DiGraph()
    ent_num = 0
    
    ent_to_idx = {}

    for k, dis_cq_type in enumerate(cq_type.split('|')):

        # Find the variables present in the query
        entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', dis_cq_type) if (len(x)>0 and x[0]!='r')])
        for idx, ent in enumerate(entities):
            if not (ent in ent_to_idx):
                ent_to_idx[ent] = ent_num
                ent_num += 1 

        graph.add_nodes_from([
            (ent_to_idx[entities[i]], {'name': entities[i], 'type': ent_type(entities[i])}) for i in range(len(entities))
        ])

        # Parse the relations in the query
        relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in dis_cq_type.split('&') if len(x)>0]

        # Add edges representing relations
        for idx, relation in enumerate(relations):
            neg = (relation[0]=='!')
            if idx == 0:
                graph.add_edge(ent_to_idx[relation[-2]], ent_to_idx[relation[-1]], name=relation[-3], neg=neg, dis_part = 1)
            else:
                graph.add_edge(ent_to_idx[relation[-2]], ent_to_idx[relation[-1]], name=relation[-3], neg=neg)

    return graph, ent_to_idx


def ent_type(entity):
    if entity[0] == 'e':
        return 'exist'
    elif entity[0] == 'f':
        return 'pred'
    elif entity[0] == 's':
        return 'const'
    else:
        return None


def augment(graph_info, node_info):
    graph, ent_to_idx = deepcopy(graph_info)

    for u, v, rel in graph.edges(data=True):
        graph.edges[u,v]['idx'] = node_info[rel['name']]

    for u, info in graph.nodes(data=True):
        if info['type'] == 'const':
            graph.nodes[u]['idx'] = node_info[info['name']]
            
    return graph