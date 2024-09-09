import json
import re

import torch
import pandas as pd
import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm



def refine_edges(new_edges, p_out, n_vals):
	new_vertices = np.unique(np.concatenate([new_edges[0], new_edges[2]]))
	degrees = np.array([len(np.unique(new_edges[2][new_edges[0]==vertex]))+len(np.unique(new_edges[0][new_edges[2]==vertex])) for vertex in new_vertices])

	# Remove each leaf with probability p_out
	out_probs = np.random.choice([0,1], new_vertices.shape, p=[1-p_out, p_out])
	out_map = (degrees==1)*(out_probs==1)
	out_ver_map = np.zeros(n_vals)
	out_ver_map[new_vertices[out_map==1]]=1
	out_edge_map = np.invert(np.logical_or(out_ver_map[new_edges[0]], out_ver_map[new_edges[2]]))

	ref_edges = new_edges[:,out_edge_map]

	return ref_edges


def generate_graph(kg_data, unobs_kg_data, n_pivots, p_out, min_vertex):

	# Retrieve the number of entities
	n_vals = kg_data[0].max()+1

	# Evaluate the neighbour sets
	neighbours = [kg_data[2][kg_data[0]==i] for i in range(n_vals)]

	# Randomly choose the free variable and generate its 2-hop neighbourhood
	unobs_edge_id = np.random.randint(unobs_kg_data.shape[1])
	free_node = unobs_kg_data[0,unobs_edge_id]
	first_pivot = unobs_kg_data[2,unobs_edge_id]
	hop2neigh = [free_node]
	for x in neighbours[free_node]:
		hop2neigh = hop2neigh + list(neighbours[x])
	hop2neigh = np.unique(hop2neigh)
	hop2neigh = hop2neigh[hop2neigh != free_node]
	hop2neigh = hop2neigh[hop2neigh != first_pivot]

	# Select n_pivot pivot nodes
	selected = np.random.choice(hop2neigh, n_pivots-1, replace=False) if n_pivots-1 <= len(hop2neigh) else hop2neigh
	selected = np.array(list(selected)+[free_node, first_pivot])
	selected_map = np.zeros(n_vals)
	selected_map[selected]=1
	new_edges = kg_data[:,selected_map[kg_data[0]]==1]

	# Refine the graph by eliminating leaf nodes
	last_shape = new_edges.shape
	ref_edges = refine_edges(new_edges, p_out, n_vals)

	# Get the new set of vertices
	ref_vertices = np.unique(np.concatenate([ref_edges[0], ref_edges[2]]))

	# Create the corresponding nx graph
	graph = nx.Graph()
	graph.add_nodes_from(ref_vertices)
	graph.add_edges_from(ref_edges.T[:,[0,2]])

	# Sample a subgraph of a reasonable size, until we get something connected
	sel_ver = np.unique(list(np.random.choice(ref_vertices, min_vertex, False))+list(selected)) if len(ref_vertices)>=min_vertex else np.unique(list(ref_vertices)+list(selected))
	subgraph = graph.subgraph(sel_ver)
	sel_map = np.zeros(n_vals)
	sel_map[sel_ver]=1
	sel_edges = ref_edges[:,(sel_map[ref_edges[0]]*sel_map[ref_edges[2]])==1]

	# Remove self loops
	sel_edges = sel_edges[:,sel_edges[0]!=sel_edges[2]]
	subgraph = nx.Graph()
	subgraph.add_nodes_from(sel_ver)
	subgraph.add_edges_from(sel_edges.T[:,[0,2]])

	return subgraph, sel_ver, sel_edges, selected, free_node


def save_graph(g, filename, free = -1, const = []):
	fig = plt.figure()

	color_map = []
	for node in g:
		if node==free:
			color_map.append('orange')
		elif node in const:
			color_map.append('#0033FF')
		else:
			color_map.append('#888888')


	nx.draw(g, ax=fig.add_subplot(), node_color=color_map)
	if True:
		# Save plot to file
		matplotlib.use("Agg")
		fig.savefig(filename)


def generate_large_query(kg_data, unobs_kg_data, min_desired_vertex_num, n_pivots, p_out, save_img = None):

	# Generate a sample subgraph of the knowledge graph
	subgraph, vertices, edges, pivots, free_node = generate_graph(kg_data, unobs_kg_data, n_pivots, p_out, min_desired_vertex_num)
	while not nx.is_connected(subgraph):
		subgraph, vertices, edges, pivots, free_node = generate_graph(kg_data, unobs_kg_data, n_pivots, p_out, min_desired_vertex_num)

	pivot_map = np.array([(1 if node in pivots else 0) for node in vertices])
	degrees = np.array([(edges[0]==vertex).sum()+(edges[2]==vertex).sum() for vertex in vertices])

	# Decide which vertices should be converted to constant
	const_map = np.random.random(degrees.shape)<0.2*n_pivots/degrees**2
	const_map[pivot_map==1]=0
	while not nx.is_connected(subgraph.subgraph(vertices[const_map==0])):
		const_map = np.random.random(degrees.shape)<0.2*n_pivots/degrees**2
		const_map[pivot_map==1]=0

	# Add checking if constant nodes do not disconnect the formula
	swap_map = np.random.randint(0, 2, edges.shape[1])
	edges[:,swap_map==1] = edges[:, swap_map==1][np.array([2,1,0])]
	edges[1,swap_map==1] = edges[1,swap_map==1] - 2*(edges[1,swap_map==1]%2) + 1

	edge_map = {rel: "r"+str(k+1) for k, rel in enumerate(np.unique(edges[1]))}
	vertex_map = {vertex: ("s"+str(k+1) if const_map[k] else "e"+str(k+1)) for k, vertex in enumerate(vertices) }
	vertex_map[free_node] = 'f1'

	query = '&'.join(['('+edge_map[rel]+'('+vertex_map[head]+','+vertex_map[tail]+'))' for head, rel, tail in edges.T])

	if save_img:
		save_graph(subgraph, save_img, free=free_node, const = vertices[const_map])

	_input = {v:int(k) for k,v in edge_map.items()}|{v:int(k) for k,v in vertex_map.items() if v[0]=='s'}
	_answer = {v:[[int(k)]] for k,v in vertex_map.items() if v[0]=='f'}
	_full_ans = {v:int(k) for k,v in vertex_map.items() if v[0]!='s'}

	return query, _input, _answer, _full_ans, edges


def quality_check(edges, free_node, obs_kg, min_vertex_num):

	# Check if the free node has at least degree 2 in the graph
	if len(np.unique(list(edges[2][edges[0]==free_node])+list(edges[0][edges[2]==free_node]))) < 2:
		return False
	
	if len(np.unique(list(edges[0])+list(edges[2]))) < min_vertex_num:
		return False

	#! Check if the query is not a subgraph of the original graph
	observable = True
	for edge in edges.T:
		if np.sum(obs_kg.T==edge, axis=1).max() != 3:
			observable = False
			break
	
	return not observable

def setdiff_nd_positivenums(a,b):
    s = np.maximum(a.max(0)+1,b.max(0)+1)
    return a[~np.isin(a.dot(s),b.dot(s))]

def generate_large_query_dataset(data_path, dataset_name,
		dataset_size,
		n_pivots,
		min_desired_vertex_num,
		p_out,
		obs_graph = 'train_kg.tsv',
		full_graph = 'valid_kg.tsv'
	):

	# Retrieve the KG data
	kg_data = pd.read_csv(data_path+'/'+full_graph, header=None, sep='\t')
	kg_data = kg_data.to_numpy().T

	obs_kg_data = pd.read_csv(data_path+'/'+obs_graph, header=None, sep='\t')
	obs_kg_data = obs_kg_data.to_numpy().T

	# Evaluate the difference between the unobservable and observable KG parts
	unobs_kg_data = setdiff_nd_positivenums(kg_data.T, obs_kg_data.T).T

	res_dataset = {}

	# Generate the samples
	for idx in tqdm(range(dataset_size)):

		query, input_data, ans, full_ans, edges = generate_large_query(kg_data, unobs_kg_data, min_desired_vertex_num, n_pivots, p_out, save_img = 'images/vcq/'+data_path.split('/')[1]+'/'+dataset_name+'_graph_'+str(idx)+'.png' if idx < 20 else None)
		while not quality_check(edges, ans['f1'][0][0], obs_kg_data, min_desired_vertex_num):
			query, input_data, ans, full_ans, edges = generate_large_query(kg_data, unobs_kg_data, min_desired_vertex_num, n_pivots, p_out, save_img = 'images/vcq/'+data_path.split('/')[1]+'/'+dataset_name+'_graph_'+str(idx)+'.png' if idx < 20 else None)

		record = [input_data, ans, {'f1': []}]

		if query in res_dataset:
			res_dataset[query].append(record)
		else:
			res_dataset[query] = [record]

	with open(data_path+'/'+dataset_name+'.json', 'w') as f:
		json.dump(res_dataset, f)