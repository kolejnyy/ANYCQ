import re
import networkx as nx
import numpy as np


# def query_to_sql(cq_type, grounding):
	
# 	# Derive the entity and relation lists 	
# 	entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
# 	relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

# 	# Define the position dictionary
# 	pos_dict = {ent: [] for ent in entities}
# 	for i in range(len(relations)):
# 		pos_dict[relations[i][1]].append((i,'a'))
# 		pos_dict[relations[i][2]].append((i,'b'))

# 	query_graph = nx.DiGraph()
# 	query_graph.add_nodes_from(entities)
# 	for rel, u, v in relations:
# 		query_graph.add_edge(u, v, rel_id = grounding[rel])
# 		query_graph.add_edge(v, u, rel_id = grounding[rel] + (1 - 2 * (grounding[rel] % 2)))
	
# 	undir_graph = query_graph.to_undirected()
# 	non_constant = np.array([x for x in entities if x[0]!='s' and x[0]!='f'])

# 	min_fbs = np.ones(100)
# 	for mask in range(2**len(non_constant)):
# 		_mask = mask
# 		fbv = []
# 		for i in range(len(non_constant)):
# 			fbv.append(False if _mask%2==0 else True)
# 			_mask = _mask // 2
# 		fbv = np.array(fbv)
# 		ind_sub = undir_graph.subgraph(non_constant[~fbv])
# 		try:
# 			cycle = nx.find_cycle(ind_sub)
# 		except:
# 			if fbv.sum() < min_fbs.sum():
# 				min_fbs = fbv
# 	min_fbs = non_constant[min_fbs]
# 	inter_query = ""

# 	vis = {v : False for v in entities}

# 	def recursive_query(node, parent):
		
# 		# If we reach a leaf node, return anything for it
# 		if len(query_graph[node]) == 1:
# 			return ""

# 		vis[node] = True
# 		output = ""

# 		conditions = []
# 		for neigh in query_graph[node]:
			
# 			if vis[neigh] and neigh[0] != 's' and neigh != 'f1' and neigh not in min_fbs:
# 				continue
# 			if neigh == parent:
# 				continue

# 			rel_id = query_graph[node][neigh]['rel_id']
			
# 			# Consider constant nodes
# 			if neigh[0] == 's':
# 				conditions.append(f"({node}.a IN (SELECT DISTINCT graph.a FROM graph WHERE graph.r={rel_id} AND graph.b={grounding[neigh]}))")
# 				continue

# 			if neigh == 'f1':
# 				conditions.append(f"({node}.a IN (SELECT DISTINCT graph.a FROM graph WHERE graph.r={rel_id} AND graph.b=f1.a))")
# 				continue
# 			if neigh in min_fbs:
# 				conditions.append(f"({node}.a IN (SELECT DISTINCT graph.a FROM graph WHERE graph.r={rel_id} AND graph.b=inter.{neigh}))")
# 				continue

# 			# Consider entity nodes
# 			conditions.append(f"({node}.a IN (SELECT DISTINCT graph.a FROM graph WHERE graph.r={rel_id} AND graph.b IN (SELECT * FROM (SELECT DISTINCT a FROM graph) AS {neigh} {recursive_query(neigh, node)})))")

# 		if len(conditions) != 0:
# 			output += 'WHERE ' + " AND ".join(conditions)

# 		return output

# 	query = "SELECT DISTINCT f1.a \n"
# 	query += "FROM (SELECT DISTINCT a FROM graph"
# 	query += ' WHERE '
# 	conds = []
# 	for rel_id, pos in pos_dict['f1']:
# 		rel_grd = grounding[relations[rel_id][0]]
# 		rev_rel = rel_grd + 1 - 2 * (rel_grd % 2)
# 		if pos == 'a':
# 			conds.append(f'graph.a IN (SELECT a FROM graph WHERE graph.r={rel_grd})')
# 		else:
# 			conds.append(f'graph.a IN (SELECT a FROM graph WHERE graph.r={rev_rel})')
# 	query += ' AND '.join(conds)
# 	query += ') AS f1'

# 	f1_cond = recursive_query('f1', None)

# 	inter_query = 'SELECT DISTINCT ' + ', '.join([f'graph{fv_node}.a AS {fv_node}' for fv_node in min_fbs]) + " FROM " + ', '.join([f'graph AS graph{fv_node}' for fv_node in min_fbs])
	
# 	inter_query += ' WHERE '
# 	inter_conds = []
# 	for fb_node in min_fbs:
# 		for rel_id, pos in pos_dict[fb_node]:
# 			rel_grd = grounding[relations[rel_id][0]]
# 			rev_rel = rel_grd + 1 - 2 * (rel_grd % 2)
# 			if pos == 'a':
# 				inter_conds.append(f'graph{fb_node}.a IN (SELECT a FROM graph WHERE graph.r={rel_grd})')
# 			else:
# 				inter_conds.append(f'graph{fb_node}.a IN (SELECT a FROM graph WHERE graph.r={rev_rel})')
# 	inter_query += ' AND '.join(inter_conds)

# 	if len(min_fbs) != 0:
# 		query += ", ("+inter_query+") AS inter"
# 	query += "\n "

# 	inter_conds = [recursive_query(fb_node, None).replace(f"{fb_node}.a", f"inter.{fb_node}") for fb_node in min_fbs]
# 	conds = [x for x in inter_conds if x != ''] + [f1_cond]
# 	conds = [x[6:] for x in conds]

# 	query += ' WHERE ' + ' AND '.join(conds)

# 	return query



def query_to_sql(cq_type, grounding):
	
	# Derive the entity and relation lists 	
	entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
	relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

	pos_rel = [len(x)==3 for x in relations]
	relations = [x[-3:] for x in relations]

	# Define the position dictionary
	pos_dict = {ent: [] for ent in entities}
	for i in range(len(relations)):
		pos_dict[relations[i][1]].append((i,'a'))
		pos_dict[relations[i][2]].append((i,'b'))

	query_graph = nx.DiGraph()
	query_graph.add_nodes_from(entities)
	for i, (rel, u, v) in enumerate(relations):
		query_graph.add_edge(u, v, rel_id = grounding[rel], sign = pos_rel[i])
		query_graph.add_edge(v, u, rel_id = grounding[rel] + (1 - 2 * (grounding[rel] % 2)), sign = pos_rel[i])
	
	undir_graph = query_graph.to_undirected()
	non_constant = np.array([x for x in entities if x[0]!='s' and x[0]!='f'])

	min_fbs = np.ones(100)
	for mask in range(2**len(non_constant)):
		_mask = mask
		fbv = []
		for i in range(len(non_constant)):
			fbv.append(False if _mask%2==0 else True)
			_mask = _mask // 2
		fbv = np.array(fbv)
		ind_sub = undir_graph.subgraph(non_constant[~fbv])
		try:
			cycle = nx.find_cycle(ind_sub)
		except:
			if fbv.sum() < min_fbs.sum():
				min_fbs = fbv
	min_fbs = non_constant[min_fbs]
	inter_query = ""

	vis = {v : False for v in entities}

	def recursive_query(node, parent):
		
		# If we reach a leaf node, return anything for it
		if len(query_graph[node]) == 1 and not parent is None:
			return ""

		vis[node] = True
		output = ""

		conditions = []
		for neigh in query_graph[node]:
			
			if vis[neigh] and neigh[0] != 's' and neigh[0] != 'f' and neigh not in min_fbs:
				continue
			if neigh == parent:
				continue

			rel_id = query_graph[node][neigh]['rel_id']
			pos_sign = "" if query_graph[node][neigh]['sign'] else "NOT "
			f_var = 'a' if rel_id % 2 == 0 else 'b'
			b_var = 'b' if rel_id % 2 == 0 else 'a'
			rel_id = rel_id - (rel_id % 2)
			
			# Consider constant nodes
			if neigh[0] == 's':
				conditions.append(f"{pos_sign}({node}.a IN (SELECT DISTINCT graph.{f_var} FROM graph WHERE graph.r={rel_id} AND graph.{b_var}={grounding[neigh]}))")
				continue

			if neigh[0] == 'f':
				conditions.append(f"{pos_sign}({node}.a IN (SELECT DISTINCT graph.{f_var} FROM graph WHERE graph.r={rel_id} AND graph.{b_var}={neigh}.a))")
				continue
			if neigh in min_fbs:
				conditions.append(f"{pos_sign}({node}.a IN (SELECT DISTINCT graph.{f_var} FROM graph WHERE graph.r={rel_id} AND graph.{b_var}=inter.{neigh}))")
				continue

			# Consider entity nodes
			conditions.append(f"{pos_sign}({node}.a IN (SELECT DISTINCT graph.{f_var} FROM graph WHERE graph.r={rel_id} AND graph.{b_var} IN (SELECT * FROM (SELECT DISTINCT a FROM graph) AS {neigh} {recursive_query(neigh, node)})))")

		if len(conditions) != 0:
			output += 'WHERE ' + " AND ".join(conditions)

		return output

	free_vars = [x for x in entities if x[0] == 'f']

	query = "SELECT DISTINCT "
	query += ", ".join([x+'.a AS '+x for x in free_vars])
	query += "\nFROM "
	f_queries = []
	for free_var in free_vars:
		_query = '(SELECT DISTINCT a FROM graph WHERE '
		conds = []
		for rel_id, pos in pos_dict[free_var]:
			if not pos_rel[rel_id]:
				continue
			rel_grd = grounding[relations[rel_id][0]]
			rev_rel = rel_grd + 1 - 2 * (rel_grd % 2)
			if pos == 'a':
				conds.append(f'graph.a IN (SELECT a FROM graph WHERE graph.r={rel_grd})')
			else:
				conds.append(f'graph.a IN (SELECT a FROM graph WHERE graph.r={rev_rel})')
		_query += ' AND '.join(conds)
		_query += ') AS '+free_var
		f_queries.append(_query)
	query += ', '.join(f_queries)

	f_conds = [recursive_query(free_var, None) for free_var in free_vars]
	f_conds = [x for x in f_conds if x != '']

	inter_query = 'SELECT DISTINCT ' + ', '.join([f'graph{fv_node}.a AS {fv_node}' for fv_node in min_fbs]) + " FROM " + ', '.join([f'graph AS graph{fv_node}' for fv_node in min_fbs])
	
	inter_query += ' WHERE '
	inter_conds = []
	for fb_node in min_fbs:
		for rel_id, pos in pos_dict[fb_node]:
			rel_grd = grounding[relations[rel_id][0]]
			rev_rel = rel_grd + 1 - 2 * (rel_grd % 2)
			if pos == 'a':
				inter_conds.append(f'graph{fb_node}.a IN (SELECT a FROM graph WHERE graph.r={rel_grd})')
			else:
				inter_conds.append(f'graph{fb_node}.a IN (SELECT a FROM graph WHERE graph.r={rev_rel})')
	inter_query += ' AND '.join(inter_conds)

	if len(min_fbs) != 0:
		query += ", ("+inter_query+") AS inter"
	query += "\n "

	inter_conds = [recursive_query(fb_node, None).replace(f"{fb_node}.a", f"inter.{fb_node}") for fb_node in min_fbs]
	conds = [x for x in inter_conds if x != ''] + f_conds
	conds = [x[6:] for x in conds]

	query += ' WHERE ' + ' AND '.join(conds)

	return query