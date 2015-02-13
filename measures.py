# Calculates the different measures but for weighted and directed graphs
from __future__ import division
import networkx as nx
import math

# Calculate the measures in blocks
class Measures():
	# graph is the required graph
	# sample_nodes can be list or dict
	def __init__(self, graph=None, sample_nodes=None):
		self._graph = graph
		if sample_nodes is not None:
			self._sample_nodes = [n for n in sample_nodes]
		else:
			self._sample_nodes = sample_nodes

	# Calulate all the available measures
	# Set false for any measure not wanted
	def combined(self, jaccard_index=True, adamic_adar_index=True, common_neighbor=True, preferential_attachment=True):
		jaccard = []
		adamic = []
		commnei = []
		prefattac = []

		counted = {}

		for x in xrange(0, len(self._sample_nodes)-1):
			node_1 =self._sample_nodes[x]
			for y in xrange(x+1, len(self._sample_nodes)-1):
				node_2 = self._sample_nodes[y]
				if (node_1,node_2) not in counted and (node_2,node_1) not in counted:
					counted[(node_1,node_2)] = 1

					set_x = set([k for k in self._graph[node_1]])
					set_y = set([k for k in self._graph[node_2]])

					set_intersection_out = set.intersection(set_x, set_y)
					set_union_out = set.union(set_x, set_y)

					graph_in = self._graph.reverse(copy=True)

					set_x = set([k for k in graph_in[node_1]])
					set_y = set([k for k in graph_in[node_2]])

					set_intersection_in = set.intersection(set_x, set_y)
					set_union_in = set.union(set_x, set_y)

					if jaccard_index:
						value_jacc = self.__nodes_jaccard(node_1, node_2, set_intersection_out, set_union_out, set_intersection_in, set_union_in)
						jaccard.append((node_1, node_2, value_jacc))

					if adamic_adar_index:
						value_adam = self.__nodes_adamic_adar(node_1, node_2, set_intersection_out, set_intersection_in)
						adamic.append((node_1, node_2, value_adam))

					if common_neighbor:
						value_comm = self.__nodes_common_neighbor(node_1, node_2, set_intersection_in, set_intersection_out)
						commnei.append((node_1, node_2, value_comm))

					if preferential_attachment:
						value_pref = self.__nodes_preferential_attachment(node_1, node_2)
						prefattac.append((node_1, node_2, value_pref))

		return (jaccard, adamic, commnei, prefattac)

	# Calculates the jaccard coefficient for sample nodes
	# Returns a list
	# jaccard(x,y) = jaccard(y,x) for directed weighted graphs as well
	def jaccard_coefficient(self):
		data = []
		counted = {}
		for x in xrange(0, len(self._sample_nodes)-1):
			node_1 =self. sample_nodes[x]
			for y in xrange(x+1, len(self._sample_nodes)-1):
				node_2 = self._sample_nodes[y]
				if (node_1,node_2) not in counted and (node_2,node_1) not in counted:
					counted[(node_1,node_2)] = 1
					value = self.__nodes_jaccard(node_1, node_2)
					data.append((node_1,node_2,value))
		return data

	# Calculates the adamic adar index for sample nodes
	# Returns a list;
	# adamic_adar(x,y) = adamic_adar(y,x) for directed weighted graphs as well
	def adamic_adar(self):
		data = []
		counted = {}
		for x in xrange(0, len(self._sample_nodes)-1):
			node_1 = self._sample_nodes[x]
			for y in xrange(x+1, len(self._sample_nodes)-1):
				node_2 = self._sample_nodes[y]
				if (node_1,node_2) not in counted and (node_2,node_1) not in counted:
					counted[(node_1,node_2)] = 1
					value = self.__nodes_adamic_adar(node_1, node_2)
					data.append((node_1,node_2,d))
		return data

	# Calculate the common neighbor for sample nodes
	def common_neighbor(self):
		data = []
		counted = {}
		for x in xrange(0, len(self._sample_nodes)-1):
			node_1 = self._sample_nodes[x]
			for y in xrange(x+1, len(self._sample_nodes)-1):
				node_2 = self._sample_nodes[y]
				if (node_1, node_2) not in counted and (node_2, node_1) not in counted:
					counted[(node_1, node_2)] = 1
					value = self.__nodes_common_neighbor(node_1, node_2)
					data.append((node_1, node_2, value))
		return data

	# Calculate the preferential attachment for sample nodes
	def preferential_attachment(self):
		data = []
		counted = {}
		for x in xrange(0, len(self._sample_nodes)-1):
			node_1 = self._sample_nodes[x]
			for y in xrange(x+1, len(self._sample_nodes)-1):
				node_2 = self._sample_nodes[y]
				if (node_1, node_2) not in counted and (node_2, node_1) not in counted:
					counted[(node_1, node_2)] = 1
					value = self.__nodes_preferential_attachment(node_1, node_2)
					data.append((node_1, node_2, value))
		return data

	# Calculate the Jaccard Index of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_jaccard(self, node_1, node_2, set_intersection_out=None, set_union_out=None, set_intersection_in=None, set_union_in=None):
		graph_in = self._graph.reverse(copy=True)

		# Need to convert to graph is we are to calculate either set union or set intersection
		if set_intersection_out is None or set_union_out is None or set_intersection_in is None or set_union_in is None:
			set_x = set([k for k in self._graph[node_1]])
			set_y = set([k for k in self._graph[node_2]])

			set_intersection_out = set.intersection(set_x, set_y)
			set_union_out = set.union(set_x, set_y)
			
			set_x = set([k for k in graph_in[node_1]])
			set_y = set([k for k in graph_in[node_2]])

			set_intersection_in = set.intersection(set_x, set_y)
			set_union_in = set.union(set_x, set_y)

		# Jaccard for out links
		# Initilize union by 0.001 to prevent divide by 0 error
		denominator = 0.01
		numerator = 0

		# Calculate the Jaccard Index
		for l in set_intersection_out:
			numerator += self._graph[node_1][l]['weight'] + self._graph[node_2][l]['weight']
		for l in set_union_out:
			if l in set_intersection_out:
				denominator += self._graph[node_1][l]['weight'] + self._graph[node_2][l]['weight']
			else:
				if l in self._graph[node_1]:
					denominator += self._graph[node_1][l]['weight']
				if l in self._graph[node_2]:
					denominator += self._graph[node_2][l]['weight']

		value = 0.5(numerator/denominator)

		# Jaccard for in links
		# Initilize union by 0.001 to prevent divide by 0 error
		denominator = 0.01
		numerator = 0

		# Calculate the Jaccard Index
		for l in set_intersection_in:
			numerator += graph_in[node_1][l]['weight'] + graph_in[node_2][l]['weight']
		for l in set_union_in:
			if l in set_intersection_in:
				denominator += graph_in[node_1][l]['weight'] + graph_in[node_2][l]['weight']
			else:
				if l in graph_in[node_1]:
					denominator += graph_in[node_1][l]['weight']
				if l in graph_in[node_2]:
					denominator += graph_in[node_2][l]['weight']

		value += 0.5(numerator/denominator)

		return value

	# Calculate the Adamic Adar Index of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_adamic_adar(self, node_1, node_2, set_intersection_out=None, set_intersection_in=None):
		graph_in = self._graph.reverse(copy=True)
		score = 0
		
		# If set intersection is not provided, calculate
		if set_intersection_in is None or set_intersection_out is None:
			set_intersection_out = set.intersection(set([k for k in self._graph[node_1]]), set([k for k in self._graph[node_2]]))
			set_intersection_in = set.intersection(set([k for k in graph_in[node_1]]), set([k for k in graph_in[node_2]]))
		
		# Calculate the adamic adar index for out links
		for l in set_intersection_out:
			numerator = (self._graph[node_1][l]['weight'] + self._graph[node_2][l]['weight'])/2
			denominator = 1
			for z in self._graph[l]:
				if self._graph[l][z]['weight'] <= 0:
					denominator = 0.001
				else:
					denominator += math.log(self._graph[l][z]['weight'])
			score += numerator/denominator

		# Calculate the adamic adar index for in links
		for l in set_intersection_in:
			numerator = (graph_in[node_1][l]['weight'] + graph_in[node_2][l]['weight'])/2
			denominator = 1
			for z in graph_in[l]:
				if graph_in[l][z]['weight'] <= 0:
					denominator = 0.001
				else:
					denominator += math.log(graph_in[l][z]['weight'])
			score += numerator/denominator

		return score/2

	# Calculate the Common neighbor of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_common_neighbor(self, node_1, node_2, set_intersection_out=None, set_intersection_in=None):
		score = 0
		graph_in = self._graph.reverse(copy=True)

		if set_intersection_out is None or set_intersection_in is None:
			set_intersection_out = set.intersection(set([k for k in self._graph[node_1]]), set([k for k in self._graph[node_2]]))
			set_intersection_in = set.intersection(set([k for k in graph_in[node_1]]), set([k for k in graph_in[node_2]]))
		
		for l in set_intersection_out:
			score += self._graph[node_1][l]['weight'] + self._graph[node_2][l]['weight']
		for l in set_intersection_in:
			score += graph_in[node_1][l]['weight'] + graph_in[node_2][l]['weight']
		
		return score/2

	# Calculate the preferential attachment of two nodes 
	# Needs to specify the two nodes
	def __nodes_preferential_attachment(self, node_1, node_2):
		graph_in = self._graph.reverse(copy=True)

		score = 0

		p1 = 0
		p2 = 0

		for n in self._graph[node_1]:
			p1 += self._graph[node_1][n]['weight']
		for n in self._graph[node_2]:
			p2 += self._graph[node_2][n]['weight']

		score += p1*p2

		p1 = 0
		p2 = 0

		for n in graph_in[node_1]:
			p1 += graph_in[node_1][n]['weight']
		for n in graph_in[node_2]:
			p2 += graph_in[node_2][n]['weight']

		score += p1*p2

		return score/2

	# Get the in degree from the adjacency list 
	def out_degree(self, adjacency_list, sample_nodes):
		out_degree = {}

		for node in sample_nodes:
			if node in adjacency_list:
				out_degree[node] = len(adjacency_list[node])
			else:
				out_degree[node] = 0
		return out_degree

	# Caluculate the out degree from adjacency list
	def in_degree(self, adjacency_list, sample_nodes):
		in_degree = {}

		for n in sample_nodes:
			in_degree[n] = 0

		for x in adjacency_list:
			for y in adjacency_list[x]:
				if sample_nodes is None or y in sample_nodes:
					in_degree[y] += 1
		return in_degree




