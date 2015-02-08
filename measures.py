# Calculates the different measures but for weighted and directed graphs
from __future__ import division
import networkx as nx
import math

class Measures():
	# graph is the required graph
	# sample_nodes can be list or dict
	def __init__(self, graph, sample_nodes):
		self.graph = graph
		self.sample_nodes = [n for n in sample_nodes]

	# Calulate all the available measures
	# Set false for any measure not wanted
	def combined(self, jaccard_index=True, adamic_adar_index=True, common_neighbor=True, preferential_attachment=True):
		jaccard = []
		adamic = []
		commnei = []
		prefattac = []

		counted = {}

		for x in xrange(0, len(self.sample_nodes)-1):
			node_1 =self. sample_nodes[x]
			for y in xrange(x+1, len(self.sample_nodes)-1):
				node_2 = self.sample_nodes[y]
				if (node_1,node_2) not in counted and (node_2,node_1) not in counted:
					counted[(node_1,node_2)] = 1

					set_x = set([k for k in self.graph[node_1]])
					set_y = set([k for k in self.graph[node_2]])

					set_intersection = set.intersection(set_x, set_y)
					set_union = set.union(set_x, set_y)

					if jaccard_index:
						value_jacc = self.__nodes_jaccard(node_1, node_2, set_intersection, set_union)
						jaccard.append((node_1, node_2, value_jacc))

					if adamic_adar_index:
						value_adam = self.__nodes_adamic_adar(node_1, node_2, set_intersection)
						adamic.append((node_1, node_2, value_jacc))

					if common_neighbor:
						value_comm = self.__nodes_common_neighbor(node_1, node_2, set_intersection)
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
		for x in xrange(0, len(self.sample_nodes)-1):
			node_1 =self. sample_nodes[x]
			for y in xrange(x+1, len(self.sample_nodes)-1):
				node_2 = self.sample_nodes[y]
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
		for x in xrange(0, len(self.sample_nodes)-1):
			node_1 = self.sample_nodes[x]
			for y in xrange(x+1, len(self.sample_nodes)-1):
				node_2 = self.sample_nodes[y]
				if (node_1,node_2) not in counted and (node_2,node_1) not in counted:
					counted[(node_1,node_2)] = 1
					value = self.__nodes_adamic_adar(node_1, node_2)
					data.append((node_1,node_2,d))
		return data

	# Calculate the common neighbor for sample nodes
	def common_neighbor(self):
		data = []
		counted = {}
		for x in xrange(0, len(self.sample_nodes)-1):
			node_1 = self.sample_nodes[x]
			for y in xrange(x+1, len(self.sample_nodes)-1):
				node_2 = self.sample_nodes[y]
				if (node_1, node_2) not in counted and (node_2, node_1) not in counted:
					counted[(node_1, node_2)] = 1
					value = self.__nodes_common_neighbor(node_1, node_2)
					data.append((node_1, node_2, value))
		return data

	# Calculate the preferential attachment for sample nodes
	def preferential_attachment(self):
		data = []
		counted = {}
		for x in xrange(0, len(self.sample_nodes)-1):
			node_1 = self.sample_nodes[x]
			for y in xrange(x+1, len(self.sample_nodes)-1):
				node_2 = self.sample_nodes[y]
				if (node_1, node_2) not in counted and (node_2, node_1) not in counted:
					counted[(node_1, node_2)] = 1
					value = self.__nodes_preferential_attachment(node_1, node_2)
					data.append((node_1, node_2, value))
		return data

	# Calculate the Jaccard Index of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_jaccard(self, node_1, node_2, set_intersection=None, set_union=None):
		# Initilize union by 0.001 to prevent divide by 0 error
		denominator = 0.01
		numerator = 0
		
		# Need to convert to graph is we are to calculate either set union or set intersection
		if set_intersection is None or set_union is None:
			set_x = set([k for k in self.graph[node_1]])
			set_y = set([k for k in self.graph[node_2]])
		
		# Set intersection not provided. So calulate it
		if set_intersection is None:
			set_intersection = set.intersection(set_x, set_y)
		
		# Set union not set. So calculate it
		if set_union is None:
			set_union = set.union(set_x, set_y)

		# Calculate the Jaccard Index
		for l in set_intersection:
			numerator += (self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight'])/2
		for l in set_union:
			if l in set_intersection:
				denominator += (self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight'])/2
			else:
				if l in self.graph[node_1]:
					denominator += self.graph[node_1][l]['weight']
				if l in self.graph[node_2]:
					denominator += self.graph[node_2][l]['weight']
		return numerator/denominator

	# Calculate the Adamic Adar Index of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_adamic_adar(self, node_1, node_2, set_intersection=None):
		score = 0
		
		# If set intersection is not provided, calculate
		if set_intersection is None:
			set_intersection = set.intersection(set([k for k in self.graph[node_1]]), set([k for k in self.graph[node_2]]))
		
		# Calculate the adamic adar index
		for l in set_intersection:
			numerator = (self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight'])/2
			denominator = 1
			for z in self.graph[l]:
				denominator += self.graph[l][z]['weight']
			denominator = math.log(denominator)
			if denominator <= 0:
				denominator = 0.01 
			score += numerator/denominator
		return score

	# Calculate the Common neighbor of two nodes 
	# Needs to specify the two nodes
	# Set intersection and union are optional
	# It will be calculayed by here if not provided
	def __nodes_common_neighbor(self, node_1, node_2, set_intersection=None):
		score = 0

		if set_intersection is None:
			set_intersection = set.intersection(set([k for k in self.graph[node_1]]), set([k for k in self.graph[node_2]]))
		
		for l in set_intersection:
			score += self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight']
		if len(set_intersection) > 0:
			score = score/(2*len(set_intersection))
		return score

	# Calculate the preferential attachment of two nodes 
	# Needs to specify the two nodes
	def __nodes_preferential_attachment(self, node_1, node_2):
		score = 0

		p1 = 0
		p2 = 0

		for n in self.graph[node_1]:
			p1 += self.graph[node_1][n]['weight']
		for n in self.graph[node_2]:
			p2 += self.graph[node_2][n]['weight']

		score = p1*p2

		return score




