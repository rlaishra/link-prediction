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
					
					# Initilize union by 0.001 to prevent divide by 0 error
					denominator = 0.01
					numerator = 0
					
					set_x = set([k for k in self.graph[node_1]])
					set_y = set([k for k in self.graph[node_2]])
					
					set_intersection = set.intersection(set_x, set_y)
					set_union = set.union(set_x, set_y)

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
					data.append((node_1,node_2,(numerator/denominator)))
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
					d = 0
					set_intersection = set.intersection(set([k for k in self.graph[node_1]]), set([k for k in self.graph[node_2]]))
					for l in set_intersection:
						numerator = (self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight'])/2
						denominator = 1
						for z in self.graph[l]:
							denominator += self.graph[l][z]['weight']
						denominator = math.log(denominator)
						if denominator <= 0:
							denominator = 0.01 
						d += numerator/denominator
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
					set_intersection = set.intersection(set([k for k in self.graph[node_1]]), set([k for k in self.graph[node_2]]))
					d = 0
					for l in set_intersection:
						d += self.graph[node_1][l]['weight'] + self.graph[node_2][l]['weight']
					if len(set_intersection) > 0:
						d = d/len(set_intersection)
					data.append((node_1, node_2, d))
		return data




