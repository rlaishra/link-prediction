# The main network file
# Cretes a directed weighted graph

from __future__ import division
import networkx as nx 
import copy

class SocialNetwork():
	# cut_off is the minimum weight below which a link is assumed to not exist
	# sample_nodes are the nodes to analyse. None if we want to analyse all nodes
	def __init__(self, cut_off, sample_nodes):
		self.graph = nx.DiGraph()
		self.weight_cut_off = cut_off
		self.sample_nodes = sample_nodes
	
	# Initialize with all the nodes
	# nodes is array or list
	def initialize_nodes(self, nodes):
		for node in nodes:
			self.graph.add_node(node)

	# Add edges to graph from adjacency list
	# If edge already exist, it is decremented according to beta and new value added to it
	# beta is the decay factor
	# beta = 1 means no decay
	# beta = 0 means weight from last time interval is removed completely
	# new_weight = weight_in_adjaceny_list + beta * previous_weight
	def add_edges(self, adjacency_list, beta):
		self.__decrement_weight(beta)
		for user1 in adjacency_list:
			for user2 in adjacency_list[user1]:
				if user1 != user2:
					if self.graph.has_edge(user1, user2):
						self.graph[user1][user2]['weight'] += adjacency_list[user1][user2]
					else:
						self.graph.add_edge(user1, user2, weight=adjacency_list[user1][user2])

	# Decrement weight in links according to beta
	def __decrement_weight(self, beta):
		for g in self.graph.edges(data=True):
			self.graph[g[0]][g[1]]['weight'] = beta*self.graph[g[0]][g[1]]['weight']

	# Return a copy of the graph
	def get_graph(self):
		return self.graph
