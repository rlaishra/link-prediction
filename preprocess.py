# Preprocessing of data like detect outliers etc

import networkx as nx 
import math, fileio, operator
from scipy.spatial import distance
from pprint import pprint

class Preprocess(object):
	
	def __init__(self):
		super(Preprocess, self).__init__()
		
	# Detect nodes that are outliers based on in degree and out degree
	# Uses density based outlier detection
	def outlier_nodes(self, adjacency_list, users_list=None, k=5):
		in_degree = self.__in_degree(adjacency_list, users_list)
		out_degree = self.__out_degree(adjacency_list, users_list)

		#construct set of all nodes
		all_nodes = list(set.union(set(in_degree), set(out_degree)))

		# Construct the features
		# [in_degree, out_degree]
		features = {}
		for x in all_nodes:
			f1 = 0
			f2 = 0
			if x in in_degree:
				f1 = in_degree[x]
			if x in out_degree:
				f2 = out_degree[x]
			features[x] = [f1, f2]

		fio = fileio.Fileio()

		if not fio.exist_preprocess_distance_matrix():
			distance_matrix = self.__distance_matrix(features)
			fio.save_reprocess_distance_matrix(distance_matrix)
		else:
			distance_matrix = fio.read_reprocess_distance_matrix()

		nearest_nodes = self.__get_k_neartest(distance_matrix, k)
		for x in nearest_nodes:
			if '1600meeting' not in nearest_nodes[x]:
				print(x, nearest_nodes[x])

		return distance_matrix

	# Get the in degree from the adjacency list 
	def __out_degree(self, adjacency_list, users_list=None):
		out_degree = {}

		for node in users_list:
			if node in adjacency_list:
				out_degree[node] = len(adjacency_list[node])
			else:
				out_degree[node] = 0

		return out_degree

	# Caluculate the out degree from adjacency list
	def __in_degree(self, adjacency_list, users_list=None):
		in_degree = {}

		for n in users_list:
			in_degree[n] = 0

		for x in adjacency_list:
			for y in adjacency_list[x]:
				if users_list is None or y in users_list:
					in_degree[y] += 1

		return in_degree

	# Calculate the distance between every two pairs of nodes
	def __distance_matrix(self, features, k=5):
		node_distance = {}
		coords = []
		nodes = []
		for f in features:
			coords.append((features[f][0], features[f][1]))
			nodes.append(f)
		
		distance_matrix = distance.cdist(coords, coords, 'euclidean')

		for x in xrange(0,len(nodes)-1):
			node_distance[nodes[x]] = {}
			for y in xrange(0, len(nodes) -1):
				if nodes[x] is not nodes[y]:
					node_distance[nodes[x]][nodes[y]] = distance_matrix[x][y]

		return node_distance

	# Get the k nearest neighbors of each node
	def __get_k_neartest(self, distance_matrix, k = 5):
		nearest_nodes = {}
		for n in distance_matrix:
			sorted_dist = sorted(distance_matrix[n].items(), key=operator.itemgetter(1))
			nearest_nodes[n] = [ v[0] for v in sorted_dist[:k] ]
		return nearest_nodes

