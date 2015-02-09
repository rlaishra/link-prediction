# Preprocessing of data like detect outliers etc

import networkx as nx 
import math, fileio
from scipy.spatial import distance
from pprint import pprint

class Preprocess(object):
	
	def __init__(self):
		super(Preprocess, self).__init__()
		
	# Detect nodes that are outliers based on in degree and out degree
	# Uses density based outlier detection
	def outlier_nodes(self, adjacency_list, k=5):
		in_degree = self.__in_degree(adjacency_list)
		out_degree = self.__out_degree(adjacency_list)

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
			#fio.save_reprocess_distance_matrix(distance_matrix)
		else:
			distance_matrix = fio.read_reprocess_distance_matrix()

		return distance_matrix

	# Get the in degree from the adjacency list 
	def __out_degree(self, adjacency_list):
		out_degree = {}

		for node in adjacency_list:
			out_degree[node] = len(adjacency_list[node])

		return out_degree

	# Caluculate the out degree from adjacency list
	def __in_degree(self, adjacency_list):
		in_degree = {}

		for x in adjacency_list:
			for y in adjacency_list[x]:
				if y not in in_degree:
					in_degree[y] = 1
				else:
					in_degree[y] += in_degree[y]

		return in_degree

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
				node_distance[nodes[x]][nodes[y]] = distance_matrix[x][y]

		return node_distance
