# Compare measures calculated using no weight, arithmetic mean of weight and geometric means

from __future__ import division
import math, operator, random
from libs import database, graph, data
from pprint import pprint

class MeasuresNoWeight():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None

	def common_neighbors(self, nodes=None):
		if self._common_neighbors is not None:
			# If common neighbors has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If common neighbors has not been calculates, calculate it
			self._common_neighbors = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					self._common_neighbors[(nodes[i], nodes[j])] = len(neighbors_out) + len(neighbors_in)
					self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
			return self._common_neighbors

	def jaccard_coefficient(self, nodes=None):
		if self._jaccard_coefficient is not None:
			# If jaccard coefficient has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If jaccard coefficient has not been calculated, calculate it
			self._jaccard_coefficient = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out_intersection = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in_intersection = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					neighbors_out_union = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in_union = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					self._jaccard_coefficient[(nodes[i], nodes[j])] = len(neighbors_out_intersection)/len(neighbors_out_union) + len(neighbors_in_intersection)/len(neighbors_in_union)
					self._jaccard_coefficient[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
			return self._common_neighbors

class MeasuresWeightAM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None

	def common_neighbors(self, nodes=None):
		if self._common_neighbors is not None:
			# If common neighbors has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If common neighbors has not been calculates, calculate it
			self._common_neighbors = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					self._common_neighbors[(nodes[i], nodes[j])] = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/2 for x in neighbors_out]) + sum([ (graph[x][nodes[i]]['weight'] + graph[x][nodes[j]]['weight'])/2 for x in neighbors_in])
					self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]

			return self._common_neighbors

	def jaccard_coefficient(self, nodes=None):
		if self._jaccard_coefficient is not None:
			# If jaccard coefficient has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If jaccard coefficient has not been calculated, calculate it
			self._jaccard_coefficient = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out_intersection = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in_intersection = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					jaccard_out_numerator = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight']) for x in neighbors_out_intersection])/(sum([ graph[nodes[i]][x]['weight'] for x in graph[nodes[i]] ]) + sum([ graph[nodes[j]][x]['weight'] for x in graph[nodes[j]] ]))
					jaccard_in = sum([ (graph[x][nodes[i]]['weight'] + graph[x][nodes[j]]['weight']) for x in neighbors_in_intersection])/(sum([ graph[x][nodes[i]]['weight'] for x in rgraph[nodes[i]] ]) + sum([ graph[x][nodes[j]]['weight'] for x in rgraph[nodes[j]] ]))


					self._jaccard_coefficient[(nodes[i], nodes[j])] = jaccard_in + jaccard_out
					self._jaccard_coefficient[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
			return self._common_neighbors

class MeasuresWeightGM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None

	def common_neighbors(self, nodes=None):
		if self._common_neighbors is not None:
			# If common neighbors has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If common neighbors has not been calculates, calculate it
			self._common_neighbors = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					self._common_neighbors[(nodes[i], nodes[j])] = sum([ math.sqrt(graph[nodes[i]][x]['weight']*graph[nodes[j]][x]['weight']) for x in neighbors_out]) + sum([ math.sqrt(graph[x][nodes[i]]['weight']*graph[x][nodes[j]]['weight']) for x in neighbors_in])
					self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]

			return self._common_neighbors

	def jaccard_coefficient(self, nodes=None):
		if self._jaccard_coefficient is not None:
			# If jaccard coefficient has been calculated, return common neighbors
			return self._common_neighbors
		else:
			# If jaccard coefficient has not been calculated, calculate it
			self._jaccard_coefficient = {}

			if nodes is None:
				nodes = self._network.get_sample_nodes()
			graph = self._network.get_graph()
			rgraph = self._network.get_reverse_graph()

			for i in xrange(0, len(nodes)):
				for j in xrange(i, len(nodes)):
					neighbors_out_intersection = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
					neighbors_in_intersection = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

					jaccard_out = sum([ math.sqrt(graph[nodes[i]][x]['weight'] * graph[nodes[j]][x]['weight']) for x in neighbors_out_intersection])/(sum([ graph[nodes[i]][x]['weight'] for x in graph[nodes[i]] ]) + sum([ graph[nodes[j]][x]['weight'] for x in graph[nodes[j]] ]))
					jaccard_in = sum([ (graph[x][nodes[i]]['weight'] + graph[x][nodes[j]]['weight']) for x in neighbors_in_intersection])/(sum([ graph[x][nodes[i]]['weight'] for x in rgraph[nodes[i]] ]) + sum([ graph[x][nodes[j]]['weight'] for x in rgraph[nodes[j]] ]))


					self._jaccard_coefficient[(nodes[i], nodes[j])] = jaccard_in + jaccard_out
					self._jaccard_coefficient[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
			return self._common_neighbors

# Sort the edges by the measure score
# Score of 0 is not ranked
# Largest to smallest
def rank_list(v):
	s = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
	rank_list = {}
	prev = None
	r = 0
	c = 0
	for x in xrange(0, len(s)):
		if prev is None or s[x][1] < prev :
			r += c 
			c = 0
			prev = s[x][1]

		rank_list[(s[x][0])] = r
		c += 1

	return rank_list

# Median rank of for the measure
# R is cutoff rank for recall; nodes with rank less than or equal to R are considered as predicted
# Less is better
def get_recall(predicted_list, actual_list, R=50):
	pre = predicted_list
	ranks = []
	predicted_list = rank_list(predicted_list)
	for n1 in actual_list.keys():
		for n2 in actual_list[n1].keys():
			if (n1, n2) in predicted_list.keys():
				ranks.append(predicted_list[(n1,n2)])
				#pprint(((n1,n2), pre[(n1,n2)], predicted_list[(n1,n2)]))
	ranks = sorted(ranks)

	count = 0
	#R = len(ranks)
	for r in ranks:
		if r > R :
			break
		count += 1
	pprint(len(ranks))
	pprint(ranks[:count])
	return count/len(ranks)

def main():
	vals = data.Data()
	db = database.Database()

	# Users
	users_valid = vals.get_valid_users()
	random.shuffle(users_valid)
	users_sample = users_valid[:5000]

	db.open()
	time_start, time_end = db.get_time_min_max()

	network = graph.SocialNetwork(0.1, users_valid)
	network.initialize_nodes(users_valid)

	# Get edges and construct the graph for 36 hours
	delta_time = 3600 	# COnstruct at 1 hour interval
	print('Constructing network')
	for i in xrange(1,36):
		print(i)
		edges = db.get_links(time_start+(i-1)*delta_time, time_start+i*delta_time, users_sample, False)
		network.add_edges(edges, 0.9)
	
	# Get the remaining edges but only between sample users
	print('Getting future edges')
	edges = db.get_links(time_start+36*delta_time, time_start+72*delta_time, users_sample, True)
	pprint(len(edges))

	# Construct edges with links list
	users_sample = []
	for n1 in edges.keys():
		users_sample.append(n1)
		for n2 in edges[n1].keys():
			users_sample.append(n2)

	users_sample = list(set(users_sample))
	pprint(len(users_sample))

	n_noweight = MeasuresNoWeight(network)
	a_noweight = MeasuresWeightAM(network)
	g_noweight = MeasuresWeightGM(network)

	print('No weight common neighbors')
	n_cm = n_noweight.common_neighbors(users_sample)
	print('AM weight common neighbors')
	a_cm = a_noweight.common_neighbors(users_sample)
	print('GM weight common neighbors')
	g_cm = g_noweight.common_neighbors(users_sample)
	
	R = 50
	print('Calculating recall no weight common neighbors')
	pprint(get_recall(n_cm, edges, R=R))
	print('Calculating recall AM weight common neighbors')
	pprint(get_recall(a_cm, edges, R=R))
	print('Calculating recall GM weight common neighbors')
	pprint(get_recall(g_cm, edges, R=R))

if __name__ == '__main__':
	main()
