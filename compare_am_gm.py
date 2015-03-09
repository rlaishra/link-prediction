# Compare measures calculated using no weight, arithmetic mean of weight and geometric means

from __future__ import division
import math, operator, random, numpy, networkx
from libs import database, graph, data
from scipy import linalg
from pprint import pprint
from sklearn import svm
from sklearn.metrics import classification_report

class MeasuresNoWeight():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None
		self._shortest_path = None
		self._katz_score = None
		self._page_rank = None

	def common_neighbors(self, nodes=None):
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

	def adamic_adar(self, nodes=None):
		self._adamic_adar = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._adamic_adar[(nodes[i], nodes[j])] = sum([1/(math.log(max(1.01,len(graph[z])))) for z in neighbors_out ]) + sum([1/(math.log(max(1.01,len(rgraph[z])))) for z in neighbors_in ])
				self._adamic_adar[(nodes[j], nodes[i])] = self._adamic_adar[(nodes[i], nodes[j])]
		return self._adamic_adar

	def preferentail_attachment(self, nodes=None):
		self._preferentail_attachment = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				self._preferentail_attachment[(nodes[i], nodes[j])] = len(graph[nodes[i]])*len(graph[nodes[j]]) + len(rgraph[nodes[i]])*len(rgraph[nodes[j]])
				self._preferentail_attachment[(nodes[j], nodes[i])] = self._preferentail_attachment[(nodes[i], nodes[j])]
		return self._preferentail_attachment

	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				neighbors_out_inter = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in_inter = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				neighbors_out_union = list(set.union(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in_union = list(set.union(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._jaccard_coefficient[(nodes[i], nodes[j])] = len(neighbors_out_inter)/max(1,len(neighbors_out_union)) + len(neighbors_in_inter)/max(1,len(neighbors_in_union))
				self._jaccard_coefficient[(nodes[j], nodes[i])] = self._jaccard_coefficient[(nodes[i], nodes[j])]

		return self._jaccard_coefficient

	def shortest_path(self, nodes=None):
		self._shortest_path = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						shortest = networkx.shortest_path(graph, source=nodes[i], target=nodes[j])
						score = 1/(math.log(1+len(shortest)))
					self._shortest_path[(nodes[i], nodes[j])] = score
		return self._shortest_path

	def katz_score(self, nodes=None):
		beta = 0.05

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						paths = networkx.all_simple_paths(graph, source=nodes[i], target=nodes[j], cutoff=3)
						for path in paths:
							path_length = len(path) - 1
							score += math.pow(beta, path_length)
					self._katz_score[(nodes[i], nodes[j])] = score
		return self._katz_score

	def page_rank(self, nodes=None):
		alpha = 0.7

		self._page_rank = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		# Construct the walk matrix
		walk_matrix = {}

		# Start for each node as root
		pr = {}
		for root in nodes:
			pr[root] = {}
			w_matrix = {}
			all_nodes = []
			
			# Do random walk (1000 runs)
			current_node = root
			current_path_nodes = [current_node]
			for _ in xrange(0,1000):
				if current_node not in all_nodes:
					all_nodes.append(current_node)

				if random.random() > alpha:
					neighbors = graph[current_node].keys()
					#pprint(neighbors)
					if len(neighbors) > 0 :
						random.shuffle(neighbors)
						#pprint(neighbors)
						next_node = neighbors[0]
						current_path_nodes.append(next_node)

						# Update the walk matrix
						if current_node not in w_matrix:
							w_matrix[current_node] = {}
						if next_node not in w_matrix[current_node]:
							w_matrix[current_node][next_node] = 0
						w_matrix[current_node][next_node] += 1
						
						current_node = next_node
					else:
						# Node has no neighbor
						# Reset
						# Update the walk matrix
						if current_node not in w_matrix:
							w_matrix[current_node] = {}
						if root not in w_matrix[current_node]:
							w_matrix[current_node][root] = 0
						w_matrix[current_node][root] += 1

						current_node = root
						current_path_nodes = [root]
				else:
					if current_node not in w_matrix:
						w_matrix[current_node] = {}
					if root not in w_matrix[current_node]:
						w_matrix[current_node][root] = 0
					w_matrix[current_node][root] += 1

					current_path_nodes = [root]
					current_node = root
			all_nodes = sorted(all_nodes)
			
			# Construct transition probability matrix
			t_matrix = []
			for n1 in all_nodes:
				if n1 not in w_matrix:
					row = [0.0] * len(all_nodes)
				else:
					row = []
					for n2 in all_nodes:
						if n2 in w_matrix[n1]:
							row.append(w_matrix[n1][n2])
						else:
							row.append(0)
					# Make values stochastic
					s = sum(row)
					if s > 0 :
						row = [x/s for x in row]
				t_matrix.append(row)

			s_matrix = numpy.array(t_matrix)
			v = linalg.eig(t_matrix, left=True, right=False)
			
			dominant = v[0].tolist().index(max(v[0]))
			r_pr = v[1][:,dominant]/sum(v[1][:,dominant])

			row = {}
			for i in xrange(0, len(all_nodes)-1):
				if (all_nodes[i] != root) :
					row[all_nodes[i]] = r_pr[i]

			for i in xrange(0, len(all_nodes)):
				if (all_nodes[i] in nodes) and (all_nodes[i] != root):
					self._page_rank[(root, all_nodes[i])] = r_pr[i]

		return self._page_rank

class MeasuresWeightAM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None
		self._shortest_path = None

	def common_neighbors(self, nodes=None):
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

				self._common_neighbors[(nodes[i], nodes[j])] = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/2 for x in neighbors_out]) + sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/2 for x in neighbors_in])
				self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]

		return self._common_neighbors

	def adamic_adar(self, nodes=None):
		self._adamic_adar = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				#adamic_adar_out = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/(2 * math.log(max(1.01, sum([graph[x][z]['weight'] for z in graph[x] ])))) for x in neighbors_out ])
				#adamic_adar_in = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/(2 * math.log(max(1.01, sum([rgraph[x][z]['weight'] for z in rgraph[x] ])))) for x in neighbors_in ])

				adamic_adar_out = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([graph[x][z]['weight'] for z in graph[x] ]))) for x in neighbors_out ])
				adamic_adar_in = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([rgraph[x][z]['weight'] for z in rgraph[x] ]))) for x in neighbors_in ])

				self._adamic_adar[(nodes[i], nodes[j])] = adamic_adar_in + adamic_adar_out
				self._adamic_adar[(nodes[j], nodes[i])] = self._adamic_adar[(nodes[i], nodes[j])]
		return self._adamic_adar

	def preferentail_attachment(self, nodes=None):
		self._preferentail_attachment = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				self._preferentail_attachment[(nodes[i], nodes[j])] = sum([ graph[nodes[i]][x]['weight'] for x in graph[nodes[i]] ])*sum([ graph[nodes[j]][x]['weight'] for x in graph[nodes[j]] ]) + sum([ rgraph[nodes[i]][x]['weight'] for x in rgraph[nodes[i]] ])*sum([ rgraph[nodes[j]][x]['weight'] for x in rgraph[nodes[j]] ])
				self._preferentail_attachment[(nodes[j], nodes[i])] = self._preferentail_attachment[(nodes[i], nodes[j])]
		return self._preferentail_attachment

	# Extended jaccard coefficient
	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				neighbors_out_union = list(set.union(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in_union = list(set.union(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				x_out = [ graph[nodes[i]][x]['weight'] if x in graph[nodes[i]].keys() else 0 for x in neighbors_out_union ]
				y_out = [ graph[nodes[j]][x]['weight'] if x in graph[nodes[j]].keys() else 0 for x in neighbors_out_union ]

				x_in = [ rgraph[nodes[i]][x]['weight'] if x in rgraph[nodes[i]].keys() else 0 for x in neighbors_in_union ]
				y_in = [ rgraph[nodes[j]][x]['weight'] if x in rgraph[nodes[j]].keys() else 0 for x in neighbors_in_union ]

				jac_out = 0
				jac_in = 0

				if len(neighbors_out_union) > 0 :
					jac_out = numpy.dot(x_out, y_out)/max(1,(math.pow(numpy.linalg.norm(x_out),2) + math.pow(numpy.linalg.norm(x_out),2) - numpy.dot(x_out,y_out)))
				
				if len(neighbors_in_union) > 0 :
					jac_in = numpy.dot(x_in, y_in)/max(1,(math.pow(numpy.linalg.norm(x_in),2) + math.pow(numpy.linalg.norm(y_in),2) - numpy.dot(x_in,y_in)))
				
				self._jaccard_coefficient[(nodes[i], nodes[j])] = jac_out + jac_in
				self._jaccard_coefficient[(nodes[j], nodes[i])] = self._jaccard_coefficient[(nodes[i], nodes[j])]
		return self._jaccard_coefficient

	def shortest_path(self, nodes=None):
		self._shortest_path = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						shortest_list = networkx.all_shortest_paths(graph, source=nodes[i], target=nodes[j])
						
						path_length = 0
						paths_count = 0
						for shortest in shortest_list:
							paths_count += 1
							path_length = len(shortest) - 1
							score_temp = 0
							for x in xrange(1, len(shortest)) :
								score_temp += graph[shortest[x-1]][shortest[x]]['weight']
							score += score_temp/path_length
						# Since there is atleast one path between the nodes, [0] is safe
						# inverse sqaure of path length to make pat length more important
						score = score/(paths_count*path_length)
						#pprint((score, paths_count, path_length))
						
					self._shortest_path[(nodes[i], nodes[j])] = score
		return self._shortest_path

	def katz_score(self, nodes=None):
		beta = 0.5

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						paths = networkx.all_simple_paths(graph, source=nodes[i], target=nodes[j], cutoff=3)
						for path in paths:
							path_length = len(path) - 1
							score_temp = 0
							for x in xrange(1, len(path)) :
								score_temp += graph[path[x-1]][path[x]]['weight']
							#score += math.pow(beta, path_length) * score_temp/path_length
							score += math.pow(beta, path_length) * score_temp
					self._katz_score[(nodes[i], nodes[j])] = score
		return self._katz_score

	def page_rank(self, nodes=None):
		alpha = 0.7

		self._page_rank = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		# Construct the walk matrix
		walk_matrix = {}

		# Start for each node as root
		pr = {}
		for root in nodes:
			pr[root] = {}
			w_matrix = {}
			all_nodes = []
			
			# Do random walk (1000 runs)
			current_node = root
			current_path_nodes = [current_node]
			for _ in xrange(0,1000):
				if current_node not in all_nodes:
					all_nodes.append(current_node)

				if random.random() > alpha:
					neighbors = graph[current_node].keys()
					if len(neighbors) > 0 :
						# Select a neighbor at random to walk to
						weight_sum = sum([graph[current_node][x]['weight'] for x in neighbors ])
						weighted_list = {}
						last_sum = 0
						for n in neighbors:
							if n not in current_path_nodes:
								weighted_list[n] = {'min': last_sum, 'max': last_sum+graph[current_node][n]['weight']/weight_sum}
								last_sum += graph[current_node][n]['weight']/weight_sum
						
						rand = random.random()
						next_node = None	# Initilize with a node

						for n in neighbors:
							if n not in current_path_nodes and weighted_list[n]['min'] <= rand and weighted_list[n]['max'] > rand:
								next_node = n
								current_path_nodes.append(next_node)
								break

						# Reset to root if no next node is found
						if next_node is None :
							next_node = root
							current_path_nodes = [root]

						# Update the walk matrix
						if current_node not in w_matrix:
							w_matrix[current_node] = {}
						if next_node not in w_matrix[current_node]:
							w_matrix[current_node][next_node] = 0
						w_matrix[current_node][next_node] += 1
						
						current_node = next_node
					else:
						# Node has no neighbor
						# Reset
						# Update the walk matrix
						if current_node not in w_matrix:
							w_matrix[current_node] = {}
						if root not in w_matrix[current_node]:
							w_matrix[current_node][root] = 0
						w_matrix[current_node][root] += 1

						current_node = root
						current_path_nodes = [root]
				else:
					if current_node not in w_matrix:
						w_matrix[current_node] = {}
					if root not in w_matrix[current_node]:
						w_matrix[current_node][root] = 0
					w_matrix[current_node][root] += 1

					current_path_nodes = [root]
					current_node = root
			all_nodes = sorted(all_nodes)
			
			# Construct transition probability matrix
			t_matrix = []
			for n1 in all_nodes:
				if n1 not in w_matrix:
					row = [0.0] * len(all_nodes)
				else:
					row = []
					for n2 in all_nodes:
						if n2 in w_matrix[n1]:
							row.append(w_matrix[n1][n2])
						else:
							row.append(0)
					# Make values stochastic
					s = sum(row)
					if s > 0 :
						row = [x/s for x in row]
				t_matrix.append(row)

			s_matrix = numpy.array(t_matrix)
			v = linalg.eig(t_matrix, left=True, right=False)
			
			dominant = v[0].tolist().index(max(v[0]))
			r_pr = v[1][:,dominant]/sum(v[1][:,dominant])

			row = {}
			for i in xrange(0, len(all_nodes)-1):
				if (all_nodes[i] != root) :
					row[all_nodes[i]] = r_pr[i]

			for i in xrange(0, len(all_nodes)):
				if (all_nodes[i] in nodes) and (all_nodes[i] != root):
					self._page_rank[(root, all_nodes[i])] = numpy.real(r_pr[i])

		return self._page_rank

class MeasuresWeightGM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None

	def common_neighbors(self, nodes=None):
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

				self._common_neighbors[(nodes[i], nodes[j])] = sum([ math.sqrt(graph[nodes[i]][x]['weight']*graph[nodes[j]][x]['weight']) for x in neighbors_out]) + sum([ math.sqrt(rgraph[nodes[i]][x]['weight']*rgraph[nodes[j]][x]['weight']) for x in neighbors_in])
				self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]

		return self._common_neighbors

	def adamic_adar(self, nodes=None):
		self._adamic_adar = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				#adamic_adar_out = sum([ math.sqrt(graph[nodes[i]][x]['weight'] * graph[nodes[j]][x]['weight'])/(math.log(1.001 + sum([graph[x][z]['weight'] for z in graph[x] ]))) for x in neighbors_out ])
				#adamic_adar_in = sum([ math.sqrt(rgraph[nodes[i]][x]['weight'] * rgraph[nodes[j]][x]['weight'])/(math.log(1.001 + sum([rgraph[x][z]['weight'] for z in rgraph[x] ]))) for x in neighbors_in ])

				adamic_adar_out = sum([ math.sqrt(graph[nodes[i]][x]['weight'] * graph[nodes[j]][x]['weight'])/(geometric_sum([graph[x][z]['weight'] for z in graph[x] ])) for x in neighbors_out ])
				adamic_adar_in = sum([ math.sqrt(rgraph[nodes[i]][x]['weight'] * rgraph[nodes[j]][x]['weight'])/(geometric_sum([rgraph[x][z]['weight'] for z in rgraph[x] ])) for x in neighbors_in ])

				self._adamic_adar[(nodes[i], nodes[j])] = adamic_adar_in + adamic_adar_out
				self._adamic_adar[(nodes[j], nodes[i])] = self._adamic_adar[(nodes[i], nodes[j])]
		return self._adamic_adar

	def shortest_path(self, nodes=None):
		self._shortest_path = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						shortest_list = networkx.all_shortest_paths(graph, source=nodes[i], target=nodes[j])
						score = 0
						path_length = 0
						paths_count = 0
						for shortest in shortest_list:
							paths_count += 1
							path_length = len(shortest) - 1 	# Number of paths is 1 less than number of nodes
							temp_score = 1
							for x in xrange(1, len(shortest)) :
								temp_score = temp_score * graph[shortest[x-1]][shortest[x]]['weight']
							score += math.pow(temp_score, 1/path_length)

						score = score/(paths_count*path_length)
						#pprint((score, paths_count, path_length))
						
					self._shortest_path[(nodes[i], nodes[j])] = score
		return self._shortest_path

	def katz_score(self, nodes=None):
		beta = 0.5

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						paths = networkx.all_simple_paths(graph, source=nodes[i], target=nodes[j], cutoff=3)
						for path in paths:
							path_length = len(path) - 1
							score_temp = 1
							for x in xrange(1, len(path)) :
								score_temp *= graph[path[x-1]][path[x]]['weight']
							score += math.pow(beta, path_length) * math.pow(score_temp, 1/path_length)
					#if score > 0:
					#	print(nodes[i], nodes[j], score)
					self._katz_score[(nodes[i], nodes[j])] = score
		return self._katz_score

# Sort the edges by the measure score
# Score of 0 is not ranked
# Largest to smallest
def rank_list(v):
	s = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
	rank_list = {}
	prev = None
	r = 1
	c = 0
	for x in xrange(0, len(s)):
		if s[x][1] == 0:
			continue
		if prev is None or s[x][1] < prev :
			r += c 
			c = 0
			prev = s[x][1]
		rank_list[(s[x][0])] = r
		c += 1
	#pprint(rank_list)
	return rank_list

# Median rank of for the measure
# R is cutoff rank for recall; nodes with rank less than or equal to R are considered as predicted
# Less is better
def get_recall(predicted_list, actual_list, R=50, directed=False):
	pre = predicted_list
	ranks = []
	predicted_list = rank_list(predicted_list)
	for n1 in actual_list.keys():
		for n2 in actual_list[n1].keys():
			if actual_list[n1][n2] < 1 :
				continue
			if (n1, n2) in predicted_list.keys():
				ranks.append(predicted_list[(n1,n2)])
				#pprint(((n1,n2), pre[(n1,n2)], predicted_list[(n1,n2)]))
			elif not directed and (n2, n1) in predicted_list.keys():
				ranks.append(predicted_list[(n1,n2)])
				#pprint(((n1,n2), pre[(n1,n2)], predicted_list[(n1,n2)]))
	ranks = sorted(ranks)

	count = 0
	R = len(ranks)
	for r in ranks:
		if r > R:
			break
		count += 1
	#pprint(len(ranks))
	#pprint(ranks[:count])
	return count/len(ranks)

# Get the product of the items in the list
def product_list(val):
	product = 1
	for x in val:
		product = product * x
	return product

# Get the geometric mean of numnybers in a list
def geometric_mean(val):
	if len(val) < 1 :
		return 1
	product = product_list(val)
	return math.pow(product, 1/len(val))

# Get the geometric sum of numnybers in a list
def geometric_sum(val):
	return max(0.001, len(val) * geometric_mean(val))

# Set the weights of links that already exist to 0
def filter_existing_links(edges, graph):
	graph_edges = graph.edges()
	for n1 in edges:
		for n2 in edges[n1]:
			# 0.25 is the cutoff
			if ((n1,n2) in graph_edges) and (graph[n1][n2]['weight'] > 0.25) :
				#print(((n1,n2),graph[n1][n2]['weight']))
				edges[n1][n2] = 0
	return edges

def compare_measures():
	vals = data.Data()
	db = database.Database()

	# Users
	users_valid = vals.get_valid_users()
	random.shuffle(users_valid)
	users_sample = users_valid[:10000]

	db.open()
	time_start, time_end = db.get_time_min_max()
	#time_start = 1422289905

	network = graph.SocialNetwork(0.1, users_valid)
	network.initialize_nodes(users_valid)

	# Get edges and construct the graph for 36 hours
	delta_time = 3600 	# COnstruct at 1 hour interval
	print('Constructing network')
	for i in xrange(1,36):
		print(i)
		edges = db.get_links(time_start+(i-1)*delta_time, time_start+i*delta_time, users_valid, True)
		network.add_edges(edges, 0.9)
	
	n_noweight = MeasuresNoWeight(network)
	a_noweight = MeasuresWeightAM(network)
	g_noweight = MeasuresWeightGM(network)

	R = 100

	for _ in xrange(0, 10):
		random.shuffle(users_valid)
		users_sample = users_valid[:10000]

		# Get the remaining edges but only between sample users
		print('Getting future edges')
		edges = db.get_links(time_start+36*delta_time, time_start+72*delta_time, users_sample, True)
		pprint(len(edges))

		edges = filter_existing_links(edges, network.get_graph())
		
		# Construct edges with links list
		users_sample = []
		for n1 in edges.keys():
			users_sample.append(n1)
			for n2 in edges[n1].keys():
				users_sample.append(n2)

		users_sample = list(set(users_sample))

		#print('')
		#print('No weight common neighbors')
		#pred = n_noweight.common_neighbors(users_sample)
		#print('Calculating recall no weight common neighbors')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('AM weight common neighbors')
		#pred = a_noweight.common_neighbors(users_sample)
		#print('Calculating recall AM weight common neighbors')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('GM weight common neighbors')
		#pred = g_noweight.common_neighbors(users_sample)
		#print('Calculating recall GM weight common neighbors')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('No weight adamic adar')
		#pred = n_noweight.adamic_adar(users_sample)
		#print('Calculating recall no weight adamic adar')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('AM weight adamic adar')
		#pred = a_noweight.adamic_adar(users_sample)
		#print('Calculating recall AM weight adamic adar')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('GM weight adamic adar')
		#pred = g_noweight.adamic_adar(users_sample)
		#print('Calculating recall GM weight adamic adar')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('No weight adamic adar')
		#pred = n_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall no weight preferentail attachment')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('AM weight adamic adar')
		#pred = a_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall AM weight preferentail attachment')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('GM weight adamic adar')
		#pred = g_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall GM weight preferentail attachment')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#pred = n_noweight.jaccard_coefficient(users_sample)
		#print('Calculating recall no weight jaccard_coefficient')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('GM weight adamic adar')
		#pred = a_noweight.jaccard_coefficient(users_sample)
		#print('Calculating recall weighted jaccard_coefficient')
		#pprint(get_recall(pred, edges, R=R))
	
		#pred = n_noweight.shortest_path(users_sample)
		#print('Calculating recall not weighted shortest path')
		#pprint(get_recall(pred, edges, R=R, directed=True))

		#pred = a_noweight.shortest_path(users_sample)
		#print('Calculating recall AM weighted shortest path')
		#pprint(get_recall(pred, edges, R=R, directed=True))

		#pred = g_noweight.shortest_path(users_sample)
		#print('Calculating recall GM weighted shortest path')
		#pprint(get_recall(pred, edges, R=R, directed=True))
	
		#pred = n_noweight.katz_score(users_sample)
		#print('Calculating recall not weighted katz score')
		#pprint(get_recall(pred, edges, R=R, directed=True))

		#pred = a_noweight.katz_score(users_sample)
		#print('Calculating recall AM weighted katz score')
		#pprint(get_recall(pred, edges, R=R, directed=True))

		#pred = g_noweight.katz_score(users_sample)
		#print('Calculating recall GM weighted katz score')
		#pprint(get_recall(pred, edges, R=R, directed=True))
		
		pred = n_noweight.page_rank(users_sample)
		print('Calculating recall not weighted rooted page rank')
		pprint(get_recall(pred, edges, R=R, directed=True))

		pred = a_noweight.page_rank(users_sample)
		print('Calculating recall weighted rooted page rank')
		pprint(get_recall(pred, edges, R=R, directed=True))

class Prediction():
	def __init__(self):
		self.vals = data.Data()
		self.db = database.Database()

		# Users
		self.users_valid = self.vals.get_valid_users()
		random.shuffle(self.users_valid)
		self.users_sample = self.users_valid[:10000]

		self.db.open()
		self.time_start, self.time_end = self.db.get_time_min_max()

		self.network = graph.SocialNetwork(0.1, self.users_valid)
		self.network.initialize_nodes(self.users_valid)

		# Get edges and construct the graph for 36 hours
		self.delta_time = 3600 	# COnstruct at 1 hour interval
		
	def update_network(self, i_start, i_end):
		for i in xrange(i_start, i_end):
			print('Time: '+str(i))
			edges = self.db.get_links(self.time_start+(i-1)*self.delta_time, self.time_start+i*self.delta_time, self.users_valid, True)
			self.network.add_edges(edges, 0.9)

	def get_features(self, i_start, i_end, sample=False,one_class=False):
		self.update_network(i_start, i_end)
		edges = self.db.get_links(self.time_start+i_start*self.delta_time, self.time_start+i_end*self.delta_time, self.users_sample, True)
		
		if sample:
			# Construct edges with links list
			self.users_sample = []
			for n1 in edges.keys():
				self.users_sample.append(n1)
				for n2 in edges[n1].keys():
					self.users_sample.append(n2)

		n_noweight = MeasuresNoWeight(self.network)
		a_noweight = MeasuresWeightAM(self.network)
		
		print('Common neighbors')
		common_neighbors = a_noweight.common_neighbors(self.users_sample)
		print('Adamic Adar')
		adamic_adar = a_noweight.adamic_adar(self.users_sample)
		print('Preferential Attachment')
		preferentail_attachment = n_noweight.preferentail_attachment(self.users_sample)
		print('Jaccard Coefficient')
		jaccard_coefficient = a_noweight.jaccard_coefficient(self.users_sample)
		print('Shortest Path')
		shortest_path = n_noweight.shortest_path(self.users_sample)
		print('Katz Score')
		katz_score = n_noweight.katz_score(self.users_sample)
		print('Rooted Page Rank')
		rooted_page_rank = a_noweight.page_rank(self.users_sample)

		features = []
		classes = []

		print('COnstructing features')
		for n1 in self.users_sample :
			for n2 in self.users_sample :
				if n1 != n2 :
					if n1 in edges.keys() and n2 in edges[n1]:
						if one_class:
							continue
						classes.append(1)
					else:
						classes.append(0)

					f = []
					if (n1,n2) in common_neighbors :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in adamic_adar :
						f.append(adamic_adar[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in jaccard_coefficient :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in preferentail_attachment :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in katz_score :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in shortest_path :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					if (n1,n2) in rooted_page_rank :
						f.append(common_neighbors[(n1,n2)])
					else:
						f.append(0)

					features.append(f)
		return features, classes

	def run(self,):
		print('Features 1')
		features1, classes1 = self.get_features(1,24,sample=True, one_class=True)
		print('Features 2')
		features2, classes2 = self.get_features(24,48,sample=False)
		print('Features 3')
		features3, classes3 = self.get_features(48,72,sample=False)
		
		print('Fitting model')
		#clf = svm.LinearSVC(class_weight={1:50, 0:1})
		#clf.fit(features1, classes2)

		clf = svm.OneClassSVM()
		clf.fit(features1)

		print('Predicting')
		#prediction = clf.predict(features2)
		prediction = clf.predict(features2)

		print('Constructing the classification report')
		print(classification_report(classes3, prediction, target_names=['class_0', 'class_1']))


if __name__ == '__main__':
	#compare_measures()
	p = Prediction()
	p.run()
