# Compare measures calculated using no weight, arithmetic mean of weight and geometric means

from __future__ import division
import math, operator, random, numpy, networkx, sys, os.path, csv, time
from libs import database, graph, data
from scipy import linalg
from scipy import stats
from pprint import pprint
from sklearn import svm, grid_search
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import covariance
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn import cross_validation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._common_neighbors[(nodes[i], nodes[j])] = len(neighbors_out) + len(neighbors_in)
				self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
		print('')
		#print(type(self._common_neighbors))
		return self._common_neighbors

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	def adamic_adar(self, nodes=None):
		self._adamic_adar = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._adamic_adar[(nodes[i], nodes[j])] = sum([1/(math.log(max(1.01,len(graph[z])))) for z in neighbors_out ]) + sum([1/(math.log(max(1.01,len(rgraph[z])))) for z in neighbors_in ])
				self._adamic_adar[(nodes[j], nodes[i])] = self._adamic_adar[(nodes[i], nodes[j])]
		print('')
		return self._adamic_adar

	def preferentail_attachment(self, nodes=None):
		self._preferentail_attachment = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				self._preferentail_attachment[(nodes[i], nodes[j])] = len(graph[nodes[i]])*len(graph[nodes[j]]) + len(rgraph[nodes[i]])*len(rgraph[nodes[j]])
				self._preferentail_attachment[(nodes[j], nodes[i])] = self._preferentail_attachment[(nodes[i], nodes[j])]
		print('')
		return self._preferentail_attachment

	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_out_inter = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in_inter = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				neighbors_out_union = list(set.union(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in_union = list(set.union(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._jaccard_coefficient[(nodes[i], nodes[j])] = len(neighbors_out_inter)/max(1,len(neighbors_out_union)) + len(neighbors_in_inter)/max(1,len(neighbors_in_union))
				self._jaccard_coefficient[(nodes[j], nodes[i])] = self._jaccard_coefficient[(nodes[i], nodes[j])]
		print('')
		return self._jaccard_coefficient

	def shortest_path(self, nodes=None):
		self._shortest_path = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						shortest = networkx.shortest_path(graph, source=nodes[i], target=nodes[j])
						score = 1/(math.log(1+len(shortest)))
					self._shortest_path[(nodes[i], nodes[j])] = score
		print('')
		return self._shortest_path

	def katz_score(self, nodes=None):
		beta = 0.005

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(0, len(nodes)):
				if j != i :
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						paths = networkx.all_simple_paths(graph, source=nodes[i], target=nodes[j], cutoff=3)
						for path in paths:
							path_length = len(path) - 1
							score += math.pow(beta, path_length)
					self._katz_score[(nodes[i], nodes[j])] = score
		print('')
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
		for i in xrange(0,len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			root = nodes[i]
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
					self._page_rank[(root, all_nodes[i])] = numpy.real(r_pr[i])

		print('')
		return self._page_rank

class MeasuresWeightAM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors_in 	= None
		self._jaccard_coefficient_in = None
		self._adamic_adar_in = None
		self._preferentail_attachment_in = None
		self._shortest_path_in = None

		self._common_neighbors_out 	= None
		self._jaccard_coefficient_out = None
		self._adamic_adar_out = None
		self._preferentail_attachment_out = None
		self._shortest_path_out = None

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	def common_neighbors(self, nodes=None):
		# If common neighbors has not been calculates, calculate it
		self._common_neighbors_in = {}
		self._common_neighbors_out = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))

				self._common_neighbors_in[(nodes[i], nodes[j])] = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/2 for x in neighbors_in])
				self._common_neighbors_in[(nodes[j], nodes[i])] = self._common_neighbors_in[(nodes[i], nodes[j])]

				self._common_neighbors_out[(nodes[i], nodes[j])] = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/2 for x in neighbors_out])
				self._common_neighbors_out[(nodes[j], nodes[i])] = self._common_neighbors_out[(nodes[i], nodes[j])]

		print('')
		return self._common_neighbors_in, self._common_neighbors_out

	def adamic_adar(self, nodes=None):
		self._adamic_adar_in = {}
		self._adamic_adar_out = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))

				adamic_adar_out = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([graph[x][z]['weight'] for z in graph[x] ]))) for x in neighbors_out ])
				adamic_adar_in = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([rgraph[x][z]['weight'] for z in rgraph[x] ]))) for x in neighbors_in ])

				self._adamic_adar_in[(nodes[i], nodes[j])] = adamic_adar_in
				self._adamic_adar_in[(nodes[j], nodes[i])] = adamic_adar_in

				self._adamic_adar_out[(nodes[i], nodes[j])] = adamic_adar_out
				self._adamic_adar_out[(nodes[j], nodes[i])] = adamic_adar_out

		print('')
		return self._adamic_adar_in, self._adamic_adar_out

	def preferentail_attachment(self, nodes=None):
		self._preferentail_attachment_in = {}
		self._preferentail_attachment_out = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				self._preferentail_attachment_in[(nodes[i], nodes[j])] = sum([ rgraph[nodes[i]][x]['weight'] for x in rgraph[nodes[i]] ])*sum([ rgraph[nodes[j]][x]['weight'] for x in rgraph[nodes[j]] ])
				self._preferentail_attachment_in[(nodes[j], nodes[i])] = self._preferentail_attachment_in[(nodes[i], nodes[j])]

				self._preferentail_attachment_out[(nodes[i], nodes[j])] = sum([ graph[nodes[i]][x]['weight'] for x in graph[nodes[i]] ])*sum([ graph[nodes[j]][x]['weight'] for x in graph[nodes[j]] ])
				self._preferentail_attachment_out[(nodes[j], nodes[i])] = self._preferentail_attachment_out[(nodes[i], nodes[j])]
		
		print('')
		return self._preferentail_attachment_in, self._preferentail_attachment_out

	# Extended jaccard coefficient
	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient_in = {}
		self._jaccard_coefficient_out = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
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
				
				self._jaccard_coefficient_in[(nodes[i], nodes[j])] = jac_in
				self._jaccard_coefficient_in[(nodes[j], nodes[i])] = self._jaccard_coefficient_in[(nodes[i], nodes[j])]

				self._jaccard_coefficient_out[(nodes[i], nodes[j])] = jac_out 
				self._jaccard_coefficient_out[(nodes[j], nodes[i])] = self._jaccard_coefficient_out[(nodes[i], nodes[j])]

		print('')
		return self._jaccard_coefficient_in, self._jaccard_coefficient_out

	def shortest_path(self, nodes=None):
		self._shortest_path = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			for j in xrange(0, len(nodes)):
				if j != i :
					self.cprint(str(i*len(nodes)+(j+1))+'/'+str(len(nodes)*len(nodes)))
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						shortest_list = networkx.all_shortest_paths(graph, source=nodes[i], target=nodes[j])
						path_length = 0
						paths_count = 0
						for shortest in shortest_list:
							paths_count += 1
							if path_length == 0:
								path_length = len(shortest)
							score += sum([ graph[shortest[x-1]][shortest[x]]['weight'] for x in xrange(1, len(shortest)) ])
						# Since there is atleast one path between the nodes, [0] is safe
						# inverse sqaure of path length to make pat length more important
						score = score/(paths_count*path_length*path_length)
						#pprint((score, paths_count, path_length))
						#print('Path count: ' + str(paths_count))
					self._shortest_path[(nodes[i], nodes[j])] = score
		print('')
		return self._shortest_path

	def katz_score(self, nodes=None):
		beta = 0.005

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			#self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(0, len(nodes)):
				if j != i :
					self.cprint(str(i*len(nodes)+(j+1))+'/'+str(len(nodes)*len(nodes)))
					score = 0

					if networkx.has_path(graph, source=nodes[i], target=nodes[j]) :
						paths = networkx.all_simple_paths(graph, source=nodes[i], target=nodes[j], cutoff=3)
						for path in paths:
							path_length = len(path) - 1
							score_temp = 0
							score += math.pow(beta, path_length) * sum([graph[path[x-1]][path[x]]['weight'] for x in xrange(1, len(path))])/path_length
							#for x in xrange(1, len(path)) :
							#	score_temp += graph[path[x-1]][path[x]]['weight']
							#score += math.pow(beta, path_length) * score_temp/path_length
							score += math.pow(beta, path_length) * score_temp
					self._katz_score[(nodes[i], nodes[j])] = score
		print('')
		return self._katz_score

	def page_rank(self, nodes=None):
		alpha = 0.7
		frac = 0.9
		prob_cutoff = 0.05

		self._page_rank = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		# Construct the walk matrix
		walk_matrix = {}

		# Start for each node as root
		pr = {}
		
		lc = int(math.ceil(math.log(1 - frac)/math.log(1 - alpha)))	# The longest path used in modified 
		pr = {}
		for i in xrange(0,len(nodes)):
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			root = nodes[i]
			paths = [[[root]]]
			all_nodes = [root]
			probabilities = {}

			for _ in xrange(1,lc):
				p = []
				for m in paths[-1]:
					if (tuple(m) in probabilities) and (probabilities[tuple(m)]*(1 - alpha) <= prob_cutoff) :
						continue
					o = m[-1]
					for q in graph[o].keys():
						if q not in m:
							p.append(m+[q])
							if q not in all_nodes:
								all_nodes.append(q)
				if len(p) > 0:
					paths.append(p)

				# Calculate the path probabilities
				for p2 in p:
					n1 = p2[-2] # From
					n2 = p2[-1] # To

					# Probability of the path
					if p2[:-1] in probabilities.keys():
						probabilities[tuple(p2)] = probabilities[tuple(p2[:-1])]*(1 - alpha)*graph[n1][n2]['weight']/(sum([graph[n1][x]['weight'] for x in graph[n1]]))
					else:
						probabilities[tuple(p2)] = 1
						for i in xrange(0, len(p2)-1):
							probabilities[tuple(p2)] = probabilities[tuple(p2)]*(1 - alpha)*graph[p2[i]][p2[i+1]]['weight']/(sum([graph[p2[i]][x]['weight'] for x in graph[p2[i]]]))
					
					# The path ends with a teleprotation back to root
					# That teleportation probability
					if tuple(p2 + [root]) not in probabilities.keys():
						probabilities[tuple(p2 + [root])] = probabilities[tuple(p2)]*(1 - alpha)
					else:
						probabilities[tuple(p2 + [root])] = probabilities[tuple(p2)]*(1 - alpha)
			
			paths = sorted(paths[1:])
			
			if len(paths) < 1:
				continue
			
			# Construct the transition matrix
			t_matrix = {}
			for n1 in all_nodes:
				t_matrix[n1] = {}
				for n2 in all_nodes:
					t_matrix[n1][n2] = 0

			for p in probabilities.keys():
				t_matrix[p[-2]][p[-1]] += probabilities[p]

			# Make a stochastic matrix
			s_matrix = []
			for n1 in all_nodes:
				s = sum([t_matrix[n1][x] for x in all_nodes])
				row = [t_matrix[n1][x]/s for x in all_nodes]
				s_matrix.append(row)
			
			s_matrix = numpy.array(s_matrix)
			v = linalg.eig(s_matrix, left=True, right=False)
			
			dominant = v[0].tolist().index(max(v[0]))
			r_pr = v[1][:,dominant]/sum(v[1][:,dominant])

			row = {}
			for i in xrange(0, len(all_nodes)-1):
				if (all_nodes[i] != root) :
					row[all_nodes[i]] = r_pr[i]

			for i in xrange(0, len(all_nodes)):
				if (all_nodes[i] in nodes) and (all_nodes[i] != root):
					self._page_rank[(root, all_nodes[i])] = numpy.real(r_pr[i])

		print('')
		return self._page_rank

class MeasuresWeightGM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	def common_neighbors(self, nodes=None):
		# If common neighbors has not been calculates, calculate it
		self._common_neighbors = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			print(str(i+1)+'/'+str(len(nodes)))
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
			print(str(i+1)+'/'+str(len(nodes)))
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
			print(str(i+1)+'/'+str(len(nodes)))
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
		beta = 0.005

		self._katz_score = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			print(str(i+1)+'/'+str(len(nodes)))
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

	# Extended jaccard coefficient
	# Same as AM
	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient = {}

		if nodes is None:
			nodes = self._network.get_sample_nodes()
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		for i in xrange(0, len(nodes)):
			print(str(i+1)+'/'+str(len(nodes)))
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

	# Same as AM
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
		for i in xrange(0,len(nodes)):
			print(str(i+1)+'/'+str(len(nodes)))
			root = nodes[i]
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
def get_recall(predicted_list, actual_list, R=50, directed=False, predicted_list2 = []):
	pre = predicted_list
	ranks = []

	if len(predicted_list2) > 0 :
		predicted_list = rank_list_agg(predicted_list, predicted_list2)
	else:
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

# Aggregated rank list
def rank_list_agg(v1, v2):
	s1 = sorted(v1.items(), key=operator.itemgetter(1), reverse=True)
	s2 = sorted(v2.items(), key=operator.itemgetter(1), reverse=True)

	rank_list1 = {}
	rank_list2 = {}
	rank_list = {}
	prev = None
	r = 1
	c = 0
	s1_edges = []
	for x in xrange(0, len(s1)):
		if (s1[x][1] == 0) or (s1[x][0][0] == s1[x][0][1]):
			continue
		if prev is None or s1[x][1] < prev :
			r += c 
			c = 0
			prev = s1[x][1]
		rank_list1[(s1[x][0])] = r
		s1_edges.append((s1[x][0]))
		c += 1
	prev = None
	r = 1
	c = 0
	s2_edges = []
	for x in xrange(0, len(s2)):
		if (s2[x][1] == 0) or (s2[x][0][0] == s2[x][0][1]):
			continue
		if prev is None or s2[x][1] < prev :
			r += c 
			c = 0
			prev = s2[x][1]
		rank_list2[(s2[x][0])] = r
		s2_edges.append((s2[x][0]))
		c += 1

	edge_list = list(set(s1_edges).intersection(set(s2_edges)))

	# Combile the two rank lists
	for edge in edge_list:
		rank_list[edge] = (rank_list1[edge] + rank_list2[edge])/2

	#pprint(rank_list)
	return rank_list

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
	#time_start, time_end = db.get_time_min_max()
	time_start = 1422289905

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
		#pprint(get_recall(pred, edges, directed=True, R=R))

		#print('')
		#print('AM weight common neighbors')
		#pred1, pred2 = a_noweight.common_neighbors(users_sample)
		#print('Calculating recall AM weight common neighbors')
		#pprint(get_recall(pred, edges, directed=True, R=R, predicted_list2=pred2))

		#print('')
		#print('GM weight common neighbors')
		#pred1, pred2 = g_noweight.common_neighbors(users_sample)
		#print('Calculating recall GM weight common neighbors')
		#pprint(get_recall(pred, edges, R=R, predicted_list2=pred2))

		#print('')
		#print('No weight adamic adar')
		#pred = n_noweight.adamic_adar(users_sample)
		#print('Calculating recall no weight adamic adar')
		#pprint(get_recall(pred, edges, directed=True, R=R))

		#print('')
		#print('AM weight adamic adar')
		#pred1, pred2 = a_noweight.adamic_adar(users_sample)
		#print('Calculating recall AM weight adamic adar')
		#pprint(get_recall(pred, edges, directed=True, R=R, predicted_list2=pred2))

		#print('')
		#print('GM weight adamic adar')
		#pred = g_noweight.adamic_adar(users_sample)
		#print('Calculating recall GM weight adamic adar')
		#pprint(get_recall(pred, edges, R=R))

		#print('')
		#print('No weight adamic adar')
		#pred = n_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall no weight preferentail attachment')
		#pprint(get_recall(pred, edges, directed=True, R=R))

		#print('')
		#print('AM weight adamic adar')
		#pred1, pred2 = a_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall AM weight preferentail attachment')
		#pprint(get_recall(pred, edges, directed=True, R=R, predicted_list2=pred2))

		#print('')
		#print('GM weight adamic adar')
		#pred = g_noweight.preferentail_attachment(users_sample)
		#print('Calculating recall GM weight preferentail attachment')
		#pprint(get_recall(pred, edges, R=R))

		print('')
		pred = n_noweight.jaccard_coefficient(users_sample)
		print('Calculating recall no weight jaccard_coefficient')
		pprint(get_recall(pred, edges, directed=True, R=R))

		print('')
		print('GM weight adamic adar')
		pred1, pred2 = a_noweight.jaccard_coefficient(users_sample)
		print('Calculating recall weighted jaccard_coefficient')
		pprint(get_recall(pred, edges, directed=True, R=R, predicted_list2=pred2))
	
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
		
		#pred = n_noweight.page_rank(users_sample)
		#print('Calculating recall not weighted rooted page rank')
		#pprint(get_recall(pred, edges, R=R, directed=True))

		#pred = a_noweight.page_rank(users_sample)
		#print('Calculating recall weighted rooted page rank')
		#pprint(get_recall(pred, edges, R=R, directed=True))

class Prediction():
	def __init__(self,N=10000,t1=1,t2=72,t3=144,t4=216, m_type=None, f1='3', f2='100', f3='11'):
		self.community_size = int(f1)
		self.a1 = f1
		self.a2 = f2
		self.a3 = f3

		self.vals = data.Data()
		self.db = database.Database()

		# Users
		self.users_valid = self.vals.get_valid_users()
		random.shuffle(self.users_valid)
		self.users_sample = self.users_valid[:N]

		self.db.open()
		#self.time_start, self.time_end = self.db.get_time_min_max()
		self.time_start = 1422289905
		#self.time_start = 788918400 	# For cond-mat

		self.network = graph.SocialNetwork(0.1, self.users_valid, weighted=True)
		self.network.initialize_nodes(self.users_valid)

		# Get edges and construct the graph for 36 hours
		self.delta_time = 3600 	# COnstruct at 1 hour interval
		#self.delta_time = 31536000 # One year for cond-mat

		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.t4 = t4

		self.sample_size = N

		self.directory = 'cache/'

		self.f1 = 's-cache-learning-features-k'+f1+'-'+f2+'-168-'+m_type+'-'+f3+'.csv'
		self.c1 = 's-cache-learning-class-k'+f1+'-'+f2+'-168-'+m_type+'-'+f3+'.csv'
		self.f2 = 's-cache-test-features-k'+f1+'-'+f2+'-168-'+m_type+'-'+f3+'.csv'
		self.c2 = 's-cache-test-class-k'+f1+'-'+f2+'-168-'+m_type+'-'+f3+'.csv'

		#self.f1 = 'cache-learning-features-test-data.csv'
		#self.c1 = 'cache-learning-class-test-data.csv'
		#self.f2 = 'cache-test-features-test-data.csv'
		#self.c2 = 'cache-test-class-test-data.csv'

		# Type of measure used
		if m_type in ['no', 'am', 'gm']:
			self.m_type = m_type
		else:
			self.m_type = 'am'

		print()

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	def update_network(self, i_start, i_end):
		print('Constructing Network: ')
		for i in xrange(i_start, i_end):
			self.cprint(str(int((i-i_start+1)*100/(i_end - i_start)))+'%'),
			edges = self.db.get_links(self.time_start+(i-1)*self.delta_time, self.time_start+i*self.delta_time, self.users_valid, True)
			self.network.add_edges(edges, 0.9)
		print('')
	
	# Get nodes in k-clique communities of max size
	# if n is int, return nodes in all communities of size > n
	def community_clique(self, k=4, n=False):
		communities = networkx.k_clique_communities(self.network.get_graph().to_undirected(), k)
		pprint(k)

		if not n:
			max_size = 0
			nodes = []
			for c in communities:
				if len(c) > max_size:
					nodes = c
					max_size = len(c)

			nodes = list(nodes)
			nodes = list(set.intersection(set(nodes), set(self.users_valid)))
		else:
			nodes = []
			for c in communities:
				if len(c) > n:
					nodes += list(c)

			nodes = list(set(nodes))
			nodes = list(set.intersection(set(nodes), set(self.users_valid)))

		random.shuffle(nodes)

		print('Community size: ' + str(len(nodes)))

		if self.sample_size:
			self.users_sample = nodes[:self.sample_size]
		else:
			self.users_sample = nodes

		self.save_community_nodes()

	def get_sample(self, edges):
		# Construct edges with links list
		self.users_sample = []
		for n1 in edges.keys():
			self.users_sample.append(n1)
			for n2 in edges[n1].keys():
				self.users_sample.append(n2)

	def get_last_class(self, edges, one_class=False):
		classes = []

		for n1 in self.users_sample :
			for n2 in self.users_sample :
				if n1 != n2 :
					if n1 in edges.keys() and n2 in edges[n1]:
						if one_class:
							continue
						classes.append(1)
					else:
						classes.append(0)
		return classes


	def construct_features(self, edges, common_neighbors, adamic_adar, preferentail_attachment, jaccard_coefficient, shortest_path, katz_score, rooted_page_rank, one_class=False):
		features = []
		classes = []
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
					for d in common_neighbors:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in adamic_adar:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in jaccard_coefficient:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in preferentail_attachment:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in katz_score:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in shortest_path:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					for d in rooted_page_rank:
						if (n1,n2) in d :
							f.append(d[(n1,n2)])
						else:
							f.append(0)

					features.append(f)
		return features, classes

	def get_features(self, i_start, i_end, sample=False,one_class=False,last=False, comm_sample=False):
		self.update_network(i_start, i_end)
		edges = self.db.get_links(self.time_start+i_start*self.delta_time, self.time_start+i_end*self.delta_time, self.users_sample, True)
		
		if sample:
			self.get_sample(edges)

		if comm_sample:
			self.community_clique(self.community_size, n=True)

		features = []
		classes = []

		if last:
			classes = self.get_last_class(edges, one_class=one_class)
			return features, classes

		if self.m_type == 'no':
			mes = MeasuresNoWeight(self.network)
		elif self.m_type == 'am':
			mes = MeasuresWeightAM(self.network)
		else:
			mes = MeasuresWeightGM(self.network)

		common_neighbors = []
		adamic_adar = []
		preferentail_attachment = []
		jaccard_coefficient = []
		rooted_page_rank = []
		katz_score = []
		shortest_path = []

		if self.m_type == 'am':
			print('Common neighbors')
			mes_in, mes_out = mes.common_neighbors(self.users_sample)
			common_neighbors.append(mes_in)
			common_neighbors.append(mes_out)
			print('Adamic Adar')
			mes_in, mes_out = mes.adamic_adar(self.users_sample)
			adamic_adar.append(mes_in)
			adamic_adar.append(mes_out)
			print('Preferential Attachment')
			mes_in, mes_out = mes.preferentail_attachment(self.users_sample)
			preferentail_attachment.append(mes_in)
			preferentail_attachment.append(mes_out)
			print('Jaccard Coefficient')
			mes_in, mes_out = mes.jaccard_coefficient(self.users_sample)
			jaccard_coefficient.append(mes_in)
			jaccard_coefficient.append(mes_out)
		else:
			print('Common neighbors')
			common_neighbors.append(mes.common_neighbors(self.users_sample))
			print('Adamic Adar')
			adamic_adar.append(mes.adamic_adar(self.users_sample))
			print('Preferential Attachment')
			preferentail_attachment.append(mes.preferentail_attachment(self.users_sample))
			print('Jaccard Coefficient')
			jaccard_coefficient.append(mes.jaccard_coefficient(self.users_sample))
		
		if self.m_type != 'no' or True:
			print('Rooted Page Rank')
			rooted_page_rank.append(mes.page_rank(self.users_sample))
			print('Katz Score')
			katz_score.append(mes.katz_score(self.users_sample))
			print('Shortest Path')
			shortest_path.append(mes.shortest_path(self.users_sample))
			
		else:
			rooted_page_rank.append([0]*len(jaccard_coefficient))
			katz_score.append([0]*len(jaccard_coefficient))
			shortest_path.append([0]*len(jaccard_coefficient))

		print('Constructing features')
		features, classes = self.construct_features(edges, common_neighbors, adamic_adar, preferentail_attachment, jaccard_coefficient, shortest_path, katz_score, rooted_page_rank, one_class=one_class)

		return features, classes

	def learn_get_f1_prob(self, features1, features2, classes1, classes2,w1=1,w0=1, algo='svm'):
		
		prediction = None

		if algo == 'svm':
			for i in xrange(0, 10):
				print('Loop: '+ str(i))
				clf = svm.SVC(class_weight={1:w1, 0:w0},kernel="rbf", probability=True)
				f, c ,t1, t2 = self.balance_data(features1, classes1, N=5)
				clf.fit(f, c)

				p = clf.predict_proba(features2)

				if prediction is None:
					prediction = [x[1] for x in p] 
				else:
					prediction = [ (prediction[i]*p[i][1]) for i in xrange(0, len(p)) ]
		elif algo == 'tree':
			clf = DecisionTreeClassifier()
			f, c ,t1, t2 = self.balance_data(features1, classes1, N=5)
			clf.fit(f, c)

			p = clf.predict_proba(features2)
			prediction = [ p[i][1] for i in xrange(0, len(p)) ]


		auc = roc_auc_score(classes2, prediction, average='weighted')
		pprint(auc)

		self.sample_data()

		# Plot the ROC curve and save
		fpr, tpr, _ = roc_curve(classes2, prediction, pos_label=1 )
		plt.figure()
		plt.plot(fpr, tpr)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.savefig('img/roc-mtype-'+self.m_type+'-auc-'+str(auc)+'-k-'+str(self.a1)+'-density-'+str(self.density)+'.png')

	# Set weight of class 0; between 1 and 0
	def learn_get_f1(self, weight0, features1, features2, classes1, classes2, report=False, algorithm='svm',w1=1,w0=1):
		print('Weight 0: ' + str(weight0))
		
		print('Fitting model')

		if algorithm == 'svm_iterate':
			clf = svm.SVC(class_weight={1:1, 0:weight0},kernel="linear")
		elif algorithm == 'decison_tree':
			clf = DecisionTreeClassifier()
		elif algorithm == 'nbc':
			clf = GaussianNB()
		elif algorithm == 'adaboost':
			clf = AdaBoostClassifier()
		elif algorithm == 'random_forest':
			clf = RandomForestClassifier()
		elif algorithm == 'svm':
			#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
			#svr = svm.SVC()
			#clf = grid_search.GridSearchCV(svr, parameters)
			clf = svm.SVC(class_weight={1:w1, 0:w0},kernel="rbf", probability=True)

		clf.fit(features1, classes1)

		print('Predicting')
		prediction1 = clf.predict(features2)
		print(sum(prediction1))
		#print(prediction[:5])
		prediction = clf.predict_proba(features2)
		#print(prediction[:5])
		
		#print(prediction[:5])

		s0 = []
		s1 = []
		for i in xrange(0, len(classes2)):
			if classes2[i] == 1:
				#print(prediction[i])
				s1.append(prediction[i][1])
			else:
				s0.append(prediction[i][1])
		print(sum(s1)/len(s1))
		print(sum(s0)/len(s0))
		print(max(s0))
		print(min(s1))
		print(stats.scoreatpercentile(s0, 99))
		print(stats.scoreatpercentile(s0, 5))

		prediction = [ 1 if x[1] >= 0.1 else 0 for x in prediction ]

		if report or True:
			print('Constructing the classification report')
			print(classification_report(classes2, prediction))
		
		print('Calculating F1 score')
		return f1_score(classes2, prediction, pos_label=1)

	# Iterate over weight to find best weight
	def learn_optimal_weight(self, features1, features2, classes2, classes3):
		w_min = 0
		w_max = 1
		w_opt = 0
		s_max = 0

		weights = []
		f1s = []

		for _ in xrange(0,20):
			w_mid = (w_max + w_min)/2
			w_1 = w_min + (w_mid - w_min)/2
			w_2 = w_mid + (w_max - w_mid)/2

			s1 = self.learn_get_f1(w_1, features1, features2, classes2, classes3)
			s2 = self.learn_get_f1(w_2, features1, features2, classes2, classes3)

			if w_1 not in weights:
				weights.append(w_1)
				f1s.append(s1)
			
			if w_2 not in weights:
				weights.append(w_2)
				f1s.append(s2)

			print('')
			print('Weight class 0: ' + str(w_1))
			print('F1 score: ' + str(s1))
			print('Weight class 0: ' + str(w_2))
			print('F1 score: ' + str(s2))

			# If f1 score change is less than 
			if (math.fabs(max(s1, s2) - s_max) < 0.00001) and (math.fabs(s1 - s2) < 0.00001) :
				print('Max found')
				if (s1 > s2) and (s1 > s_max) :
					w_opt = w_1
				elif (s2probab > s_max):
					w_opt = w_2
				break

			if s1 > s2 :
				w_max = w_mid
				if s1 > s_max:
					w_opt = w_1
					s_max = s1
			else:
				w_min = w_mid
				if s2 > s_max:
					w_opt = w_2
					s_max = s2
		return w_opt, {'weights': weights, 'f1':f1s}

	# Iterate over weights
	def iterate_weight(self, features1, features2, classes2, classes3):
		w_min = 0.05
		w_max = 1

		weights = []
		f1s = []

		for i in xrange(0,20):
			s1 = self.learn_get_f1(w_min+i*0.05, features1, features2, classes2, classes3)
			
			weights.append(w_min+i*0.05)
			f1s.append(s1)

			print('')
			print('Weight class 0: ' + str(w_min + 0.05*i))
			print('F1 score: ' + str(s1))

		return {'weights': weights, 'f1':f1s}

	def supervised_learn(self, feature1=None, class1=None, feature2=None, class2=None, algorithm='svm' ):
		if self.cache_check():
			print('Cache found. Reading.')
			features1, classes2, features2, classes3 = self.cache_read()
		else:
			print('Cache not found. Constructing.')
			print('Features 1')
			features1, classes1 = self.get_features(self.t1,self.t2,sample=False, one_class=False, comm_sample=True)
			print('Features 2')
			features2, classes2 = self.get_features(self.t2,self.t3,sample=False)
			print('Features 3')
			features3, classes3 = self.get_features(self.t3,self.t4,sample=False, last=True)

			self.cache_save(features1, classes2, features2, classes3)

		if feature1 is not None:
			features1 = feature1

		if feature2 is not None:
			features2 = feature2

		if class1 is not None:
			classes2 = class1

		if class2 is not None:
			classes3 = class2
		
		#features1, classes2, w0, w1 = self.balance_data(features1, classes2, N=5)
		#features2, classes3, t_0, t_1 = self.balance_data(features2, classes3, N=5)
		
		w0 = 1
		w1 = 1

		#t_f, t_c, w0, w1 = self.balance_data(features1, classes2, N=10)

		if algorithm == 'svm_prob':
			self.learn_get_f1_prob(features1, features2, classes2, classes3)
			return True
		elif algorithm == 'tree_prob':
			self.learn_get_f1_prob(features1, features2, classes2, classes3, algo='tree')
			return True

		w_opt = None
		if algorithm == 'svm_iterate':
			data1 = self.iterate_weight(features1, features2, classes2, classes3)
			w_opt, data2 = self.learn_optimal_weight(features1, features2, classes2, classes3)

			# Plot the f1 vs weights
			plt.scatter(data1['weights'], data1['f1'], c='red')
			plt.scatter(data2['weights'], data2['f1'], c='blue')
			plt.xlabel('Class 0 weights')
			plt.ylabel('F1 Score')
			plt.savefig('img/f1-vs-weight-'+str(int(time.time()))+'.png')
			plt.clf()
		
		self.learn_get_f1(w_opt, features1, features2, classes2, classes3, report=True, algorithm=algorithm, w0=w0, w1=w1)
		

	def anomaly_report(self, prediction, classes):
		TP = 0
		TN = 0
		FP = 0
		FN = 0

		threshold = min(prediction)*0.9

		for i in xrange(0, len(prediction)):
			if (prediction[i] >= threshold) and (classes[i] == 0):
				TN += 1
			elif (prediction[i] < threshold) and (classes[i] == 1):
				TP += 1
			elif (prediction[i] >= threshold) and (classes[i] == 1):
				FN += 1
			elif (prediction[i] < threshold) and (classes[i] == 0):
				FP += 1
		accuracy = (TP+TN)/(TP+TN+FP+FN)
		recall = TP/(TP+FN)
		precision = TP/(TP+FP)
		f1 = (2*recall*precision)/(recall+precision)

		#print('Accuracy: ' + str(accuracy))
		#print('Recall: ' + str(recall))
		#print('Precision: ' + str(precision))
		#print('F1: '+ str(f1))
		print('TPR: '+ str(TP/(TP+FN)))
		print('FPR: '+ str(FP/(FP+TN)))
		print((TP,TN,FP,FN))

		return accuracy, recall, precision, f1

	# Save nodes in the community
	def save_community_nodes(self):
		with open('cache/community_nodes.csv', 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in self.users_sample:
				datawriter.writerow([row])

	# Cache the training and testing data
	def cache_save(self, features1, classes1, features2, classes2):
		with open(self.directory+self.f1, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in features1:
				datawriter.writerow(row)

		with open(self.directory+self.c1, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in classes1:
				datawriter.writerow([row])

		with open(self.directory+self.f2, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in features2:
				datawriter.writerow(row)

		with open(self.directory+self.c2, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in classes2:
				datawriter.writerow([row])

	def cache_read(self):
		features1 = []
		features2 = []
		classes1 = []
		classes2 = []

		with open(self.directory+self.f1, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				features1.append([float(x) for x in row])

		with open(self.directory+self.f2, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				features2.append([float(x) for x in row])

		with open(self.directory+self.c1, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				classes1.append(int(row[0]))

		with open(self.directory+self.c2, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				classes2.append(int(row[0]))

		return features1, classes1, features2, classes2

	# Check if cache files are present
	def cache_check(self):
		return os.path.exists(self.directory+self.f1) and os.path.exists(self.directory+self.f2) and os.path.exists(self.directory+self.c1) and os.path.exists(self.directory+self.c2) 

	# Filter out positive class
	# For anomaly detection
	def filter_classes(self, feature, classes):
		f = []
		c = []

		pos_index = []

		for i in xrange(0, len(classes)):
			if classes[i] == 1:
				pos_index.append(i)
			else:
				c.append(classes[i])

		for i in xrange(0, len(feature)):
			if i not in pos_index:
				f.append(feature[i])

		return f,c

	# Valance the classes in training data
	def balance_data(self, feature, classes, N=1):
		f1 = []
		c1 = []
		f2 = []
		c2 = []
		w1 = 1
		w0 = 1

		pos_index = []

		for i in xrange(0, len(classes)):
			if classes[i] == 1:
				pos_index.append(i)
				c1.append(classes[i])
			else:
				c2.append(classes[i])

		w1 = len(classes)/len(c1)
		w0 = len(classes)/len(c2)

		print((len(classes), len(c1), len(c2)))

		for i in xrange(0, len(feature)):
			if i in pos_index:
				f1.append(feature[i])
			else:
				f2.append(feature[i])

		random.shuffle(f2)

		f = f1*N + f2[:N*len(f1)]
		c = [1]*(len(f1)*N) + [0]*(N*len(f1))

		return f, c, w0, w1

	def get_predicted_outliers(self, feature, classes, predicted):
		f = []
		c = []

		for i in xrange(0, len(predicted)):
			if predicted[i] == -1 :
				f.append(feature[i])
				c.append(classes[i])

		return f, c

	def anomaly_detection(self, feature1=None, class1=None, feature2=None, class2=None):
		if self.cache_check():
			print('Cache found. Reading.')
			features1, classes2, features2, classes3 = self.cache_read()
		else:
			print('Cache not found. Constructing.')
			print('Features 1')
			features1, classes1 = self.get_features(self.t1,self.t2,sample=False, one_class=False, comm_sample=True)
			print('Features 2')
			features2, classes2 = self.get_features(self.t2,self.t3,sample=False)
			print('Features 3')
			features3, classes3 = self.get_features(self.t3,self.t4,sample=False, last=True)

			self.cache_save(features1, classes2, features2, classes3)

		if feature1 is not None:
			features1 = feature1

		if feature2 is not None:
			features2 = feature2

		if class1 is not None:
			classes2 = class1

		if class2 is not None:
			classes3 = class2

		features1, classes2 = self.filter_classes(features1, classes2)

		
		print('Fitting model')
		clf = svm.OneClassSVM(nu=(220/9900), kernel="rbf", gamma=0.1)
		clf.fit(features1)

		print('Predicting')
		prediction = clf.predict(features2)

		print('Constructing the classification report')
		self.anomaly_report(prediction, classes3)

		f, c = self.get_predicted_outliers(features2, classes3, prediction)
		
		return f, c

	def community_data(self):
		graph = self.network.get_graph()
		
		for x in xrange(3,7):
			for i in xrange(1,10):
				self.community_clique(x, n=False)
				size = len(self.users_sample)

				#print(x)
				#print('Community size: '+str(size))


				#Density
				d1 = 0
				weights = []

				for n in self.users_sample:
					#pprint(graph[n])
					w = [graph[n][m]['weight'] for m in graph[n].keys() if m in self.users_sample]
					d1 += len(w)
					#w = [graph[n][m]['weight'] for m in graph[n].keys()]
					weights += w

				density = d1/(size*(size - 1))
				#mean = numpy.mean(weights)
				#count = len(weights)
				#weights = [x for x in weights if x > mean*0.75 and x < mean * 1.25]


				print('Density: '+str(density))
				pprint('Weights Std: '+ str(numpy.std(weights)))
				print('')


	def sample_data(self):
		graph = self.network.get_graph()
		size = len(self.users_sample)

		print('Community size: '+str(size))


		#Density
		d1 = 0
		weights = []

		for n in self.users_sample:
			w = [graph[n][m]['weight'] for m in graph[n].keys() if m in self.users_sample]
			weights += w
			d1 += len(w)

		density = d1/(size*(size - 1))
		variance = numpy.var(weights)

		self.density = str(density)
		self.variance = str(variance)

		print('Density: '+str(density))
		pprint('Weights Variance: '+ str(variance))

	def save_network(self):
		pass


	def run(self, v1, v2=None):
		if v1 == 'l':
			self.supervised_learn(algorithm='svm_prob')
		elif v1 == 'a':
			self.anomaly_detection()
		elif v1 == 'anomaly_loop':
			f = None
			c = None
			for _ in xrange(1,10):
				f, c = self.anomaly_detection(feature2=f, class2=c)
		elif v1 == 'al':
			f, c = self.anomaly_detection()
			self.supervised_learn(feature2=f, class2=c, algorithm='random_forest')
		elif v1 == 'community':
			self.update_network(150,318)
			self.community_clique(5)
		elif v1 == 'community_data':
			self.update_network(150,486)
			#self.update_network(150,654)
			self.community_data();
		elif v1 == 'draw':
			self.update_network(1,24)
			print('Getting the community')
			self.community_clique(5)
			print('Drawing the graph')
			networkx.draw_networkx(self.network.get_graph(),pos=networkx.spring_layout(self.network.get_graph()),nodelist=self.users_sample)
			print('Saving the graph')
			plt.savefig('img/'+'network-'+v2+'.png')
		elif v1 == 'links_count':
			self.update_network(0,720)
			positive = self.network.get_graph().number_of_edges()
			nodes = self.network.get_graph().number_of_nodes()
			negative = (nodes*nodes - nodes) - positive
			print((nodes, positive, negative, positive/negative))
			print(networkx.density(self.network.get_graph()))

if __name__ == '__main__':
	compare_measures()
	#arg1 = sys.argv[1]
	#arg2 = sys.argv[2]
	
	#p = Prediction(N=100, m_type=arg2, t1=150, t2=318, t3=486, t4=654, f1='5', f2='100', f3='30')
	#p = Prediction(N=100, m_type=arg2, t1=0, t2=2, t3=3, t4=4, f1='6', f2='cond-mat', f3='41')
	#p.run(arg1, arg2)
