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