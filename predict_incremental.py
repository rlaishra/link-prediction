from __future__ import division
import math, operator, random, numpy, networkx, sys, os.path, csv, time, sqlite3
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

class MeasuresWeightAM():
	def __init__(self, network):
		self._network = network

		self._common_neighbors 	= None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferentail_attachment = None
		self._shortest_path = None

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
			self.cprint(str(i+1)+'/'+str(len(nodes)))
			for j in xrange(i, len(nodes)):
				neighbors_out = list(set.intersection(set(graph[nodes[i]].keys()), set(graph[nodes[j]].keys())))
				neighbors_in = list(set.intersection(set(rgraph[nodes[i]].keys()), set(rgraph[nodes[j]].keys())))

				self._common_neighbors[(nodes[i], nodes[j])] = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/2 for x in neighbors_out]) + sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/2 for x in neighbors_in])
				self._common_neighbors[(nodes[j], nodes[i])] = self._common_neighbors[(nodes[i], nodes[j])]
		print('')
		return self._common_neighbors

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

				#adamic_adar_out = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/(2 * math.log(max(1.01, sum([graph[x][z]['weight'] for z in graph[x] ])))) for x in neighbors_out ])
				#adamic_adar_in = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/(2 * math.log(max(1.01, sum([rgraph[x][z]['weight'] for z in rgraph[x] ])))) for x in neighbors_in ])

				adamic_adar_out = sum([ (graph[nodes[i]][x]['weight'] + graph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([graph[x][z]['weight'] for z in graph[x] ]))) for x in neighbors_out ])
				adamic_adar_in = sum([ (rgraph[nodes[i]][x]['weight'] + rgraph[nodes[j]][x]['weight'])/(2 * math.log(1.001 + sum([rgraph[x][z]['weight'] for z in rgraph[x] ]))) for x in neighbors_in ])

				self._adamic_adar[(nodes[i], nodes[j])] = adamic_adar_in + adamic_adar_out
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
				self._preferentail_attachment[(nodes[i], nodes[j])] = sum([ graph[nodes[i]][x]['weight'] for x in graph[nodes[i]] ])*sum([ graph[nodes[j]][x]['weight'] for x in graph[nodes[j]] ]) + sum([ rgraph[nodes[i]][x]['weight'] for x in rgraph[nodes[i]] ])*sum([ rgraph[nodes[j]][x]['weight'] for x in rgraph[nodes[j]] ])
				self._preferentail_attachment[(nodes[j], nodes[i])] = self._preferentail_attachment[(nodes[i], nodes[j])]
		print('')
		return self._preferentail_attachment

	# Extended jaccard coefficient
	def jaccard_coefficient(self, nodes=None):
		self._jaccard_coefficient = {}

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
				
				self._jaccard_coefficient[(nodes[i], nodes[j])] = jac_out + jac_in
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

class NetworkDB():
	def __init__(self):
		self.file_name = 'network_db.db'

	# Open a database connection
	def _open(self):
		self.connection = sqlite3.connect(self.file_name)
		self.cursor = self.connection.cursor()

	# Commit after making changes to the database
	def _commit(self):
		self.connection.commit()

	# Close database connection
	def _close(self):
		self.connection.close()

	# Add neighbors
	# edge from node to nodes in neighbors
	# neighbors is a list
	def add_neighbors(self, node, neighbors):
		self._open()
		for n in neighbors:
			vals = (node, n)
			self.cursor.execute('INSERT INTO `edges` VALUES (?,?)', vals)
		self._commit()

	# Get all the in neighbors for node
	# Returns a list
	def get_in_neighbors(self, node):
		self._open()
		neighbors = []
		for row in self.cursor.execute('SELECT * FROM `edges` WHERE `node_to` = ?', (node,)):
			neighbors.append(row[0])
		self._close()
		return neighbors

	# Get all the in neighbors for node
	# Returns a list
	def get_out_neighbors(self, node):
		self._open()
		neighbors = []
		for row in self.cursor.execute('SELECT * FROM `edges` WHERE `node_from` = ?', (node,)):
			neighbors.append(row[1])
		self._close()
		return neighbors


class MeasuresWeightAMIncremental(object):
	
	def __init__(self, network, nodes, decay_factor=0.9):
		self._network = network
		self._nodes = nodes
		self._decay_factor = decay_factor
		self._default_link_weight = 1

		self._common_neighbors 	= {}
		self._jaccard_coefficient = {}
		self._adamic_adar = {}
		self._preferentail_attachment = {}
		self._shortest_path = {}

		self._db = NetworkDB()

		self._edges = []

		for i in xrange(0, len(self._nodes)):
			for j in xrange(i+1, len(self._nodes)):
				self._edges.append((self._nodes[i], self._nodes[j]))
				self._edges.append((self._nodes[j], self._nodes[i]))

				self._common_neighbors[(self._nodes[i], self._nodes[j])] = 0
				self._common_neighbors[(self._nodes[j], self._nodes[i])] = 0

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	# Call only once per decay time
	def decay(self):
		for edge in self._edges:
			self._common_neighbors[edge] *= self._decay_factor 

	def _update_common_neighbor(self, u, v, link_exist=False):
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		# Get all nodes that has v as an out neighbor
		nodes1 = rgraph[v].keys()
		for n in nodes1:
			if n in self._nodes:
				if link_exist and (n != u):
					self._common_neighbors[(n,v)] += self._default_link_weight/2
				elif not link_exist:
					self._common_neighbors[(n,v)] += (self._default_link_weight + graph[n][v]['weight'])/2
				self._common_neighbors[(v,n)] = self._common_neighbors[(n,v)]
		
		# Get all nodes that has u as in neighbor
		nodes2 = graph[u].keys()
		for n in nodes2:
			if n in self._nodes:
				if link_exist and (n != v):
					self._common_neighbors[(u,n)] += self._default_link_weight/2
				elif not link_exist:
					self._common_neighbors[(u,n)] += (self._default_link_weight + graph[u][n]['weight'])/2
				self._common_neighbors[(n,u)] = self._common_neighbors[(u,n)]

	# New node is added. Update the common neighbor scores
	def update(self, edge):
		u = edge[0]
		v = edge[1]

		if u in self._nodes and v in self._nodes:
			# Check if edge is new or already exist
			graph = self._network.get_graph()
			link_exist = v in graph[u]

			self._update_common_neighbor(u,v, link_exist)

# The main class
class Main(object):
	def __init__(self):
		self.vals = data.Data()
		self.db = database.Database()

		# Users
		self.users_valid = self.vals.get_valid_users()
		self.users_sample = self._get_nodes()

		self.db.open()
		#self.time_start, self.time_end = self.db.get_time_min_max()
		self.time_start = 1422289905
		self.delta_time = 3600

		self.network = graph.SocialNetwork(0.1, self.users_valid, weighted=True)
		self.network.initialize_nodes(self.users_valid)

		self.measures = MeasuresWeightAMIncremental(self.network, self.users_sample)

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	def _get_nodes(self):
		nodes = []
		with open('cache/community_nodes.csv', 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				nodes.append(row[0])
		return nodes

	# Convert edges in adjacency list to list of tuples
	def convert_adjacency_to_list(self, edges):
		ls = []
		for n1 in edges:
			for n2 in edges[n1]:
				ls.append((n1,n2))
		return ls

	def run(self):
		for i in xrange(1, 654):
			self.cprint(str(int((i+1)*100/654))+'%'),
			edges = self.db.get_links(self.time_start+(i-1)*self.delta_time, self.time_start+i*self.delta_time, self.users_valid, True)
			#print(edges)
			self.measures.decay()
			for e in self.convert_adjacency_to_list(edges):
				self.measures.update(e)
			self.network.add_edges(edges, 0.9)


def main():
	m = Main()
	m.run()

if __name__ == '__main__':
	main()