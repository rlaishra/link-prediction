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


class MeasuresWeightAMIncremental():
	
	def __init__(self, network, nodes, decay_factor=0.9):
		self._network = network
		self._nodes = nodes
		self._decay_factor = decay_factor
		self._default_link_weight = 1

		self.common_neighbors 	= {}
		self.jaccard_coefficient = {}
		self.adamic_adar = {}
		self.preferentail_attachment = {}
		self.shortest_path = {}
		self.katz_score = {}
		self.rooted_page_rank = {}

		self._db = NetworkDB()

		self._edges = []

		for i in xrange(0, len(self._nodes)):
			for j in xrange(i+1, len(self._nodes)):
				self._edges.append((self._nodes[i], self._nodes[j]))
				self._edges.append((self._nodes[j], self._nodes[i]))

				self.common_neighbors[(self._nodes[i], self._nodes[j])] = 0
				self.common_neighbors[(self._nodes[j], self._nodes[i])] = 0

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	# Call only once per decay time
	def decay(self):
		for edge in self._edges:
			self.common_neighbors[edge] *= self._decay_factor 

	def _update_common_neighbor(self, u, v, link_exist=False):
		graph = self._network.get_graph()
		rgraph = self._network.get_reverse_graph()

		# Get all nodes that has v as an out neighbor
		nodes1 = rgraph[v].keys()
		for n in nodes1:
			if n in self._nodes:
				if link_exist and (n != u):
					self.common_neighbors[(n,v)] += self._default_link_weight/2
				elif not link_exist:
					self.common_neighbors[(n,v)] += (self._default_link_weight + graph[n][v]['weight'])/2
				self.common_neighbors[(v,n)] = self.common_neighbors[(n,v)]
		
		# Get all nodes that has u as in neighbor
		nodes2 = graph[u].keys()
		for n in nodes2:
			if n in self._nodes:
				if link_exist and (n != v):
					self.common_neighbors[(u,n)] += self._default_link_weight/2
				elif not link_exist:
					self.common_neighbors[(u,n)] += (self._default_link_weight + graph[u][n]['weight'])/2
				self.common_neighbors[(n,u)] = self.common_neighbors[(u,n)]

	# New node is added. Update the common neighbor scores
	def update(self, edge):
		u = edge[0]
		v = edge[1]

		if u in self._nodes and v in self._nodes:zzzzzzzzzz
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
		self.prediction_time = 72 	# Time in future to predict

		self.network = graph.SocialNetwork(0.1, self.users_valid, weighted=True)
		self.network.initialize_nodes(self.users_valid)

		self.measures = MeasuresWeightAMIncremental(self.network, self.users_sample)
		
		self.dbconn = sqlite3.connect('db/features.db')
		self.dbcursor = self.dbconn.cursor()


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

	def save_features_class(self, features, mtype, i, edges=None):
		for edge in features:
			if features[edge] > 0 :
				val = (edge[0], edge[1], i, mtype, features[edge])
				self.dbcursor.execute('INSERT INTO `features` (`from_node`, `to_node`, `time`, `type`, `value`) VALUES (?, ?, ?, ?, ?) ', val)
			
			if edges is not None and edge in edges:
				val = (edge[0], edge[1], i, 1)
				self.dbcursor.execute('INSERT INTO `classes` (`from_node`, `to_node`, `time`, `value`) VALUES (?, ?, ?, ?) ', val)
		self.dbconn.commit()

	def run(self):
		time_min = 1
		time_max = 880
		for i in xrange(1, time_max - self.prediction_time):
			self.cprint(str(int((i+1)*100/802))+'%'),
			edges = self.db.get_links(self.time_start+(i-1)*self.delta_time, self.time_start+i*self.delta_time, self.users_valid, True)
			self.measures.decay()
			edge_list = self.convert_adjacency_to_list(edges)
			for e in edge_list:
				self.measures.update(e)

			#save measure
			self.save_features_class(self.measures.common_neighbors, 'common_neighbors', i, edge_list)

			self.network.add_edges(edges, 0.9)


def main():
	m = Main()
	m.run()

if __name__ == '__main__':
	main()