from __future__ import division
from pprint import pprint
from scipy import linalg
import numpy, time, math, copy, random
import networkx as nx
import data, database, graph

# Compare the new page rank algorithm with the unmodified version

class PageRankTest():
	def __init__(self):
		self._db = database.Database()
		self._data = data.Data()

		self._db.open()
		time_start, time_end = self._db.get_time_min_max()
		pprint('Time Start: '+ str(time_start))
		pprint('Time End: '+ str(time_end))

		self._nodes_sample = self._data.get_sample_users()[:1000]
		self._nodes = self._data.get_valid_users()
		self._edges = self._db.get_links(time_start, time_end, self._nodes, True)

		self._sn = graph.SocialNetwork(0.1, self._nodes_sample, True)
		self._sn.initialize_nodes(self._nodes)
		self._sn.add_edges(self._edges, 0.9)

		#The preset constants
		self._alpha = 0.3		# Teleportation probability
		self._f = 0.9			# Fraction of paths to consider
		self._prob_cutoff = 0.01 # The minimum probability above which path length will not be calculated

	# The unmodified rooted page rank
	# returns the page rank and time taken milli seconds
	def page_rank_unmodified(self):
		# Keep track of execution time
		time_start = time.time()
		
		# Construct the walk matrix
		walk_matrix = {}

		# Start for each node as root
		pr = {}
		for root in self._nodes_sample:
			pr[root] = {}
			w_matrix = {}
			all_nodes = []
			
			# Do random walk (1000 runs)
			current_node = root
			current_path_nodes = [current_node]
			for _ in xrange(0,2000):
				if current_node not in all_nodes:
					all_nodes.append(current_node)

				if random.random() > self._alpha:
					neighbors = self._sn.get_graph()[current_node].keys()
					if len(neighbors) > 0 :
						# Select a neighbor at random to walk to
						weight_sum = sum([self._sn.get_graph()[current_node][x]['weight'] for x in neighbors ])
						weighted_list = {}
						last_sum = 0
						for n in neighbors:
							if n not in current_path_nodes:
								weighted_list[n] = {'min': last_sum, 'max': last_sum+self._sn.get_graph()[current_node][n]['weight']/weight_sum}
								last_sum += self._sn.get_graph()[current_node][n]['weight']/weight_sum
						
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

			pr[root] = row
		
		return pr, int(round((time.time() - time_start)*1000))


	def page_rank_modified(self):
		time_start = time.time()
		lc = int(math.ceil(math.log(1 - self._f)/math.log(1 - self._alpha)))	# The longest path used in modified 
		
		pr = {}
		graph = self._sn.get_graph()

		for root in self._nodes_sample:
			paths = [[[root]]]
			all_nodes = [root]
			probabilities = {}

			for _ in xrange(1,lc):
				p = []
				for m in paths[-1]:
					if (tuple(m) in probabilities) and (probabilities[tuple(m)]*(1 - self._alpha) <= self._prob_cutoff) :
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
						probabilities[tuple(p2)] = probabilities[tuple(p2[:-1])]*(1 - self._alpha)*graph[n1][n2]['weight']/(sum([graph[n1][x]['weight'] for x in graph[n1]]))
					else:
						probabilities[tuple(p2)] = 1
						for i in xrange(0, len(p2)-1):
							probabilities[tuple(p2)] = probabilities[tuple(p2)]*(1 - self._alpha)*graph[p2[i]][p2[i+1]]['weight']/(sum([graph[p2[i]][x]['weight'] for x in graph[p2[i]]]))
					
					# The path ends with a teleprotation back to root
					# That teleportation probability
					if tuple(p2 + [root]) not in probabilities.keys():
						probabilities[tuple(p2 + [root])] = probabilities[tuple(p2)]*(1 - self._alpha)
					else:
						probabilities[tuple(p2 + [root])] = probabilities[tuple(p2)]*(1 - self._alpha)
			
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

			pr[root] = row
		return pr, int(round((time.time() - time_start) * 1000))


if __name__ == '__main__':
	test = PageRankTest()
	o_pr, o_time = test.page_rank_unmodified()
	pprint('Original Execution time: ' + str(o_time))
	m_pr, m_time = test.page_rank_modified()
	pprint('Modified Execution time: ' + str(m_time))
	
	error = 0
	total = 0

	for n1 in m_pr:
		if n1 in o_pr:
			for n2 in m_pr[n1]:
				if n2 in o_pr[n1] and o_pr[n1][n2] > 0:
					error += (o_pr[n1][n2] - m_pr[n1][n2])
					total +=(o_pr[n1][n2])
	pprint('Percentage error: ' + str(math.fabs(error*100/total)))