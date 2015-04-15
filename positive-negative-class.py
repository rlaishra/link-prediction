from __future__ import division
import csv, sys
import networkx as nx

DATA = 'db/'

# Create a graph with edges created betweeen time t_start and t_end
def createGraph(t_start, t_end, filename, graph):
	nodes = []
	edges = []
	with open(DATA + filename, 'rb') as csvfile:
		datereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in datereader:
			if (len(row) > 3) and (float(row[1]) <= t_end) and (float(row[1]) > t_start):
				for x in xrange(3,len(row)):
					if row[x] is not '':
						for y in xrange(x+1,len(row)):
							if row[y] is not '':
								graph.add_edge(row[x], row[y], weight=1)
								if row[x] not in nodes:
									nodes.append(row[x])
								if row[y] not in nodes:
									nodes.append(row[x])
								edges.append((row[x], row[y]))
								edges.append((row[y], row[x]))
	return graph

if __name__ == '__main__':
	t1 = 788918400 # 1995-01-01
	t2 = 916290000 # end of 1999
	t3 = 946530000 # 2000

	t1995 = 788918400
	t1998 = 883612800
	t1999 = 915148800
	t2000 = 946684800

	graph = nx.Graph()
	graph = createGraph(t1, t3, sys.argv[1], graph)
	nodes = graph.number_of_nodes()
	positive =  graph.number_of_edges()
	negative = (nodes*nodes - nodes) - positive
	print((nodes, positive, negative, positive/negative))
	print(nx.density(graph))