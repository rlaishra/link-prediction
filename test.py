import database, graph, random, measures
from pprint import pprint


def main():

	db = database.Database('database.db')
	db.open()

	# Get all the valid users
	users = db.valid_users()
	
	# Because of computational limitaions, we will do annalysis on only 10% sample nodes
	users_sample = random.sample(users, int(0.01*len(users)))
	
	sn = graph.SocialNetwork(0.1, users_sample)
	sn.initialize_nodes(users)

	time_start, time_end = db.get_time_min_max()

	i = 1
	delta_t = 3600
	adamic_adar = {}
	jaccard = {}
	while time_start + i*delta_t < time_end:
		print(i)
		edges = db.get_links(time_start+(i-1)*delta_t, time_start+i*delta_t, None)
		sn.add_edges(edges, 0.9)
		m = measures.Measures(sn.get_graph(), users_sample)
		ada = m.adamic_adar()
		jac = m.jaccard_coefficient()
		
		for d in ada:
			if (d[0], d[1]) not in adamic_adar:
				adamic_adar[(d[0],d[1])] = [d[2]]
			else:
				adamic_adar[(d[0],d[1])].append(d[2])
		i += 1


	db.close()


if __name__ == '__main__':
	main()