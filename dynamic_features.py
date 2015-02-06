# Gets the features over time

import database, graph, random, measures, fileio
from pprint import pprint


def main():
	fio = fileio.Fileio()

	db = database.Database('database.db')
	db.open()

	# Get all the valid users
	if not fio.exist_valid_users():
		users = db.valid_users()
		fio.save_valid_users(users)
	else:
		users = fio.read_valid_users()

	# Because of computational limitaions, we will do annalysis on only 10% sample nodes
	users_sample = random.sample(users, int(0.01*len(users)))
	
	sn = graph.SocialNetwork(0.1, users_sample)
	sn.initialize_nodes(users)

	time_start, time_end = db.get_time_min_max()

	i = 1
	delta_t = 3600
	adamic_adar = {}
	jaccard = {}

	# Calculates the adamic adar and jaccard coefficient over time
	while time_start + i*delta_t < time_end:
		print(i)
		edges = db.get_links(time_start+(i-1)*delta_t, time_start+i*delta_t, None)
		sn.add_edges(edges, 0.9)
		m = measures.Measures(sn.get_graph(), users_sample)
		ada = m.adamic_adar()
		print('Adamic-Adar')
		jac = m.jaccard_coefficient()
		print('Jaccard')
		
		for d in ada:
			if (d[0], d[1]) not in adamic_adar:
				adamic_adar[(d[0],d[1])] = [d[2]]
			else:
				adamic_adar[(d[0],d[1])].append(d[2])

		for d in jac:
			if (d[0], d[1]) not in jaccard:
				jaccard[(d[0],d[1])] = [d[2]]
			else:
				jaccard[(d[0],d[1])].append(d[2])
		i += 1
		if i > 10:
			break

	# Extract non zero values
	nz_adamic = {}
	nz_jaccard = {}

	for n in adamic_adar:
		if not all(x == 0 for x in adamic_adar[n]):
			nz_adamic[n] = adamic_adar[n]
	for n in jaccard:
		if not all(x == 0 for x in jaccard[n]):
			nz_jaccard[n] = jaccard[n]

	pprint(nz_adamic)
	pprint(nz_jaccard)


	db.close()


if __name__ == '__main__':
	main()