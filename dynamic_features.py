# Gets the features over time

import database, graph, random, measures, fileio
from pprint import pprint


def main():
	cutoff = 0.1 	# Links below this weight are considered to be broken
	frac = 0.02		# Fraction of the total valid users to sample
	delta_t = 3600	# Time interval in seconds
	beta = 0.9		# Link weight decay factor


	fio = fileio.Fileio()

	db = database.Database('database.db')
	db.open()

	# Get all the valid users
	if not fio.exist_valid_users():
		users = db.valid_users()
		fio.save_valid_users(users)
	else:
		users = fio.read_valid_users()

	# Because of computational limitaions, we will do annalysis on only a fraction of nodes
	if not fio.exist_sample_users():
		users_sample = random.sample(users, int(frac*len(users)))
		fio.save_sample_users(users_sample)
	else:
		users_sample = fio.read_sample_users()

	time_start, time_end = db.get_time_min_max()

	i = 1
	adamic_adar = {}
	jaccard = {}
	common_neighbor = {}

	# Calculates the adamic adar and jaccard coefficient over time
	# If all measures exist, skip over the whole while loop and read everything from the files
	if not all([fio.exist_measure_jaccard(), fio.exist_measure_adamicadar(), fio.exist_measure_commonneighbor()]):
		sn = graph.SocialNetwork(cutoff, users_sample)
		sn.initialize_nodes(users)
		while time_start + i*delta_t < time_end:
			print(i)
			edges = db.get_links(time_start+(i-1)*delta_t, time_start+i*delta_t, None)
			sn.add_edges(edges, beta)
			m = measures.Measures(sn.get_graph(), users_sample)
			
			# Calculate the score only if the CSV do not exist
			if not fio.exist_measure_adamicadar():
				ada = m.adamic_adar()
				print('Adamic-Adar')

				for d in ada:
					if (d[0], d[1]) not in adamic_adar:
						adamic_adar[(d[0],d[1])] = [d[2]]
					else:
						adamic_adar[(d[0],d[1])].append(d[2])
				
			# Calculate the score only if the CSV do not exist
			if not fio.exist_measure_jaccard():
				jac = m.jaccard_coefficient()
				print('Jaccard')
			
				for d in jac:
					if (d[0], d[1]) not in jaccard:
						jaccard[(d[0],d[1])] = [d[2]]
					else:
						jaccard[(d[0],d[1])].append(d[2])

			# Calculate the score only if the CSV do not exist
			if not fio.exist_measure_commonneighbor():
				cne = m.common_neighbor()
				print('Common Neighbor')
			
				for d in cne:
					if (d[0], d[1]) not in common_neighbor:
						common_neighbor[(d[0],d[1])] = [d[2]]
					else:
						common_neighbor[(d[0],d[1])].append(d[2])

			i += 1

		# If CSV exist, read from that
		# Otherwise save the data
		if fio.exist_measure_adamicadar():
			adamic_adar = fio.read_measure_adamicadar()
		else:
			fio.save_measure_adamicadar(adamic_adar)

		if fio.exist_measure_jaccard():
			jaccard = fio.read_measure_jaccard()
		else:
			fio.save_measure_jaccard(jaccard)

		if fio.exist_measure_commonneighbor():
			common_neighbor = fio.read_measure_commonneighbor()
		else:
			fio.save_measure_commonneighbor(common_neighbor)

	else:
		# All the measures exist in CSV files
		# So read from file instead of recalculating again
		jaccard = fio.read_measure_jaccard()
		adamic_adar = fio.read_measure_adamicadar()
		common_neighbor = fio.read_measure_commonneighbor()

	# Extract non zero values
	nz_adamic = {}
	nz_jaccard = {}
	nz_commnei = {}

	for n in adamic_adar:
		if not all(x == 0 for x in adamic_adar[n]):
			nz_adamic[n] = adamic_adar[n]
	for n in jaccard:
		if not all(x == 0 for x in jaccard[n]):
			nz_jaccard[n] = jaccard[n]
	for n in common_neighbor:
		if not all(x == 0 for x in common_neighbor[n]):
			nz_commnei[n] = common_neighbor[n]

	for n in nz_adamic:
		for x in xrange(1,len(nz_adamic[n]) -1):
			if nz_adamic[n][x-1] > 0 and nz_adamic[n][x] > 0:
				print(nz_adamic[n][x]/nz_adamic[n][x-1])

	pprint(len(nz_adamic))

	#for n in nz_jaccard:
	#	for x in xrange(1,len(nz_jaccard[n]) -1):
	#		if nz_jaccard[n][x-1] > 0 and nz_jaccard[n][x] > 0:
	#			print(nz_jaccard[n][x]/nz_jaccard[n][x-1])

	#pprint(len(nz_jaccard))

	#for n in nz_commnei:
	#	for x in xrange(1,len(nz_commnei[n]) -1):
	#		if nz_commnei[n][x-1] > 0 and nz_commnei[n][x] > 0:
	#			print(nz_commnei[n][x]/nz_commnei[n][x-1])

	#pprint(len(nz_commnei))


	db.close()


if __name__ == '__main__':
	main()