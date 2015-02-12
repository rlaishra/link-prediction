# Gets the features over time

import database, graph, random, measures, cache, preprocess
from pprint import pprint


def main():
	cutoff = 0.1 	# Links below this weight are considered to be broken
	sample_size = 2500		# Fraction of the total valid users to sample
	delta_t = 2*3600	# Time interval in seconds
	beta = 0.9		# Link weight decay factor
  
	fio = cache.Cache()

	db = database.Database()
	db.open()

	# Get all the valid users
	if not fio.exist_valid_users():
		users = db.valid_users()
		fio.save_valid_users(users)
	else:
		users = fio.read_valid_users()

	# Because of computational limitaions, we will do annalysis on only a fraction of nodes
	if not fio.exist_sample_users():
		users_sample = random.sample(users, sample_size)
		fio.save_sample_users(users_sample)
	else:
		users_sample = fio.read_sample_users()

	time_start, time_end = db.get_time_min_max()
	prep = preprocess.Preprocess()
	
	# Check if there are outliers in the selected sample
	outliers = prep.outlier_nodes(db.get_links(time_start, time_end, users_sample), users_sample, 10, 0.1, True)
	pprint(len(outliers))
	pprint(len(users_sample))
	# Remove the outliers from the users_sample
	for n in outliers:
		users_sample.remove(n)
	pprint(len(users_sample))

	i = 1
	adamic_adar = {}
	jaccard = {}
	common_neighbor = {}
	preferential_attchment = {}

	# Calculates the adamic adar and jaccard coefficient over time
	# If all measures exist, skip over the whole while loop and read everything from the files
	if not all([fio.exist_measure_jaccard(), fio.exist_measure_adamicadar(), fio.exist_measure_commonneighbor(), fio.exist_measure_preferentialattachment()]):
		sn = graph.SocialNetwork(cutoff, users_sample)
		sn.initialize_nodes(users)

		while time_start + i*delta_t < time_end:
			print(i)
			edges = db.get_links(time_start+(i-1)*delta_t, time_start+i*delta_t, users_sample)
			sn.add_edges(edges, beta)
			m = measures.Measures(sn.get_graph(), users_sample)
			
			# Get the measure values only if they dont exist in file
			jac, ada, cne, pref = m.combined(not fio.exist_measure_jaccard(), not fio.exist_measure_adamicadar(), not fio.exist_measure_commonneighbor(), not fio.exist_measure_preferentialattachment())

			# Append measures to time series lists
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
			
			for d in cne:
				if (d[0], d[1]) not in common_neighbor:
					common_neighbor[(d[0],d[1])] = [d[2]]
				else:
					common_neighbor[(d[0],d[1])].append(d[2])

			for d in pref:
				if (d[0], d[1]) not in preferential_attchment:
					preferential_attchment[(d[0],d[1])] = [d[2]]
				else:
					preferential_attchment[(d[0],d[1])].append(d[2])

			i += 1
			if i > 5:
				break

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

		if fio.exist_measure_preferentialattachment():
			preferential_attchment = fio.read_measure_preferentialattachment()
		else:
			fio.save_measure_preferentialattachment(preferential_attchment)

	else:
		# All the measures exist in CSV files
		# So read from file instead of recalculating again
		jaccard = fio.read_measure_jaccard()
		adamic_adar = fio.read_measure_adamicadar()
		common_neighbor = fio.read_measure_commonneighbor()
		preferential_attchment = fio.read_measure_preferentialattchment()

	db.close()


if __name__ == '__main__':
	main()