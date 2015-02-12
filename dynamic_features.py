# Gets the features over time
# Adamic-Adar, Jaccard, Preferential Attachment, Common Neighbors

import random
import database, graph, measures, cache, preprocess, config, data
from pprint import pprint

class DynamicFeatures():
	def __init__(self, run_config):
		self._run = run_config
		self._config = config.Graph()
		self._data = data.Data(run_config)
		self._db = database.Database()
		self._cache = cache.Cache()
		self._config_data = config.Data()
	
	def generate(self):
		# Generate users
		users_valid = self._data.get_valid_users()
		users_sample = self._data.get_sample_users()

		db = database.Database()
		db.open()
		time_start, time_end = db.get_time_min_max()

		i = 1
		adamic_adar = {}
		jaccard = {}
		common_neighbor = {}
		preferential_attchment = {}

		if not all([self._cache.exist_measure_jaccard(), self._cache.exist_measure_adamicadar(), self._cache.exist_measure_commonneighbor(), self._cache.exist_measure_preferentialattachment()]):
			# If diasances are not in cache calculate
			sn = graph.SocialNetwork(self._config.density_cutoff, users_sample)
			sn.initialize_nodes(users_valid)

			while time_start + i*self._config_data.delta_t < time_end:
				if self._run.verbose:
					print(str(i) + ' Calculating features between '+str(time_start+(i-1)*self._config_data.delta_t)+' and '+str(time_start+i*self._config_data.delta_t))

				# Get edges
				edges = db.get_links(time_start+(i-1)*self._config_data.delta_t, time_start+i*self._config_data.delta_t, users_sample)
				if self._run.verbose:
					print(str(len(edges)) + ' edges found')

				# Add edges
				sn.add_edges(edges, self._config.decay_factor)

				# Initilize measures class
				m = measures.Measures(sn.get_graph(), users_sample)

				# Get the measure values only if they dont exist in file
				if self._run.verbose:
					print('Generating features')
				jac, ada, cne, pref = m.combined(not self._cache.exist_measure_jaccard(), not self._cache.exist_measure_adamicadar(), not self._cache.exist_measure_commonneighbor(), not self._cache.exist_measure_preferentialattachment())
				if self._run.verbose:
					print('Features generated')

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

			# If cache exist, read from that
			# Otherwise save the data
			if self._cache.exist_measure_adamicadar():
				if self._run.verbose:
					print('Reading Adamic Adar from cache')
				adamic_adar = self._cache.read_measure_adamicadar()
			else:
				if self._run.verbose:
					print('Saving Adamic Adr to cache')
				self._cache.save_measure_adamicadar(adamic_adar)

			if self._cache.exist_measure_jaccard():
				if self._run.verbose:
					print('Reading Jaccard from cache')
				jaccard = self._cache.read_measure_jaccard()
			else:
				if self._run.verbose:
					print('Saving Jaccard to cache')
				self._cache.save_measure_jaccard(jaccard)

			if self._cache.exist_measure_commonneighbor():
				if self._run.verbose:
					print('Reading Common neighbors from cache')
				common_neighbor = self._cache.read_measure_commonneighbor()
			else:
				if self._run.verbose:
					print('Saving cache to cache')
				self._cache.save_measure_commonneighbor(common_neighbor)

			if self._cache.exist_measure_preferentialattachment():
				if self._run.verbose:
					print('Reading Preferential attachment from cache')
				preferential_attchment = self._cache.read_measure_preferentialattachment()
			else:
				if self._run.verbose:
					print('Saving preferential attachment to cache')
				self._cache.save_measure_preferentialattachment(preferential_attchment)
		else:
			# If diasances are in cache, read from cache
			if self._run.verbose:
				print('Cache found reading from cache')
			#All the measures exist in CSV files
			# So read from file instead of recalculating again
			if self._run.verbose:
				print('Reading Jaccard from cache')
			jaccard = self._cache.read_measure_jaccard()
			if self._run.verbose:
				print('Reading Adamic Adar from cache')
			adamic_adar = self._cache.read_measure_adamicadar()
			if self._run.verbose:
				print('Reading Common neighbors from cache')
			common_neighbor = self._cache.read_measure_commonneighbor()
			if self._run.verbose:
				print('Reading preserential attachment from cache')
			preferential_attchment = self._cache.read_measure_preferentialattchment()

		db.close()