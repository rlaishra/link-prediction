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
	
	# Calculate features 
	# Features list is a list of features to calculate
	# a -> admaic_adar
	# p -> preferential attachment
	# j -> jaccard
	# c -> common neighbors
	def generate(self, features_list=None):
		# if feature_list is not None, check to make sure that the vars are valid
		allowed_measures = ['a', 'j', 'c', 'p']
		if not all([x in allowed_measures for x in features_list]):
			print('Invalid measure')
			return False

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
			# If distances are not in cache calculate
			weighted = False
			sn = graph.SocialNetwork(self._config.density_cutoff, users_sample, weighted)
			sn.initialize_nodes(users_valid)

			is_jaccard = (not self._cache.exist_measure_jaccard()) and (features_list is None or 'j' in features_list)
			is_adamic = (not self._cache.exist_measure_adamicadar()) and (features_list is None or 'a' in features_list)
			is_cnome = (not self._cache.exist_measure_commonneighbor()) and (features_list is None or 'c' in features_list)
			is_prefa = (not self._cache.exist_measure_preferentialattachment()) and (features_list is None or 'p' in features_list)

			while time_start + i*self._config_data.delta_t < time_end:
				if self._run.verbose:
					print(str(i) + ' Calculating features between '+str(time_start+(i-1)*self._config_data.delta_t)+' and '+str(time_start+i*self._config_data.delta_t))

				# Get edges
				edges = db.get_links(time_start+(i-1)*self._config_data.delta_t, time_start+i*self._config_data.delta_t, users_valid, True)
				if self._run.verbose:
					print(str(len(edges)) + ' edges found')

				# Add edges
				sn.add_edges(edges, self._config.decay_factor)

				# Initilize measures class
				m = measures.Measures(sn, users_sample)

				# Get the measure values only if they dont exist in file
				if self._run.verbose:
					print('Generating features')

				jac, ada, cne, pref = m.combined( is_jaccard , is_adamic, is_cnome, is_prefa)
				
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

				if i >= 48:
					break

			# If cache exist, read from that
			# Otherwise save the data
			if self._cache.exist_measure_adamicadar() and 'a' in features_list:
				if self._run.verbose:
					print('Reading Adamic Adar from cache')
				adamic_adar = self._cache.read_measure_adamicadar()
			elif len(adamic_adar) > 0:
				if self._run.verbose:
					print('Saving Adamic Adr to cache')
				self._cache.save_measure_adamicadar(adamic_adar)

			if self._cache.exist_measure_jaccard() and 'j' in features_list:
				if self._run.verbose:
					print('Reading Jaccard from cache')
				jaccard = self._cache.read_measure_jaccard()
			elif len(jaccard) > 0:
				if self._run.verbose:
					print('Saving Jaccard to cache')
				self._cache.save_measure_jaccard(jaccard)

			if self._cache.exist_measure_commonneighbor() and 'c' in features_list:
				if self._run.verbose:
					print('Reading Common neighbors from cache')
				common_neighbor = self._cache.read_measure_commonneighbor()
			elif len(common_neighbor) > 0:
				if self._run.verbose:
					print('Saving cache to cache')
				self._cache.save_measure_commonneighbor(common_neighbor)

			if self._cache.exist_measure_preferentialattachment() and 'p' in features_list:
				if self._run.verbose:
					print('Reading Preferential attachment from cache')
				preferential_attchment = self._cache.read_measure_preferentialattachment()
			elif len(preferential_attchment) > 0:
				if self._run.verbose:
					print('Saving preferential attachment to cache')
				self._cache.save_measure_preferentialattachment(preferential_attchment)

		db.close()


	# Read the dynamic features are return them
	# Ca only read one value at a time
	def read(self, feature):
		if feature not in ['a', 'j', 'c', 'p']:
			print('Invalid measure')
			return False

		if feature is 'j':
			if self._run.verbose:
				print('Reading Jaccard from cache')
			jaccard = self._cache.read_measure_jaccard()
			return jaccard

		if feature is 'a':
			if self._run.verbose:
				print('Reading Adamic Adar from cache')
			adamic_adar = self._cache.read_measure_adamicadar()
			return adamic_adar

		if feature is 'c':
			if self._run.verbose:
				print('Reading Common neighbors from cache')
			common_neighbor = self._cache.read_measure_commonneighbor()
			return common_neighbor

		if feature is 'p':
			if self._run.verbose:
				print('Reading preserential attachment from cache')
			preferential_attchment = self._cache.read_measure_preferentialattchment()
			return preferential_attchment