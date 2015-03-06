# Provides an interface between database.py and rest of the stuffs
# Handles all data related stuff like preprocessing etc.

import database, preprocess, config, cache
import random
from pprint import pprint

class Data():
	def __init__(self, run_config=None):
		self._db = database.Database()
		self._cache = cache.Cache()
		self._config = config.Data()
		self._run = run_config

	# Get a list of valid users
	# Users who have atleast one in degree and one out degree
	def get_valid_users(self):
		if self._cache.exist_valid_users():
			if self._run is not None and self._run.verbose:
				print('Valid users cache found')
			# Cache file exist; read from cache and return
			valid_users = self._cache.read_valid_users()
			return valid_users
		else:
			if self._run is not None and self._run.verbose:
				print('Valid users cache not found. Generating from the database.')
			# No cache; read from database and save
			self._db.open()
			valid_users = self._db.valid_users()
			self._db.close()

			if self._run is not None and self._run.verbose:
				print('Valid users generated. Saving in cache.')
			self._cache.save_valid_users(valid_users)
			return valid_users

	# Get a sample from the valid users
	# This is necessary in cases where the number of valid users is too large
	# Optional parameter - the valid users
	def get_sample_users(self, valid_users=None):
		if self._cache.exist_sample_users():
			if self._run is not None and self._run.verbose:
				print('Sample users cache found')
			# Check if sample users is in cache
			# If sampel uses exist in cache read and return that
			sample_users = self._cache.read_sample_users()
			return sample_users
		else:
			if self._run is not None and self._run.verbose:
				print('Sample users cache not found. Generating.')
			# If not in cache generate sample users and save
			if valid_users is None:
				valid_users = self.get_valid_users()
			sample_users = random.sample(valid_users, self._config.sample_size)
			
			if self._run is not None and self._run.verbose:
				print('Sample users generated. Checking for outliers.')

			# Check for outliers in user_samples
			self._db.open()
			time_start, time_end = self._db.get_time_min_max()
			prep = preprocess.Preprocess()
	
			# Check if there are outliers in the selected sample
			links = self._db.get_links(time_start, time_end, sample_users)
			outliers = prep.outlier_nodes(links, sample_users, self._config.density_neighbors, self._config.density_cutoff, True)
			if self._run is not None and self._run.verbose:
				print(str(len(outliers)) + ' outliers found')
			
			# Remove the outliers from the users_sample
			for n in outliers:
				sample_users.remove(n)

			if self._run is not None and self._run.verbose:
				print('Outliers removed. Saving sample users in cache.')

			self._cache.save_sample_users(sample_users)
			self._db.close()

			return sample_users



