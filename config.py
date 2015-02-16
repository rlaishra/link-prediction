# The config class
# Make changes to config variables here

# Config to set during run timr
class Run():
	def __init__(self):
		# Option to print stuff while running
		self._verbose = False
		
	@property
	def verbose(self):
		return self._verbose

	@verbose.setter
	def verbose(self, value):
		self._verbose = value
	


# Config for misc stuff
class Data():
	def __init__(self):
		self._data = Graph()

		# The sample size of users to get from valid users
		self._sample_size = 2500

		# The time interval to use in seconds
		self._delta_t = 3600

		# The number of neighbors to use during density calculation in proprocess
		self._density_neighbors = 5

		# The minimum density below which node is considered an outlier
		self._density_cutoff = self._data.density_cutoff

	@property
	def sample_size(self):
		return self._sample_size

	@property
	def delta_t(self):
		return self._delta_t

	@property 
	def density_neighbors(self):
		return self._density_neighbors

	@property 
	def density_cutoff(self):
		return self._density_cutoff


# Config for graph
class Graph():
	def __init__(self):
		
		# Lower limit of a link weight
		# Link is considered broken 
		self._min_weight = 0.1
		
		# The link decay factor
		# Between 0 and 1
		self._decay_factor = 1

		# The minimum density below which node is considered an outlier
		self._density_cutoff = 0.1

	@property
	def min_weight(self):
		return self._min_weight

	@property 
	def decay_factor(self):
		return self._decay_factor

	@property 
	def density_cutoff(self):
		return self._density_cutoff

# Database config
class Database():
	def __init__(self):
		self._name = 'database.db'

	@property 
	def name(self):
		return self._name 

# Config for cache file names
class Cache():
	def __init__(self):
		# Cache on or off
		# If false, nothing will be saved in files
		# If true, appropriate measures will be caches in csv files
		self._is_enabled = True

		# File names for cache
		self._cache_dir = 'cache'
		self._valid_users = 'valid_users.csv'
		self._sample_users = 'sample_users.csv'
		self._measure_jaccard = 'measure_jaccard.csv'
		self._measure_adamicadar = 'measure_adamicadar.csv'
		self._measure_commonneighbor = 'measure_commonneighbor.csv'
		self._measure_preferentialattachment = 'measure_preferentialattachment.csv'
		self._preprocess_distance_matrix = 'preporcess_distance_matrix.csv'

	@property 
	def is_enabled(self):
		return self._is_enabled

	@property
	def valid_users(self):
		return self._cache_dir + '/' + self._valid_users

	@property 
	def sample_users(self):
		return self._cache_dir + '/' + self._sample_users

	@property 
	def measure_jaccard(self):
		return self._cache_dir + '/' + self._measure_jaccard

	@property 
	def measure_adamic_adar(self):
		return self._cache_dir + '/' + self._measure_adamicadar

	@property 
	def measure_preferential_attachment(self):
		return self._cache_dir + '/' + self._measure_preferentialattachment

	@property 
	def measure_common_neighbor(self):
		return self._cache_dir + '/' + self._measure_commonneighbor

	@property 
	def preprocess_distance_matrix(self):
		return self._cache_dir + '/' + self._preprocess_distance_matrix