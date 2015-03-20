# Predict by incrementally updating the features
# All features here are weighted and directed

class FeatureIncrement(object):
	"""docstring for FeatureIncrement"""
	def __init__(self, network, edges, decay_factor=0.9):
		self._network = network
		self._decay_factor = decay_factor
		self._edges = edges

		self._common_neighbor = None
		self._jaccard_coefficient = None
		self._adamic_adar = None
		self._preferential_atachment = None
		self._shortest_path = None
		self._rooted_page_rank = None

	def cprint(self, v):
		sys.stdout.write("\r%s" % v)
		sys.stdout.flush()

	# Decay all the features
	# Call this only once per unit time
	def decay(self):
		for i in xrange(0, len(self._edges)):
			self._common_neighbor[i][1] *= self._decay_factor
			self._jaccard_coefficient[i][1] *= self._decay_factor
			self._adamic_adar[i][1] *= self._decay_factor
			self._preferential_atachment[i][1] *= self._decay_factor
			self._shortest_path[i][1] += self._decay_factor
			self._rooted_page_rank[i][1] *= self._decay_factor

	