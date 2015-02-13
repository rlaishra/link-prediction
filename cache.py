import csv, os.path, config

# This class is used to save and read stuffs into file
# Always check with exist_ function before reading


class Cache():
	def __init__(self):
		self._config = config.Cache()

		self.file_name_valid_users = self._config.valid_users
		self.file_name_sample_users = self._config.sample_users
		self.file_name_measure_jaccard = self._config.measure_jaccard
		self.file_name_measure_adamicadar = self._config.measure_adamic_adar
		self.file_name_measure_commonneighbor = self._config.measure_common_neighbor
		self.file_name_measure_preferentialattachment = self._config.measure_preferential_attachment
		self.file_name_preprocess_distance_matrix = self._config.preprocess_distance_matrix

	# Check if the valid users CSV file exist
	def exist_valid_users(self):
		return os.path.exists(self.file_name_valid_users) and self._config.is_enabled

	# Save valid users into file
	def save_valid_users(self, data):
		if not self._config.is_enabled:
			return False

		with open(self.file_name_valid_users, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row])

	# REad valid users from file
	def read_valid_users(self):
		data = []
		with open(self.file_name_valid_users, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data.append(row[0])
		return data

	# Check if the valid users CSV file exist
	def exist_sample_users(self):
		return os.path.exists(self.file_name_sample_users) and self._config.is_enabled

	# Save valid users into file
	def save_sample_users(self, data):
		if not self._config.is_enabled:
			return False

		with open(self.file_name_sample_users, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row])

	# REad valid users from file
	def read_sample_users(self):
		data = []
		with open(self.file_name_sample_users, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data.append(row[0])
		return data

	# Check if jaccard measures CSV file exist
	def exist_measure_jaccard(self):
		return os.path.exists(self.file_name_measure_jaccard) and self._config.is_enabled

	# Save the Jaccard measures data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_jaccard(self, data):
		if not self._config.is_enabled:
			return False

		# Save the data in a file
		with open(self.file_name_measure_jaccard, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row[0], row[1]] + data[row])

	# Read jaccard data from CSV file
	# Retuns a dict with nodes tuple as key and measures as values
	def read_measure_jaccard(self):
		data = {}
		with open(self.file_name_measure_jaccard, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data[(row[0], row[1])] = [float(x) for x in row[2:]]
		return data

	# Check if adamic adar measures CSV file exist
	def exist_measure_adamicadar(self):
		return os.path.exists(self.file_name_measure_adamicadar) and self._config.is_enabled

	# Save the adamic adar data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_adamicadar(self, data):
		if not self._config.is_enabled:
			return False

		# Save the data in a file
		with open(self.file_name_measure_adamicadar, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row[0], row[1]] + data[row])

	# Read adamic adar data from CSV file
	# Retuns a dict with nodes tuple as key and measures as values
	def read_measure_adamicadar(self):
		data = {}
		with open(self.file_name_measure_adamicadar, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data[(row[0], row[1])] = [float(x) for x in row[2:]]
		return data

	# Check if common neighbor measures CSV file exist
	def exist_measure_commonneighbor(self):
		return os.path.exists(self.file_name_measure_commonneighbor) and self._config.is_enabled

	# Save the common neighbor data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_commonneighbor(self, data):
		if not self._config.is_enabled:
			return False

		# Save the data in a file
		with open(self.file_name_measure_commonneighbor, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row[0], row[1]] + data[row])

	# Read common neighbor data from CSV file
	# Retuns a dict with nodes tuple as key and measures as values
	def read_measure_commonneighbor(self):
		data = {}
		with open(self.file_name_measure_commonneighbor, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data[(row[0], row[1])] = [float(x) for x in row[2:]]
		return data

	# Check if preferential attchment measures CSV file exist
	def exist_measure_preferentialattachment(self):
		return os.path.exists(self.file_name_measure_preferentialattachment) and self._config.is_enabled

	# Save the preferentialattachment data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_preferentialattachment(self, data):
		if not self._config.is_enabled:
			return False

		# Save the data in a file
		with open(self.file_name_measure_preferentialattachment, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row[0], row[1]] + data[row])

	# Read preferentialattachment data from CSV file
	# Retuns a dict with nodes tuple as key and measures as values
	def read_measure_preferentialattachment(self):
		data = {}
		with open(self.file_name_measure_preferentialattachment, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data[(row[0], row[1])] = [float(x) for x in row[2:]]
		return data

	# Check if distance matrix CSV file exist
	def exist_preprocess_distance_matrix(self):
		return os.path.exists(self.file_name_preprocess_distance_matrix) and self._config.is_enabled

	# Save the distance matrix data
	def save_reprocess_distance_matrix(self, data):
		if not self._config.is_enabled:
			return False

		# Flatten the data into a list of lists
		# First two items in a list represents names of nodes
		data_flat = []
		for m in data:
			for n in data[m]:
				data_flat.append([m, n, data[m][n]])

		# Save the data in a file
		with open(self.file_name_preprocess_distance_matrix, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data_flat:
				datawriter.writerow(row)

	# Read distance matrix data from CSV file
	def read_reprocess_distance_matrix(self):
		data = {}
		with open(self.file_name_preprocess_distance_matrix, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				if row[0] not in data:
					data[row[0]] = {}
				data[row[0]][row[1]] = float(row[2])
		return data


