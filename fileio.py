import csv, os.path

# This class is used to save and read stuffs into file


class Fileio():
	def __init__(self):
		self.file_name_valid_users = 'fileio/valid_users.csv'
		self.file_name_sample_users = 'fileio/sample_users.csv'
		self.file_name_measure_jaccard = 'fileio/measure_jaccard.csv'
		self.file_name_measure_adamicadar = 'fileio/measure_adamicadar.csv'
		self.file_name_measure_commonneighbor = 'fileio/measure_commonneighbor.csv'
		self.file_name_measure_preferentialattachment = 'fileio/measure_preferentialattachment.csv'
		self.file_name_preprocess_distance_matrix = 'fileio/preporcess_distance_matrix.csv'

	# Check if the valid users CSV file exist
	def exist_valid_users(self):
		return os.path.exists(self.file_name_valid_users)

	# Save valid users into file
	def save_valid_users(self, data):
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
		return os.path.exists(self.file_name_sample_users)

	# Save valid users into file
	def save_sample_users(self, data):
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
		return os.path.exists(self.file_name_measure_jaccard)

	# Save the Jaccard measures data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_jaccard(self, data):
		# Flatten the data into a list of lists
		# First two items in a list represents names of nodes
		data_flat = []
		for m in data:
			data_flat.append([m[0], m[1]] + data[m])

		# Save the data in a file
		with open(self.file_name_measure_jaccard, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data_flat:
				datawriter.writerow(row)

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
		return os.path.exists(self.file_name_measure_adamicadar)

	# Save the adamic adar data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_adamicadar(self, data):
		# Flatten the data into a list of lists
		# First two items in a list represents names of nodes
		data_flat = []
		for m in data:
			data_flat.append([m[0], m[1]] + data[m])

		# Save the data in a file
		with open(self.file_name_measure_adamicadar, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data_flat:
				datawriter.writerow(row)

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
		return os.path.exists(self.file_name_measure_commonneighbor)

	# Save the common neighbor data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_commonneighbor(self, data):
		# Flatten the data into a list of lists
		# First two items in a list represents names of nodes
		data_flat = []
		for m in data:
			data_flat.append([m[0], m[1]] + data[m])

		# Save the data in a file
		with open(self.file_name_measure_commonneighbor, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data_flat:
				datawriter.writerow(row)

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
		return os.path.exists(self.file_name_measure_preferentialattachment)

	# Save the preferentialattachment data
	# format of data (node1, node1): [m1, m2, m3, ...]
	def save_measure_preferentialattachment(self, data):
		# Flatten the data into a list of lists
		# First two items in a list represents names of nodes
		data_flat = []
		for m in data:
			data_flat.append([m[0], m[1]] + data[m])

		# Save the data in a file
		with open(self.file_name_measure_preferentialattachment, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data_flat:
				datawriter.writerow(row)

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
		return os.path.exists(self.file_name_preprocess_distance_matrix)

	# Save the distance matrix data
	def save_reprocess_distance_matrix(self, data):
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


