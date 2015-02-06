import csv, os.path

# This class is used to save and read stuffs into file


class Fileio():
	def __init__(self):
		self.file_name_valid_users = 'fileio/valid_users.csv'

	def exist_valid_users(self):
		return os.path.exists(self.file_name_valid_users)

	def save_valid_users(self, data):
		with open(self.file_name_valid_users, 'wb') as csvfile:
			datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				datawriter.writerow([row])

	def read_valid_users(self):
		data = []
		with open(self.file_name_valid_users, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				data.append(row[0])
		return data