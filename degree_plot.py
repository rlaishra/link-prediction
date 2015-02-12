# Plot the in degree and out degree of the networks

import database, graph, random, measures, fileio, preprocess
from pprint import pprint

def main():
	sample_size = 2500
	
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
		users_sample = random.sample(users, sample_size)
		fio.save_sample_users(users_sample)
	else:
		users_sample = fio.read_sample_users()

	time_start, time_end = db.get_time_min_max()

if __name__ == '__main__':
	main()
