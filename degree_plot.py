# Plot the in degree and out degree of the networks

from libs import database, graph, measures, cache, preprocess, config, data
import random
import matplotlib.pyplot as plt
from pprint import pprint


class DegreePlot():
	def __init__(self, run_config):
		self._run = run_config
		self._config = config.Data()
		self._data = data.Data(run_config)
		self._db = database.Database()
	
	# Plot over the entire time series combined
	def cumulative(self):
		# Get the sample users
		sample_users = self._data.get_sample_users()

		# Get liks that contains the sample users
		if self._run.verbose:
			print('Fetching links.')
		self._db.open()
		self._db.open()
		time_start, time_end = self._db.get_time_min_max()
		links = self._db.get_links(time_start, time_end, sample_users)
		if self._run.verbose:
			print(str(len(links)) + ' links fetched')

		if self._run.verbose:
			print('Calculating in degree and out degree')
		mes = measures.Measures()
		in_degree = mes.in_degree(links, sample_users)
		out_degree = mes.out_degree(links, sample_users)
		if self._run.verbose:
			print('In degree and out degree calculated')

		# Initialize x axis and y axis
		# X axis is the degree; y axis is the count
		# X axis max is the max value of degree
		# y1 indegree; y2 out degree
		x = xrange(0, max([in_degree[x] for x in in_degree]+[out_degree[x] for x in out_degree])+1)
		y1 = [0]*len(x)
		y2 = [0]*len(x)

		for node in in_degree:
			y1[in_degree[node]] += 1
		for node in out_degree:
			y2[out_degree[node]] += 1
		
		# Plot the in degree and out degree
		plt.subplot(2,1,1)
		plt.plot(x, y1)
		plt.xlabel('In Degree')
		plt.ylabel('Count')
		plt.yscale('log')
		plt.xscale('log')

		plt.subplot(2,1,2)
		plt.plot(x, y2)
		plt.xlabel('Out Degree')
		plt.ylabel('Count')
		plt.yscale('log')
		plt.xscale('log')

		plt.savefig('degree_plot_cumulative.png')
		plt.show()



def main():
	sample_size = 5000
	
	fio = fileio.Fileio()

	db = database.Database('db/database.db')
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
