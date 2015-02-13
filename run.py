# The main script
# invoke everything from this script

import config, data, degree_plot, dynamic_features
import sys

# Set run time configs
class RuntimeConfig():
	def __init__(self):
		self._config = config.Run()
		self._functions = {
			'v'	: self._verbose,
		}

	def _verbose(self):
		self._config.verbose = True

	# Start config with -
	# v -> verbose mode on
	def set(self, vars):
		if vars[0] != '-':
			print('RuntimeConfig not set')
			return False
		vars = vars[1:]

		for f in vars:
			self._functions[f]()
		return self._config

# The main class
class Run():
	def __init__(self, config_run):
		self._config_run = config_run
		self._functions = {
			'test' 				: self._test,
			'valid_users'		: self._valid_users,
			'sample_users'		: self._sample_users,
			'degree_plot'		: self._degree_plot,
			'dynamic_features'	: self._dynamic_features,
		}

	def start(self, argv):
		if argv[0] not in self._functions.keys():
			print('Invalid Parameter')
			return False
		self._functions[argv[0]](argv[1:])

	# A test function
	def _test(self, argv):
		fea = dynamic_features.DynamicFeatures(self._config_run)
		values = fea.read(argv[0])
		counter = 0
		for l in values:
			if any([x > 0 for x in values[l]]):
				counter += 1
		print(counter)
		print(len(values))

	# Generate valid users and save them in the cache
	def _valid_users(self, argv):
		dat = data.Data(self._config_run)
		count = len(dat.get_valid_users())
		print(str(count) + ' valid users generated')

	# Generate sample users and save them in the cache
	def _sample_users(self, argv):
		dat = data.Data(self._config_run)
		count = len(dat.get_sample_users())
		print(str(count) + ' sample users generated')

	# Degree plot
	# Cululative or time series
	def _degree_plot(self, argv):
		plot_type = argv[0]
		if plot_type not in ['cumulative', 'time_series']:
			print('Invalid plot type')
			return False
		plt = degree_plot.DegreePlot(self._config_run)
		if plot_type == 'cumulative':
			plt.cumulative()
		else:
			pass

	def _dynamic_features(self, argv):
		fea = dynamic_features.DynamicFeatures(self._config_run)
		fea.generate(argv)

# Command format:
# python run.py -<run options> <task_name> <task_arg>

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Missing parmeters')
	else:
		r = RuntimeConfig()
		config_run = r.set(sys.argv[1])

		run = Run(config_run)
		run.start(sys.argv[2:])