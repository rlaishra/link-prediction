# The main script
# invoke everything from this script

import sys, config, data

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
	def set(self, str):
		if str[0] != '-':
			print('RuntimeConfig not set')
			return False
		str = str[1:]

		for f in str:
			self._functions[f]()
		return self._config

# The main class
class Run():
	def __init__(self, config_run):
		self._config_run = config_run
		self._functions = {
			'test' 			: self._test,
			'valid_users'	: self._valid_users,
			'sample_users'	: self._sample_users,
		}

	def start(self, argv):
		if argv not in self._functions.keys():
			print('Invalid Parameter')
			return False
		self._functions[argv]()

	# A test function
	def _test(self):
		dat = data.Data(self._config_run)
		dat.get_valid_users()

	# Generate valid users and save them in the cache
	def _valid_users(self):
		dat = data.Data(self._config_run)
		count = len(dat.get_valid_users())
		print(str(count) + ' valid users generated')

	# Generate sample users and save them in the cache
	def _sample_users(self):
		dat = data.Data(self._config_run)
		count = len(dat.get_sample_users())
		print(str(count) + ' sample users generated')


if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Missing parmeters')
	else:
		r = RuntimeConfig()
		config_run = r.set(sys.argv[1])

		run = Run(config_run)
		run.start(sys.argv[2])