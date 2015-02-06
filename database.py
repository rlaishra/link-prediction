# The main database class 

from __future__ import division
import sqlite3, time

class Database(object):

	def __init__(self, file_name):
		self.file_name = file_name

	# Open a database connection
	def open(self):
		self.connection = sqlite3.connect(self.file_name)
		self.cursor = self.connection.cursor()

	# Commit after making changes to the database
	def commit(self):
		self.connection.commit()

	# Close database connection
	def close(self):
		self.connection.close()

	# Users are valid if they have atleast one inlink and one outlink
	# Returns a dict with user names as keys and 1 as value
	def valid_users(self):
		names = {}
		for row in self.cursor.execute("SELECT DISTINCT `from` FROM (SELECT * FROM `mentions` WHERE `from` IN (SELECT `to` FROM `mentions`) INTERSECT SELECT * FROM `mentions` WHERE `to` IN (SELECT `from` FROM `mentions`))"):
			names[row[0]] = 1
		return names

	# Returns the adjacency list of the links between time_start and time_end
	# time_start inclusive
	# Value is count of links
	# Directed
	# If users is not None, only links between users returned
	def get_links(self, time_start, time_end, users):
		links = {}
		for row in self.cursor.execute("SELECT * FROM `mentions` WHERE `time` < ? AND `time` >= ?", (time_end, time_start)):
			if users is not None and (row[0] not in users) and (row[1] not in users):
				continue
			if row[0] not in links:
				links[row[0]] = {}
			if row[1] in links[row[0]]:
				links[row[0]][row[1]] = links[row[0]][row[1]] + 1
			else:
				links[row[0]][row[1]] = 1
		return links

	# Return the minimum and max time
	def get_time_min_max(self):
		self.cursor.execute("SELECT MIN(`time`), MAX(`time`) FROM `mentions`")
		return self.cursor.fetchone()
