# -*- coding: utf-8 -*-
# Process the cond-mat file and add to sqlite-db

import sqlite3, csv, hashlib

FILE = 'db/cond-mat'
DB_NAME = 'db/cond-mat.db'

# The database class
class Database():
	
	def __init__(self):
		self.conn = sqlite3.connect(DB_NAME)
		self.cursor = self.conn.cursor()

	def addData(self, ufrom, uto , utime):
		data = (ufrom, uto, utime)
		print(data)
		self.cursor.execute("INSERT INTO `mentions` (`from`, `to`, `time`) VALUES (?,?,?)", data)

	def close(self):
		self.conn.commit()
		self.conn.close()


def readFile():
	names = []
	db = Database()
	with open(FILE, 'rb') as csvfile:
		datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in datareader:
			if len(row) > 3 :
				for x in xrange(3, len(row)):
					for y in xrange(x+1, len(row)):
						if row[y] != '':
							u1 = hashlib.sha224(row[x]).hexdigest()
							u2 = hashlib.sha224(row[y]).hexdigest()
							db.addData(u1, u2, int(float(row[1])))
							db.addData(u2, u1, int(float(row[1])))
	db.close()

if __name__ == '__main__':
	readFile()
