import random

def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata)*fraction)
	return ldata[:breakPoint], ldata[breakPoint:]