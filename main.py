from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math


DATAPOINTS = 40
EPSILON = 10 ** -6

# options: linear, poly, radial, sigmoid
KERNEL = "poly"

# params for poly:
DEGREE = 2

# params for raidal:
SIGMA = 2

# params for sigmoid:
K = 3
DELTA = 0.5

def generateData():
	""" Generate the datasets in form of two different classes with different distrubutions 
	Returns a list of 3-tuples, (x-value, y-value, class)"""
	
	assert(DATAPOINTS % 2 == 0)
	classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(int(DATAPOINTS/4))]
	classA += [(random.normalvariate(2.5, 1), random.normalvariate(0.0, 0.5), 1.0) for i in range(int(DATAPOINTS/4))]

	classB = [(random.normalvariate(0.0, 1.5), random.normalvariate(-1.9, 0.5), -1.0) for i in range(int(DATAPOINTS/2))]

	data = classA + classB
	random.shuffle(data)

	pylab.hold(True)
	pylab.plot([p[0] for p in classA],
		   [p[1] for p in classA],
		   'bo')
	pylab.plot([p[0] for p in classB],
		   [p[1] for p in classB],
		   'ro')
	return data

def kernel(x,y):
	if KERNEL == "linear":
		return numpy.dot(x,y) + 1
	elif KERNEL == "poly":
		return (numpy.dot(x,y) + 1) ** DEGREE
	elif KERNEL == "radial":
		v = numpy.subtract(x,y)
		return numpy.exp(-((numpy.dot(v,v)**2)/(2*SIGMA**2)))	
	elif KERNEL == "sigmoid":
		return numpy.tanh(K*numpy.dot(x,y) - DELTA)

def createP(data):
	"""Create the P-matrix used by qp. Returns a numpy matrix of P"""
	P = [[] for i in range(DATAPOINTS)]
	for i in range(DATAPOINTS):
		for j in range(DATAPOINTS):
			P[i].append(data[i][2]*data[j][2]*kernel(data[i][:2], data[j][:2]))
	
	return numpy.matrix(P)

def getSupportVectors(alpha, data):
	"""Get the supportvectors (non zero alphas) from the dataset and returns them 
	together with their datapoints as a list of tuples"""
	res = []
	for i in range(len(data)):
		if abs(alpha[i]) > EPSILON:
			res.append((alpha[i], data[i]))
	return res

def indicator(x,y, support):
	res = 0
	for i in range(len(support)):
		alpha, datapoint = support[i]
		res += alpha*datapoint[2] * kernel((x,y),datapoint[:2])
	
	return res	

def main():
	data = generateData()

	q = numpy.empty((DATAPOINTS,1))
	q[:] = -1.0

	h = numpy.empty((DATAPOINTS,1))
	h[:] = 0.0

	G = numpy.empty((DATAPOINTS,DATAPOINTS))
	numpy.fill_diagonal(G, -1)

	P = createP(data)

	r = qp(matrix(P),matrix(q),matrix(G),matrix(h))

	alpha = list(r['x'])
	
	xrange = numpy.arange(-4, 4, 0.05)
	yrange = numpy.arange(-4, 4, 0.05)
	support = getSupportVectors(alpha, data)
	
	grid = matrix([[indicator(x,y,support) for y in yrange] for x in xrange])

	pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), 
			colors=('red', 'black', 'blue'),
			linewidths=(1, 1, 1))
	
	pylab.show()

main()
