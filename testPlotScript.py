from matplotlib import pyplot
import numpy
import time

def plotLoop():
	#pyplot.ion()
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	img = ax.imshow(numpy.zeros((128,128)))
	pyplot.draw()
	raw_input("")
	for i in range(10):
		data = numpy.random.random((128,128))
		ax.imshow(data)

		pyplot.draw()
		print ("PLOT!")
		pyplot.pause(0.5)
	return img


if __name__ == "__main__":
	plotLoop()