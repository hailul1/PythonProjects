import sys
import numpy as np
import matplotlib.pyplot as plt

#Normalize data
def min_max(list):
	lst_min = min(list)
	lst_max = max(list)
	j = 0
	for i in list:
		list[j] = float((list[j]-lst_min)/(lst_max-lst_min))
		j += 1

#Calculate Square Error as SUM(1/2*(y-y^)^2)
def square_error(m, b, list):
	y_hat = 0.0 # y^ = mx + b 
	sum_erro = 0.0
	SE = 0.0
	for i in list:
		y_hat = (m*i[0]) + b 
		sum_error = (((i[1]-y_hat))**2)/2
		SE = SE + sum_error
	return SE

# Calculate partial derivatives to determine new m and be values
#  dSE/db = -(y-y^)
#  dSE/dm = -(y-y^)x
# Update m and b using:
# b = b - alpha * partial deriv sum(dSE/db)
# m = m - alpha * partial deriv sum(dSE/dm)
def gradient(b, m, list, alpha):
	gradient1 = 0.0
	gradient2 = 0.0
	for i in list:
		y_hat = (m*i[0]) + b
		tmp1 = -(i[1]-y_hat)
		tmp2 = -(i[1]-y_hat)*i[0]
		gradient1 = gradient1 + tmp1
		gradient2 = gradient2 + tmp2

	b = b - (alpha*gradient1)
	m = m - (alpha*gradient2)
	return m, b

# Run gradient descent
def model(m, b, alpha, iterate, data):
	list = np.genfromtxt(data, delimiter=',')
	listcopy = list.copy()
	x_list = []
	y_list = []
	for i in list:
		x_list.append(float(i[0]))
		y_list.append(float(i[1]))
	min_max(x_list)
	min_max(y_list)
	k = 0
	d = 0
	for i in list:
		i[0] = float(x_list[k])
		i[1] = float(y_list[k])
		k += 1
		d += 1
	m = np.random.rand(1)
	b = np.random.rand(1)
	alpha = 0.01
	iterate = 1000
	for x in range(iterate): 
		SE = square_error(m,b, list)
		m, b = gradient(b, m, list, alpha)

	xlst = []
	ylst = []
	for i in listcopy:
		xlst.append(float(i[0]))
		ylst.append(float(i[1]))
	plt.axis([50, 80, 50, 80])
	t = np.linspace(50, 80)
	print ("Final SE = %s" % SE)
	print ("Final Slope = %s" % m)
	print ("Final Intercept =  %s" % b)
	plt.plot(xlst, ylst, 'bo', t, m * t + b, 'r-')
	plt.show()
		


m = 0.0
b = 0.0
alpha = 0.00
iterate = 0
data = open("myData.txt", "r")

if __name__ == '__main__':	model(m, b, alpha, iterate, data)


