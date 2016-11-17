import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
z = iris.data[:,:2]
print z
x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y = np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
scaled_x = x
scaled_y = y
data = np.matrix([[scaled_x[i],scaled_y[i]] for i in xrange(len(scaled_x))])
mean_x = np.mean(x)
mean_y = np.mean(y)
scaled_x = x-mean_x
scaled_y = y-mean_y
data = np.matrix([[scaled_x[i],scaled_y[i]] for i in xrange(len(scaled_x))])
cov=np.cov(scaled_x,scaled_y)
np.dot(np.transpose(data),data)
eig_val, eig_vec = np.linalg.eig(cov)
xmin, xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
plt.plot([eig_vec[:,0][0],0], [eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0], [eig_vec[:,1][1],0],color='blue')
new_data = np.transpose(np.dot(eig_vec, np.transpose(data)))
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in xrange(len(eig_val))]
eig_pairs.sort(reverse=True)
feature = eig_pairs[0][1]
new_data_reduced = np.transpose(np.dot(feature,np.transpose(data)))
plt.plot(new_data_reduced[:,0],[1.2]*10,'*',color='green')
plt.plot(new_data[:,0],new_data[:,1],'^',color='blue')
plt.plot(scaled_x,scaled_y,'o',color='red')
print new_data_reduced
plt.show()
