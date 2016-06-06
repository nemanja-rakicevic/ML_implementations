import numpy as np 
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance


def findClosest(points, centres):
	num_k = centres.shape[1]
	num_sampl = points.shape[1]
	dist_k = np.zeros((num_k, num_sampl))

	for i in range(0, num_k):
		dist_k[i,:] = [distance.euclidean(a, centres[:,i]) for a in points.T]

	return dist_k.argmin(axis=0)

def checkImprovement(old, new):
	thrsh = 0.005

	# print np.mean(old-new)
	if np.abs(np.mean(old-new)) < thrsh:
		print '\nStopping.'
		return False
	else:
		print '\nImproving...'
		return True



##########################
#### Task definition #####
##########################

num_k = input("Enter number of clusters: ")
num_sampl = 1000
colors = cm.rainbow(np.linspace(0, 1, num_k))


############################################
#### Generate random data distributions ####
############################################

x = np.zeros((2, num_k * num_sampl))
mean_list = list()
cov_list = list()

# plot the ground truth data
fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(121).set_title('Ground truth')
for i in range(0, num_k):
	mean_list.append(np.random.randn(2)*8)
	tmp = np.random.randn(2,2)
	cov_list.append(np.dot(tmp, tmp.T))
	span = range(i*num_sampl, i*num_sampl+num_sampl)
	x[:, span] = np.random.multivariate_normal(mean_list[i], cov_list[i], num_sampl).T
	# plot data and their means
	plt.scatter(x[0, span].T, x[1, span].T, color=colors[i], label='$ \mu({i:.2f},{j:.2f})$'.format(i=mean_list[i][0],j=mean_list[i][1]))
	plt.scatter(mean_list[i][0].T, mean_list[i][1].T, s=300, marker=(4, 0), facecolor = colors[i], edgecolor='black', linewidth='2')

plt.axis('equal')
plt.legend(loc='best').get_frame().set_alpha(0.5)
plt.tight_layout()
plt.draw()


##########################
#### Perform K-means #####
##########################

# Initialise cluster centers guesses
clstr = np.zeros((2, num_k))
new_clstr = np.zeros((2, num_k))
for k in range(0, num_k):
	clstr[:,k] = np.random.randn(2)

improving = True
it = 0

# plot initial cluster center positions
plt.subplot(122).set_title('K-means clustering')
plt.axis('equal')
plt.scatter(x[0, :].T, x[1, :].T, color = 'black')
plt.scatter(clstr[0, :].T, clstr[1, :].T, s=300, marker=(7, 1), facecolor = 'red', edgecolor='black', linewidth='2')
plt.pause(2)

while (improving or it>100):

	plt.subplot(122)
	plt.cla()
	# go through all points, and assign each to a particular cluster centre
	affiliation = findClosest(x, clstr)
	# draw() - change the colors depending on affiliation
	for cp in range(0, num_k):
		plt.scatter(x[0, affiliation==cp].T, x[1, affiliation==cp].T, color=colors[cp],label='$ \mu({i:.2f},{j:.2f})$'.format(i=clstr[0,cp],j=clstr[1,cp]))
		if not x[:, affiliation==cp].any():
			new_clstr[:,cp] = clstr[:,cp]
		else:
			new_clstr[:,cp] = np.mean(x[:, affiliation==cp], axis=1)

	for cc in range(0, num_k):
		plt.scatter(clstr[0, cc].T, clstr[1, cc].T, s=300, marker=(7, 1), facecolor = colors[cc], edgecolor='black', linewidth='2')
	
	plt.axis('equal')
	plt.legend(loc='best').get_frame().set_alpha(0.5)
	plt.draw()
	plt.pause(1)
	# check improvement
	improving = checkImprovement(clstr, new_clstr)
	it = it+1
	# recalculate the new cluster centers
	clstr = new_clstr
	new_clstr = np.zeros((2, num_k))


print '\nDone after {iter} iterations.'.format(iter = it)
plt.show()	


