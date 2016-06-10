########################################################
# Basic implemenatation of the Gaussian Mixture Models
# using Expectation Maximisation.
# The samples and initial values are initialised as
# random.
# Nice explanation: http://mccormickml.com/2014/08/04/gaussian-mixture-models-tutorial-and-matlab-code/
# Other implementation: http://math.stackexchange.com/questions/1388853/implementation-of-em-algorithm-for-gaussian-mixture-models-using-matlab
########################################################

import numpy as np 
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from scipy.spatial import distance
from scipy.stats import multivariate_normal as mv_gauss


def findClosest(points, centres):
	num_k = centres.shape[1]
	num_sampl = points.shape[1]
	dist_k = np.zeros((num_k, num_sampl))

	for i in range(0, num_k):
		dist_k[i,:] = [distance.euclidean(a, centres[:,i]) for a in points.T]

	return dist_k.argmin(axis=0)

def checkImprovement(old, new, it):
	thrsh = 0.0005

	# print np.mean(old-new)
	if np.abs(np.mean(old-new)) < thrsh:
		print '\niter: %i - Stopping.'%it
		return False
	else:
		print '\niter: %i - Improving...'%it
		return True

def getGauss(x, mu, cov):
	return [(1/(np.sqrt(np.linalg.det(cov)*np.power(2*np.pi, 2)))) * np.exp(-0.5 * np.dot(np.dot((x[:,i]-mu).T, np.linalg.inv(cov)), (x[:,i]-mu))) for i in range(0,x.shape[1])]


def getPosteriors(x, mu, cov, priors):
	num_k = mu.shape[1]
	# print getGauss(x, mu[:,0], cov[0])
	# print mv_gauss.pdf(x.T, mu[:,0], cov[0]) 
	# print priors
	posteriors = np.zeros((num_k, x.shape[1]))
	for p in range(0, num_k):
		posteriors[p,:] = mv_gauss.pdf(x.T, mu[:,p], cov[p]) * priors[p]
		# np.tile(priors[p],(1,x.shape[1]))
		# posteriors[p,:] = getGauss(x, mu[:,p], cov[p]) * priors[p]
	# sum_posteriors = np.sum(posteriors, axis = 0)
	posteriors/=np.sum(posteriors, axis = 0)
	return posteriors

def drawElipse(mu, cov, *arg):
	# Draw an elipse representing the covariance matrix at a certain confidence level
	nstd = 2
	# get eigenvaluse to determine the orientation
	vals,vecs = np.linalg.eigh(cov)
	order = vals.argsort()
	vals = vals[order[::-1]]
	vecs = vecs[:,order]
	theta = np.degrees(np.arctan2(*vecs[:,0]))
	width,height = 2 * nstd * np.sqrt(vals)
	if len(arg):
		ellip = Ellipse(xy=mu, width=width, height=height, angle=-theta, facecolor='none', edgecolor=arg[0])
	else:
		ellip = Ellipse(xy=mu, width=width, height=height, angle=-theta, facecolor='none')
	return ellip


########################################################################################################
#### Task definition #####
########################################################################################################

num_k = input("Enter number of clusters: ")
num_sampl = 100
colors = cm.rainbow(np.linspace(0, 1, num_k))


############################################
#### Generate random data distributions ####
############################################

x = np.zeros((2, num_k * num_sampl))
mean_list = list()
cov_list = list()

# plot the ground truth data
fig = plt.figure(figsize=(20, 10))
ax1 = plt.subplot(121)
plt.subplot(121).set_title('Ground truth')
for i in range(0, num_k):
	mean_list.append(np.random.randn(2)*num_k)
	tmp = np.random.randn(2,2)
	cov_list.append(np.dot(tmp, tmp.T))
	span = range(i*num_sampl, i*num_sampl+num_sampl)
	x[:, span] = np.random.multivariate_normal(mean_list[i], cov_list[i], num_sampl).T
	# plot data and their means
	plt.scatter(x[0, span].T, x[1, span].T, color=colors[i], label='$ \mu({i:.2f},{j:.2f})$'.format(i=mean_list[i][0],j=mean_list[i][1]))
	plt.scatter(mean_list[i][0].T, mean_list[i][1].T, s=300, marker=(4, 0), facecolor = colors[i], edgecolor='black', linewidth='2')
	ax1.add_artist(drawElipse(mean_list[i], cov_list[i], colors[i]))

plt.axis('equal')
plt.legend(loc='best').get_frame().set_alpha(0.5)
# plt.tight_layout()
plt.draw()


#################################
#### Perform GMM through EM #####
#################################

# Initialise cluster centers, priors and covariance guesses
priors = np.tile([1./num_k], (num_k))
means = np.random.randn(2, num_k)*num_k
# means = np.zeros((2, num_k))
new_means = np.zeros((2, num_k))
covariances = list()
new_covariances = [np.zeros((2,2))] * num_k
for cc in range(0,num_k):
	tmp = np.random.randn(2,2) * num_k
	covariances.append(np.dot(tmp, tmp.T))
	# covariances.append(np.eye(2)*30)

improving = True
it = 0

# plot initial distribution centre positions and confidence elipses
ax2 = plt.subplot(122)
plt.subplot(122).set_title('GMM clustering')
plt.axis('equal')
plt.scatter(x[0, :].T, x[1, :].T, color = 'black')
plt.scatter(means[0, :].T, means[1, :].T, s=300, marker=(7, 1), facecolor = 'red', edgecolor='black', linewidth='2')
for pl in range(0,num_k):	
	ax2.add_artist(drawElipse(means[:,pl], covariances[pl]))
plt.pause(2)


while (improving and it<=100):
# while (it<=50):
	### Expectation step
	# mean, cov - fixed
	# compute probability of a sample belonging to a cluster (w - weight)
	posts = getPosteriors(x, means, covariances, priors)
	# print posts


	### ~~ Maximisation step
	# update: mean, cov
	# based on the weighted samples
	priors = np.sum(posts, axis=1)

	for k in range(0,num_k):
		new_means[:,k] = np.sum(x * posts[k,:], axis=1) / priors[k]
		tmp = x - new_means[:,k].reshape(2,1)
		new_covariances[k] = sum([ np.outer(tmp[:,r],tmp[:,r]) * posts[k,r] for r in range(0,posts.shape[1])]) / priors[k]
		# TO DO: check for singular covariance matrices !!!

	priors /= priors.sum()

	### re-plot: new means, covariance ellipsoids and data affiliation
	plt.cla()
	plt.subplot(122).set_title('GMM clustering - iteration: %i'%it)

	plt.scatter(x[0, :].T, x[1, :].T, color = 'black')

	for cc in range(0, num_k):
		idx = posts[cc,:] > 1./len(posts[cc,:])
		plt.scatter(x[0, idx].T, x[1, idx].T, color=colors[cc],label='$ \mu({i:.2f},{j:.2f})$'.format(i=new_means[0,cc],j=new_means[1,cc]))
		plt.scatter(new_means[0, cc].T, new_means[1, cc].T, s=300, marker=(7, 1), facecolor = colors[cc], edgecolor='black', linewidth='2')
		ax2.add_artist(drawElipse(new_means[:,cc], new_covariances[cc], colors[cc]))
	
	plt.axis('equal')
	plt.legend(loc='best').get_frame().set_alpha(0.5)
	plt.draw()
	plt.pause(0.4/num_k)

	### check improvement
	improving = checkImprovement(means, new_means, it)
	print '\n % of points per cluster: ', priors*100
	it = it+1

	# update
	means = new_means
	new_means = np.zeros((2, num_k))
	covariances = new_covariances
	new_covariances = [np.zeros((2,2))] * num_k


print '\nDone after {iter} iterations.'.format(iter = it)
plt.show()	


