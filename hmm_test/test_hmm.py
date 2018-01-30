
"""
my implementation based on https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/

"""


import numpy as np 

"""
# n - states dim (fair or biased coin used)
# m - observations dim (coin toss is heads or tails)

# transition matrix (n x n)
A 

# observation matrix (n x m)
O 
# probability observation matrix (n x m)
O_hat

# sequence of states (n x k)
X 
# sequence of observations (m x k)
Y
"""

def viterbi(A, O, Y):
    n, m = O.shape
    k = Y.shape[0]
    probs = np.zeros((n, k+1))
    probs[:,0] = np.arange(n)
    # probs[:,0] = np.ones((n)) / n
    for t in range(k):
        # get probabilities for all states at each timestep based on observations
        probs[:,t+1] = np.dot(np.transpose(A), probs[:,t]) * O[int(Y[t])]
        # normalise
        probs[:,t+1] /= np.sum(probs[:,t+1])
    # estimate optimal state sequence
    path = np.argmax(probs, axis=0)

    return path, probs


def forward_backward(A, O, Y):
    n,m = O.shape
    k = Y.shape[0]
    fw = np.zeros((n,k+1))
    bw = np.zeros((n,k+1))
    # forward part
    fw[:,0] = 1.0/n 
    for t in range(k):
        fw[:,t+1] = np.dot(np.dot(fw[:,t], A), O)
        fw[:,t+1] /= np.sum(fw[:,t+1])
    # backward part
    bw[:,-1] = 1.0
    for t in range(k, 0,-1):
        O_hat = np.diag(O[:, int(Y[t-1])])  # multiplying with the appropriate observation probabilities
        bw[:,t-1] = np.dot(np.dot(A, O_hat), bw[:,t])
        bw[:,t-1] /= np.sum(bw[:,t-1])
    # combination
    prob = fw * bw
    prob /= np.sum(prob, axis=0)
    
    return prob, fw, bw 


def maximization(P, F, B, A, O, Y):
    """
    NOT GOOD:   because using only values from the observations and
                not everything provided by forward-backwad algorithm
    """
    n,m = O.shape
    k = Y.shape[0]
    theta = np.zeros((n,m,k))
    # update theta
    for i in range(k):
        alpha = Y[i]
        beta  = Y[i+1]
        theta[alpha, beta, i] = F[alpha, i] * A[alpha, beta] * B[beta, i+1] * O[beta, Y[i]]
        theta[:, :, i] /= np.sum(theta, axis=2)
    # update transition matrix
    A = np.sum(theta, axis=2) / np.tile(np.sum(P, axis=2), (2,1)).T
    # update observation matrix
    for j in range(m):
        O[:,j] = np.sum(P[:,Y==j]) / np.sum(P, axis=1) 


def baum_welch(n, m, Y):
    # initialisaiton
    k = Y.shape[0]
    A = np.ones((n,n)) / n
    O = np.ones((n,m)) / m
    theta = np.zeros((n,n,k))
    # run algorithm till convergence
    while True:
        # split steps
        A_old = A
        O_old = O
        A = np.ones((n,n))
        O = np.ones((n,m))
        # _E_ step
        P,F,B = forward_backward(A_old, O_old, Y)
        # _M_ step
        # update theta
        for alpha in range(n):
            for beta in range(n):
                for t in range(k):
                    theta[alpha, beta, t] = F[alpha, t] * A_old[alpha, beta] * B[beta, t+1] * O_old[beta, int(Y[t])]


        # update transition matrix
        A = np.sum(theta, axis=2) / np.tile(np.sum(P, axis=1), (2,1)).T
        A /= np.sum(A, 1)

        # for a_ind in xrange(num_states):
        #     for b_ind in xrange(num_states):
        #         A_mat[a_ind, b_ind] = np.sum( theta[a_ind, b_ind, :] ) / np.sum(P[a_ind,:])
        # A_mat = A_mat / np.sum(A_mat,1)


        # update observation matrix
        for j in range(m):
            x_ind = np.array(np.where(Y==j))+1
            O[:,j] = np.sum(P[:, x_ind]) / np.sum(P[:,1:], axis=1)
        O /= np.sum(O, 1)

        # for a_ind in xrange(num_states):
        #     for o_ind in xrange(num_obs):
        #         right_obs_ind = np.array(np.where(observ == o_ind))+1
        #         O_mat[a_ind, o_ind] = np.sum(P[a_ind,right_obs_ind])/ np.sum( P[a_ind,1:])
        # O_mat = O_mat / np.sum(O_mat,1)


        # check convergence
        if np.linalg.norm(A_old - A) < 0.00001 and np.linalg.norm(O_old - O) < 0.00001:
            break

    return A, O



# TEST
A_mat = np.array([[.6, .4], [.2, .8]])
O_mat = np.array([[.5, .5], [.15, .85]])

num_obs = 25
observations1 = np.random.randn( num_obs )
observations1[observations1>0] = 1
observations1[observations1<=0] = 0

observations2 = np.random.random(num_obs)
observations2[observations2>.15] = 1
observations2[observations2<=.85] = 0


Z, pp = viterbi(A_mat, O_mat, observations1)



A_mat, O_mat = baum_welch(2,2,observations1)
A_mat, O_mat = baum_welch(2,2,observations2)