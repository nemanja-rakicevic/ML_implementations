
"""
proper implementation from https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/
"""


import numpy as np
# functions and classes go here


def viterbi_alg(A_mat, O_mat, observations):
    # get number of states
    num_obs = observations.size
    num_states = A_mat.shape[0]
    # initialize path costs going into each state, start with 0
    log_probs = np.zeros(num_states)
    # initialize arrays to store best paths, 1 row for each ending state
    paths = np.zeros( (num_states, num_obs+1 ))
    paths[:, 0] = np.arange(num_states)
    # start looping
    for obs_ind, obs_val in enumerate(observations):
        # for each obs, need to check for best path into each state
        for state_ind in range(num_states):
            # given observation, check prob of each path
            temp_probs = log_probs + \
                         np.log(O_mat[state_ind, int(obs_val)]) + \
                         np.log(A_mat[:, state_ind])
            # check for largest score
            best_temp_ind = np.argmax(temp_probs)
            # save the path with a higher prob and score
            paths[state_ind,:] = paths[best_temp_ind,:]
            paths[state_ind,(obs_ind+1)] = state_ind
            log_probs[state_ind] = temp_probs[best_temp_ind]
    # we now have a best stuff going into each path, find the best score
    best_path_ind = np.argmax(log_probs)
    # done, get out.
    return (best_path_ind, paths, log_probs)


def fb_alg(A_mat, O_mat, observ):
    # set up
    k = observ.size
    (n,m) = O_mat.shape
    prob_mat = np.zeros( (n,k) )
    fw = np.zeros( (n,k+1) )
    bw = np.zeros( (n,k+1) )
    # forward part
    fw[:, 0] = 1.0/n
    for obs_ind in xrange(k):
        f_row_vec = np.matrix(fw[:,obs_ind])
        fw[:, obs_ind+1] = f_row_vec * \
                           np.matrix(A_mat) * \
                           np.matrix(np.diag(O_mat[:, int(observ[obs_ind])]))
        fw[:,obs_ind+1] = fw[:,obs_ind+1]/np.sum(fw[:,obs_ind+1])
    # backward part
    bw[:,-1] = 1.0
    for obs_ind in xrange(k, 0, -1):
        b_col_vec = np.matrix(bw[:,obs_ind]).transpose()
        bw[:, obs_ind-1] = (np.matrix(A_mat) * \
                            np.matrix(np.diag(O_mat[:, int(observ[obs_ind-1])])) * \
                            b_col_vec).transpose()
        bw[:,obs_ind-1] = bw[:,obs_ind-1]/np.sum(bw[:,obs_ind-1])
    # combine it
    prob_mat = np.array(fw)*np.array(bw)
    prob_mat = prob_mat/np.sum(prob_mat, 0)
    # get out
    return prob_mat, fw, bw

 
def baum_welch( num_states, num_obs, observ ):
    # allocate
    A_mat = np.ones( (num_states, num_states) )
    A_mat = A_mat / np.sum(A_mat,1)
    O_mat = np.ones( (num_states, num_obs) )
    O_mat = O_mat / np.sum(O_mat,1)
    theta = np.zeros( (num_states, num_states, observ.size) )
    while True:
        old_A = A_mat
        old_O = O_mat
        A_mat = np.ones( (num_states, num_states) )
        O_mat = np.ones( (num_states, num_obs) )
        # expectation step, forward and backward probs
        P,F,B = fb_alg( old_A, old_O, observ)
        # need to get transitional probabilities at each time step too
        for a_ind in xrange(num_states):
            for b_ind in xrange(num_states):
                for t_ind in xrange(observ.size):
                    theta[a_ind,b_ind,t_ind] = \
                    F[a_ind,t_ind] * \
                    B[b_ind,t_ind+1] * \
                    old_A[a_ind,b_ind] * \
                    old_O[b_ind, int(observ[t_ind])]
        # form A_mat and O_mat
        for a_ind in xrange(num_states):
            for b_ind in xrange(num_states):
                A_mat[a_ind, b_ind] = np.sum( theta[a_ind, b_ind, :] )/ \
                                      np.sum(P[a_ind,:])
        A_mat = A_mat / np.sum(A_mat,1)
        for a_ind in xrange(num_states):
            for o_ind in xrange(num_obs):
                right_obs_ind = np.array(np.where(observ == o_ind))+1
                O_mat[a_ind, o_ind] = np.sum(P[a_ind,right_obs_ind])/ \
                                      np.sum( P[a_ind,1:])
        O_mat = O_mat / np.sum(O_mat,1)
        # compare
        if np.linalg.norm(old_A-A_mat) < .00001 and np.linalg.norm(old_O-O_mat) < .00001:
            break
    # get out
    return A_mat, O_mat
 
num_obs = 25
observations1 = np.random.randn( num_obs )
observations1[observations1>0] = 1
observations1[observations1<=0] = 0
A_mat, O_mat = baum_welch(2,2,observations1)
print observations1
print A_mat
print O_mat
observations2 = np.random.random(num_obs)
observations2[observations2>.15] = 1
observations2[observations2<=.85] = 0
A_mat, O_mat = baum_welch(2,2,observations2)
print observations2
print A_mat
print O_mat
A_mat, O_mat = baum_welch(2,2,np.hstack( (observations1, observations2) ) )
print A_mat
print O_mat


# we have what we need, do viterbi
# the transition matrix
A_mat = np.array([[.6, .4], [.2, .8]])
# the observation matrix
O_mat = np.array([[.5, .5], [.15, .85]])
best_path_ind, paths, log_probs = viterbi_alg(A_mat, O_mat, observations1)
print "obs1 is " + str(observations1)
print "obs1, best path is" + str(paths[best_path_ind,:])