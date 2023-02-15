# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:59:39 2022

@author: trive
"""


import itertools
import numpy as np
import math
from numpy import linalg as LA
from random import choices
import matplotlib.pyplot as plt  
from scipy.linalg import sqrtm
import random

## State for each agent; return set [sinit, g]

def local_states():
    s = ["sinit","g"]
    return s


## returns [sinit, g]^n i.e., each possible state. we call it global state

def global_states():
    n_sta = n_local_states**n
    sta = list(itertools.product(local_state, repeat=n))
    sta = [list(sta[i]) for i in range(n_sta)]
    return sta


## returns the set of all actions for agent $i$; takes the input the feature dimension
## returns the list [-1,1]^{d-1}; d is the input to this function

def local_actions():
    basic_actions  = [-1,1]
    n_local_actions = len(basic_actions)**(feature_dimension-1)
    a = list(itertools.product(basic_actions, repeat=feature_dimension-1))
    a = [list(a[i]) for i in range(n_local_actions)]
    return a


## returns the gloabl action, i.e., [a_1, a_2, \dots, a_{2^({d-1}n})] 
## Each a_i is a vector of size n.
## Each element of this vector a_i is also a vector

def global_actions():
    n_act = n_local_actions**n
    act = list(itertools.product(local_action, repeat=n))
    act = [list(act[i]) for i in range(n_act)]
    return act


## returns the features for any state, s^i ; action a^i; and the next state s_prime^i by agent $i$
## output is a d-dimensional vector

def local_features_prob(s_i,a_i,s_prime_i):
    a_top = np.transpose(a_i)
    if s_i == local_state[0] and s_prime_i == local_state[0]:
        phi = list(-1*a_top)
        # phi = list(phi[0])
        phi.append((1-delta)/n)
        phi = np.transpose(phi)
        # print("phi",phi)
        return list(phi)
    elif s_i == local_state[0] and s_prime_i == local_state[1]:
        phi = list(a_top)
        # phi = list(phi[0])
        phi.append(delta/n)    
        phi = np.transpose(phi)
        # print("phi",phi)
        return list(phi)
    elif s_i == local_state[1] and s_prime_i == local_state[0]:
        phi = np.zeros(feature_dimension)
        # print("phi",phi)
        return list(phi)
    else:
        phi = np.zeros(feature_dimension-1)
        phi = np.transpose(phi)
        phi = list(phi)
        # phi = list(phi[0])
        phi.append(1/n)
        # print("phi",phi)
        return list(phi)
    

def global_features_prob(s,a,s_prime):
    feat=[]
    if s != global_state[-1]:
        for i in range(n):
            feat = feat + local_features_prob(s[i],a[i],s_prime[i])
#            print(feat)
    if s == global_state[-1] and s_prime != global_state[-1]:
        feat = [0 for i in range(n*feature_dimension)]
    if s == global_state[-1] and s_prime == global_state[-1]:
        feat = [0 for i in range(n*(feature_dimension) -1)]
        feat.append(2**(n-1))
    return feat


## returns the set of all transition probability parameters theta for agent i

def local_thetas():
    basic_theta = [-1*Delta/(n*(feature_dimension-1)), Delta/(n*(feature_dimension-1))]
    b = list(itertools.product(basic_theta, repeat=feature_dimension-1))
    n_theta = len(basic_theta)**(feature_dimension -1)
    b = [list(b[i]) for i in range(n_theta)]
    for i in range(len(b)):
        b[i].append(1/(2**(n-1)))
    return b


## global theta parameters that is of the form [theta^1, 1/n, theta^2, 1/n, \dots, theta^n, 1/n] -- dimension is nd
## theta^i are from the above function possible_theta, theta^i is of (d-1) dimension

def global_thetas():
    n_global_theta = n_local_theta**n
    th = list(itertools.product(local_theta, repeat=n))
    th = [list(th[i]) for i in range(n_global_theta)]
    t=[]
    for i in range(len(th)):
        tt=[]
        for j in range(len(th[i])):
            tt = tt+th[i][j]
        t.append(tt)
    return t


## P_{theta}(s_prime|s,a) = < theta, global features >; theta is from global_thetas

def prob(s,a,s_prime,theta):
    p = np.inner(theta, global_features_prob(s,a,s_prime))
    return p

## returns the probability vector: given a state and action, 
## each component will represent the prob of going to next state.

def prob_sa(s,a,theta):
    pr=[]
    for i in global_state:
        pr.append(prob(s, a, i, theta))
#    print("probs",sum(pr))
    return pr

## Given a state and an action pair this function returns 
## congestion seen by each agent

def congestion(s,a):
    indices = [i for i, x in enumerate(s) if x == "sinit"]
    # print(indices)
    congestion = [0 for i in range(n)]
    for i in indices:
        counter = 0
        for j in indices:
            if a[i] == a[j]:
                counter = counter+1
        congestion[i] = counter
    return congestion


## Returns the local cost of each agent for given state and action 
## c(s,a) = [K^1(s,a)* sum_j 1_{a^1 = a^j}, K^2(s,a)* sum_j 1_{a^2 = a^j}, dots, K^n(s,a)* sum_j 1_{a^n = a^j}]
## That is c(s,a) = [c^1(s,a), c^2(s,a), dots, c^n(s,a)]

def local_cost(s,a):
    pc= []
    for i in range(n):
        if s[i]!=local_state[-1]:
            pc.append(random.uniform(c_min,1))  ## replaced this by 
        else: 
            pc.append(0)
            # pc.append(5.5)
    con = congestion(s, a)
    c = [pc[i]*con[i] for i in range(n)]
    return c

## Given an agent, it return the cost \bar{c}^i(s,a)

def local_cost_agent(s,a,agent):
    lc = local_cost(s,a)
#    print(N)
    b = N.index(agent)
    return lc[b]

## This retursn 1/n \sum_{i} \bar{c}^i(s,a) for given state action pair (s,a)

def averag_global_cost(s,a):
    loc_cost = []
    for i in N:
        loc_cost.append(local_cost_agent(s,a,i))
    avg_global_cost = np.average(loc_cost)
    return avg_global_cost


## The cost features for given state, action pair is [psi(s^1,a^1), psi(s^2, a^2), \dots, psi(s^n, a^n)].
## The features for each agent are of dimension n. 
## If the state is sinit it is congestion, if it is g, then it is 0
## We also append [1,1,1,1] to the features -- they are elements of the adjacency matrix

def global_cost_feature(s, a):
    psi = congestion(s,a)
    return psi


## This function returns <psi(s,a), w> which is approximation for \bar{c}(s,a)
## If each agent wants to estimate the global cost function, they should use w^i in place of w.
## Note that w^i is also of the same dimension as w

def approx_global_cost(s,a,w):
    cf = global_cost_feature(s,a)
    # print("cf",cf)
    cost_approx = np.inner(cf, w)
    return cost_approx


## fetures for cost to go term; takes input 
## global state s; global action a; and the value fuction V 
## Note that V \in R^|n_global_state|
        
def feature_V(s,a,V):
    f = [0 for i in range(feature_dimension*n)]
    for i in range(len(global_state)):
        g = np.multiply(global_features_prob(s, a, global_state[i]),V[i])
#        print(g)
        f =  f + g
    return list(f)


def consensus(i,j):
    L=[[1/n for p in range(n)] for q in range(n)]
    return L[i][j]

## This function returns the inner product < phi(s_prime| s, a), theta > for any given state, action, next state and theta
    
def feat_theta_inner(s,a,s_prime, theta):
    f = global_features_prob(s,a,s_prime)
    # print(f)
    b = np.inner(f,theta)
    # print(b)
    return b

## This returns  \sum_{s_prime} <phi(s_prime | s,a), theta> for any given state and action pair

def feat_sum(s,a,theta):
    su = 0
    for i in range(n_global_state):
        su = su + feat_theta_inner(s,a, global_state[i], theta)
    return su



def cal_B():
    temp = []
    for i in range(n_global_state):
        for j in range(n_global_action):
            for th in global_theta:
                if feat_sum(global_state[i],global_action[j],th) ==1:
                    temp.append(th)
                    
    set_B = []
    for i in range(n_global_state):
        for j in range(n_global_action):
            for k in range(n_global_state):
                for th in temp:
                    if feat_theta_inner(global_state[i],global_action[j],global_state[k], th) >=0 and feat_theta_inner(global_state[i],global_action[j],global_state[k], th)<=1:
                        set_B.append(th)
    return set_B


## This function picks up the common elements from any two list 
    
def common_member(list_1, list_2):
    result = [i for i in list_1 if i in list_2]
    return result


## Given the confidence set, it returns \mathcal C \intersection \mathcal B

def con_inter_cal_B(c_set):
    a =  common_member(c_set, c_B)
    return a


## This takes input a global state, global action, V, and a set con
## It outputs min_{theta \in con} <phi_V(s,a), theta>
## Note that this set con will be obtained from the main algorithm
## it has to be C intersection B


def st_with_min(s,a,V,con):
    ctg =[]
    common_theta = con_inter_cal_B(con)
    for theta in common_theta:
        ctg.append(np.inner(theta, feature_V(s,a,V)))
    ct = min(ctg)
    return ct

## This is EVI sub routine, we can use the w_compliment here as the input to this subroutine, and those will than be used
## for the purpose of the consensus update.

def EVI(con_set, epsilon, q, S, N, w):
#    print(len(con_set))
    Q=[]
    V_current=[]
    V_previous=[]
    for j in range(len(S)):
        Q.append([[0 for ac in range(n_global_action)] for st in range(n_global_state)])
        V_previous.append([math.inf for st in range(n_global_state)])
        V_current.append([0 for st in range(n_global_state)])
    for j in range(len(S)):
        cm = con_inter_cal_B(con_set[j])
        if len(cm)!=0:
            while LA.norm(np.subtract(V_current[j], V_previous[j])) >= epsilon[j]:
                V_previous[j] = V_current[j]
                for st in range(n_global_state):
                    for ac in range(n_global_action):
                        gs = global_state[st]
                        ga = global_action[ac]        
                        cost_approx = approx_global_cost(gs , ga, w[j])
                  ## right now this is big number, may be because of w?
                        Q[j][st][ac] =  cost_approx + (1-q)* st_with_min(gs, ga, V_current[j], con_set[j])
                V_current[j] = [min(Q[j][st]) for st in range(n_global_state)]
    return Q, V_current


## intermidate functon to be used to get phi_V phi_V^T for a given state and action pair and the value function    
    
def phi_phi_top(s,a,V):
    h = [feature_V(s, a, V)]
    matrix = np.dot(np.transpose(h), h)
    return matrix


## An intermidiate tetrm used in the confidence set, called the confidence radius.
## Changes with t, so t is input.

def beta_t(t):
    a = t**2 + ((t**3 * B**2)/float(lamda))
    b = float(4*a/delta)
    c = math.log(b)
    e = (c*n*feature_dimension)**0.5
    beta = B*e + (lamda*n*feature_dimension)**0.5
    return beta

## Confidence set, that takes Sigma, theta_hat, and t as input 
## and outputs a confidence radius

def confidence_set(sigma, theta_hat, t):
    cs = []
    for i in range(len(global_theta)):
        global_theta[i] = list(global_theta[i])        
        # all_theta[i].append(1)
        diff = np.subtract(global_theta[i],theta_hat)
        root_sigma = sqrtm(sigma) 
        b = np.dot(root_sigma, diff)
        c = LA.norm(b)
        if c <= beta_t(t):
            cs.append(global_theta[i])
    return cs

def min_max(Q, agent_index):
    indeces = [[] for i in range(n_local_actions)]
    for k in range(n_local_actions):
        for i in range(n_global_action):
            if global_action[i][agent_index] == local_action[k]:
                indeces[k].append(i)
#    print("agent",agent_index)
#    print("ind",indeces)
    max_set = [[] for i in range(n_local_actions)]
    for i in range(n_local_actions):
        for j in indeces[i]:
            max_set[i].append(Q[j])
#    print("max",max_set)
    ma=[]
    for i in range(len(max_set)):
        ma.append(max(max_set[i]))
#    print("ma",ma)
    min_max = min(ma)
    res = [i for i, j in enumerate(ma) if j == min_max]
#    print(res)
    return res

def min_min(Q, agent_index):
    indeces = [[] for i in range(n_local_actions)]
    for k in range(n_local_actions):
        for i in range(n_global_action):
            if global_action[i][agent_index] == local_action[k]:
                indeces[k].append(i)
#    print("agent",agent_index)
#    print("ind",indeces)
    min_set = [[] for i in range(n_local_actions)]
    for i in range(n_local_actions):
        for j in indeces[i]:
            min_set[i].append(Q[j])
#    print("max",max_set)
    ma=[]
    for i in range(len(min_set)):
        ma.append(max(min_set[i]))
#    print("ma",ma)
    min_min = min(ma)
    res = [i for i, j in enumerate(ma) if j == min_min]
#    print(res)
    return res



## returns all the index of the actions for which the Q(s) of a given state s is minimum
    
def min_Q_index(s,Q_s):
    min_value = min(Q_s)
#    print(min_value)
    res = [i for i, j in enumerate(Q_s) if j == min_value]
    return res


## Now the main algorithm
    
def MLEVIS(lamda, rho, K):
    t = 1
    j = [0 for i in range(n)]
    t_j = [0 for i in range(n)]
    eps_j = [0 for i in range(n)]
    Sigma_prev = [lamda* np.identity(n*feature_dimension) for i in range(n)]
#    print("sigma_prev",Sigma_prev)
    Sigma_t_j = [lamda* np.identity(n*feature_dimension) for i in range(n)]
    Sigma_inverse = [0* np.identity(n*feature_dimension) for i in range(n)]
    Sigma_new = Sigma_prev
    b = [[0 for i in range(n*feature_dimension)] for i in range(n)]
    Q = [[] for i in range(n)]
    V = [[] for i in range(n)]
    w = []
    w_tilde = []
    for i in range(n):
        for st in range(n_global_state):
            if global_state[st] != global_state[-1]:
                Q[i].append([1 for ac in range(n_global_action)])
                V[i].append(1)
            else: 
                Q[i].append([0 for ac in range(n_global_action)])
                V[i].append(0)
        w.append([0 for d in range(n)])
        w_tilde.append([0 for d in range(n)])
#    print(Q)
    co = []
    for k in range(1, K+1):
        print("episode", k)
#        with open("episodes.txt", "w") as file:
#            file.writelines(str("Episodes="))
#            file.writelines(str(k))
#            file.writelines(str("\n"))
        s_current = global_state[0]
#        s_current_index = global_state.index(s_current)
#        print(s_current_index)
        c_k=[]
        while s_current != global_state[-1]:   ## This need to be incorporated
            s_current_index = global_state.index(s_current)
            min_a_index = []
            for i in range(n):
                if s_current[i] !="g":
                    min_a_index.append(np.random.choice(min_max(Q[i][s_current_index], i)))   # to be changed according to the algorithm min max
                else:
                    min_a_index.append(n_local_actions-1)
#            min_a_index_set = min_Q_index(s_current_index, Q[s_current_index])
#            print(min_a_index_set)
#            min_a_index = random.choice(min_a_index_set)
#            print("min_set",min_a_index)
            a_current = []
            for i in range(n):
                a_current.append(local_action[min_a_index[i]])
##            print(a_current)
#            a_current = global_action[min_a_index]
            # ck_temp = [approx_global_cost(s_current, a_current, w[i]) for i in range(n)]
            c_k.append(averag_global_cost(s_current, a_current))
            
            # c_k.append(np.mean(ck_temp))
    #                print(prob_sa(global_state, a_current, global_theta[-1]))
            s_next = choices(global_state,prob_sa(global_state, a_current, global_theta[-1]))        
            s_next = s_next[0]
            s_next_index = global_state.index(s_next)
            step_size = 1/t
            cost_approx = approx_global_cost(s_current, a_current, w[i]) 
            appr_diff = np.subtract(local_cost_agent(s_current, a_current, N[i]), cost_approx)
            feat_local_cost_prod = np.dot(appr_diff, global_cost_feature(s_current, a_current))
            for i in range(n):
                w_tilde[i] = np.add(w[i], np.dot(step_size, feat_local_cost_prod))
            for i in range(n):
#                print(i)
                Sigma_new[i] = np.add(Sigma_prev[i], phi_phi_top(s_current, a_current, V[i]))
#                print(Sigma_new[i])
                b[i] = np.add(b[i], np.multiply(feature_V(s_current, a_current, V[i]), V[i][s_next_index]))
#            print("sn",Sigma_new)
            # print(b)
            S = []
            S_c = []
            for i in range(n):
                if np.linalg.det(Sigma_new[i]) >= 2*np.linalg.det(Sigma_t_j[i]) or t >=2*t_j[i]:
                    S.append(N[i])
                else:
                    S_c.append(N[i])
#            print("S",S)
#            print(S_c)
            
            if len(S) != 0:
                cof_set = []
                for i in range(len(S)):
                    j[i] = j[i] + 1
                    t_j[i] = t
                    eps_j[i] = 1/t_j[i]
                    Sigma_inverse[i] = np.linalg.inv(Sigma_new[i])
                    theta_hat_j = np.dot(Sigma_inverse[i], b[i])
                    theta_hat_j = list(theta_hat_j)
                    cof_set.append(confidence_set(Sigma_t_j[i], theta_hat_j, t_j[i]))
                
                EV = EVI(cof_set, eps_j, 1/t, S, N, w)
#                print(EV[0])
                for i in range(len(S)):
                    Q[i] = EV[0][i]
                    V[i] = EV[1][i]
                    Sigma_t_j[i] = Sigma_new[i]
            # print(Q)
            w_temp = [[0 for i in range(n)] for i in range(n)]
            for k in range(n):
                w[i] = np.add(w_temp[i], np.dot(consensus(i,k), w_tilde[k]))
            t = t+1
#            print(t)
            Sigma_prev = Sigma_new
            s_current = s_next
        co.append(c_k)
    return co
    


n=5
N = [(i+1) for i in  range(n)]
feature_dimension = 2
delta = 0.2
Delta = 0.1
# q = 0.1
lamda = 0.1
rho = 0
K=7000

c_min = 0.1

B=2.15


local_state = local_states()
n_local_states = len(local_state)
# print("ls",local_state)

global_state = global_states()
n_global_state = len(global_state)
#print("gs",global_state)

local_action = local_actions()
n_local_actions = len(local_action)
#print("la",n_local_actions)
#
global_action = global_actions()
n_global_action = len(global_action)
#print("ga",global_action)
#print(global_action)
#print(n_global_action)


local_theta = local_thetas()
n_local_theta = len(local_theta)
#print(local_theta)

global_theta = global_thetas()
n_global_theta = len(global_theta)
# print(n_global_theta)

c_B = cal_B()
# print(len(c_B))

V_star = B

#ev = EVI(global_theta, [0.1 for i in range(2)], 0.1, [1,2], N)
#print(ev[0])

ml = MLEVIS(lamda, rho, K)
#print(ml)

def cost_k():
    co_k=[]
    for i in range(K):
        co_k.append(sum(ml[i]) - V_star)
    # with open("cost_each_episode.py", "w") as file:
    #     file.writelines(str("co_k="))
    #     file.writelines(str(co_k))
    #     file.writelines(str("\n"))
    return co_k

cost_ep_k_minus_v_star = cost_k()
# print("df",cost_ep_k_minus_v_star)

def cum_reg():
    cu_reg = []
    for i in range(K):
        cu_reg.append(sum(cost_ep_k_minus_v_star[:i+1]))
    # with open("cum_reg.py", "w") as file:
    #     file.writelines(str("cum_reg="))
    #     file.writelines(str(cu_reg))
    #     file.writelines(str("\n"))
    # print(cu_reg)
    return cu_reg


cumulative_reg = cum_reg()
# print(cumulative_reg)

def reg_by_k():
    avg_reg_by_k = []
    for i in range(K):
        avg_reg_by_k.append(sum(cost_ep_k_minus_v_star[:i+1])/(i+1))
    # with open("avg_reg.py", "w") as file:
    #     file.writelines(str("avg_reg="))
    #     file.writelines(str(avg_reg_by_k))
    #     file.writelines(str("\n"))
    return avg_reg_by_k
    
av_re_by_k = reg_by_k()


with open("data_r1.py", "w") as file:
    file.writelines(str("cum_reg_r1="))
    file.writelines(str(cumulative_reg))
    file.writelines(str("\n"))
    
    file.writelines(str("avg_reg_r1="))
    file.writelines(str(av_re_by_k))
    file.writelines(str("\n"))


print("last_cr", cumulative_reg[-1])
print("last_avg_reg", av_re_by_k[-1])

episode = [math.sqrt(i) for i in range(K)]
plt.plot(episode,av_re_by_k)
plt.show()

plt.plot(episode,cumulative_reg)
plt.show()




    
    
