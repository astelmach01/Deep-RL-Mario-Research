
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
def get_20news():
    cats = ['soc.religion.christian', 'alt.atheism', 'comp.sys.mac.hardware', 'comp.graphics', 'comp.windows.x']
    train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers', 'quotes'))
    return train, test

train, test = get_20news()

topics = ['religion', 'computers']
map_target_name_to_class = {'soc.religion.christian':'religion', 
                            'alt.atheism':'religion', 
                            'comp.sys.mac.hardware':'computers', 
                            'comp.graphics':'computers', 
                            'comp.windows.x':'computers'}
y_topic_train = [map_target_name_to_class[train.target_names[i]] for i in train.target]
y_topic_test = [map_target_name_to_class[test.target_names[i]] for i in test.target]

## Array of 0/1 - religion = 0 and computers = 1 
y_train = np.array([topics.index(k) for k in y_topic_train])
y_test = np.array([topics.index(k) for k in y_topic_test])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000, min_df=5, binary=True, 
                             stop_words="english")
X = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)
X.shape, X_test.shape

n, d = X.shape
k = 2

'''
We saw in lecture that estimating Naive Bayes parameters in an MLE framework is trivial due to the factorization, which is one reason the model is so popular. But in the spirit of reinforcing that the model is not the optimization method, we will take a circuitous route: Here you are asked to estimate Naive Bayes model parameters via stochastic gradient descent (SGD).
There are several reasons why you would not do this in practice --- first and foremost because the MLE estimate is so straightforward. However, as an exercise this is valuable in highlighting that distinction between choice of optimization and model, and hopefully this will also provide some hands-on experience working with gradient descent on likelihoods, a very common general strategy in ML.
Note that in reality we need to perform constrained optimization here, since and values must be valid probabilities i.e., between 0 and 1. We ask you to enforce this constraint, such that you ensure parameters constitute valid probabilities; you may do this in an ad-hoc way.
As a reminder, the model here is as follows.

Your task is to find estimates that maximize this liklihood, given data (X, y) using SGD. Assume y_i is in 0,1, i.e., u is the Bernoulli parameter, and phi e R2 x V represents the vocabulary distribution for each topic
'''


def mu_ll(mu):
  ''' Return the log likelihood w/r/t \mu (class prevalence) for NB '''
  mu = np.exp(mu)
  return sum([y * np.log(mu) + (1-y) * np.log(1-mu) for y in y_train])

mu_range = list(np.linspace(.1, .9, 20))
mu_lls   = [mu_ll(np.log(mu_)) for mu_ in mu_range]
best_mu_idx = np.argmax(np.array(mu_lls))
print("Best \mu is: {:.2f}".format(mu_range[best_mu_idx]))
plt.plot(mu_range, mu_lls);
print(y_train[y_train==1].shape[0] / y_train.shape[0])


'''
Write down the partials for  u  and  phi  parameters; show your work.
Hint: We are assuming the two class case - you might want to think of / write these 

Since we are minimizing the loss in gradient descent (not ascent), we take the negative of this equation, resulting in the following partial derivatives:
 

'''


def SGD_estimate_mu(epochs=500, alpha=1e-6, verbose=True):
  # Initialize \mu to uniform probability over classes
  
  mu = 0.5

  # Keep track of \mu estimates and corresponding likelihoods 
  likelihoods, mus = np.zeros(epochs), np.zeros(epochs)

  def partial_mu(mu):
      return sum([-y/mu + (1-y) / (1-mu) for y in y_train])

  # implement sgd
  for epoch in range(epochs):
      #print(partial_mu(mu))
      mu = mu - alpha * partial_mu(mu)
      mus[epoch] = mu
      likelihoods[epoch] = mu_ll(np.log(mu))

      if verbose:
        print("Epoch: {:3d}  mu: {:.3f}  likelihood: {:.3f}".format(epoch, mu, likelihoods[epoch]))

      # break if likelihood has converged
      if epoch > 1 and np.abs(likelihoods[epoch] - likelihoods[epoch-1]) < 1e-5:
          break
      
  
  return np.array(mus), np.array(likelihoods)

mus, mu_likelihoods = SGD_estimate_mu()

plt.plot(mus, mu_likelihoods)

best_mu_idx = np.argmax(mu_likelihoods)
print(mus[best_mu_idx])


'''
And now for  phi . Note that you must return valid probabilities.
Hint: You probably want to start by writing a funciton to compute the log likelihood of observed words given  phi so you can visualize the likelihood over training.

'''

def ll_phi(phi_):
  '''
  Return the likelihood of observed instances (*given* labels) under parameter
  estimates phi. 
  '''
  
  # take the log likelihood of the observed words given the parameter estimates
  """
    total = 0
  for i in range(n):
    for j in range(d):
      if X[i,j] == 1:
        total += np.log(phi_[1,j])
  return total
  
  """
  # use numpy to speed up computation
  return np.sum(np.log(phi_[1, X[:,:] == 1]))


  

def SGD_estimate_phi(epochs=80, alpha=1e-6, 
                     verbose=True, ll_every=2) :
  '''
  Return the *best* \phi estimate observed (w/r/t loss) and an array of 
  likelihoods calculated during training. (Does not return all \phi estimates
  because it would be quite large.)
  '''
  # Initialize \phi to uniform probabilities across tokens for each class
  v_size = X.shape[1]
  # phi = np.ones((2, v_size)) * np.log(1/v_size)
  phi = np.ones((2, v_size)) * 1/v_size
  best_phi = None
  phi_lls = np.zeros(epochs)

  # We only hold on to the "best" \phi estimate for return
  best_ll = -np.inf

  def partial_phi(phi_, c):
      return  sum([X[i: ] * (y_train[i] == c) for i in range(n)])
      
  print(phi)
  # implement sgd
  for epoch in range(epochs):
    #  phi = phi - alpha * np.array([[y * np.log(phi[0,i]) + (1-y) * np.log(1-phi[0,i]) for i in range(v_size)] for y in y_train])
      for c in range(2):
          phi[c] = phi[c] - alpha * partial_phi(phi[c], c)
          
      phi = phi / phi.sum(axis=1, keepdims=True)
      
      l = ll_phi(phi)
      phi_lls[epoch] = l

      if l > best_ll:
        best_phi = phi.copy()
        best_ll = l
        
      if verbose and  epoch % ll_every == 0:
          print("Epoch: {}\tLoss: {:.5f}".format(epoch, phi_lls[epoch]))
  
  
  return best_phi, phi_lls

best_phi, phi_lls = SGD_estimate_phi(epochs=100, alpha=1e-7)