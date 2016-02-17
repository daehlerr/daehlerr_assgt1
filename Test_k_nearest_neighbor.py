#### Cell 1
# Run some setup code for this notebook.
print 'Inside Test_k_nearest_neighbor.py ...'

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import sys
import psutil

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
### %matplotlib inline  ** Doesn't work in ordinary Python

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
### %load_ext autoreload
### %autoreload 2

#### Cell 2

print 'About to load raw CIFAR-10 data ...'
# Load the raw CIFAR-10 data.
# cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
### !dir
### !dir cs231n
### !dir cs231n\datasets
cifar10_dir = r'cs231n\datasets\cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 
print '  Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print '      Test data shape: ', X_test.shape
print '    Test labels shape: ', y_test.shape
print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
print
#### Cell 3 Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#### Cell 4
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
## num_training = 500

mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
## num_test = 50

mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

#### Cell 5
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

## DEBUGGING:
print 'After sampling, '
print '    Test shape = ', X_test.shape
print 'Training shape = ', X_train.shape
print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
print

# print X_train.shape, X_test.shape

#### Cell 6
from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

#### Cell 7
# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print 'After two-loop classifier, dists shape = ', dists.shape
print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
print
# print dists.shape

#### Cell 8
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')

#### Cell 9
# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

#### Cell 10
# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'
print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
print
  
#### Cell 11
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'
print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
print
  
#### Cell 12
# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation

#### Cell 13
print
print 'Done with Test_k_nearest_neighbor.py'

#### Cell 14

#### Cell 15

#### Cell 16

#### Cell 17

#### Cell 18

#### Cell 19

#### Cell 20

#### Cell 21

#### Cell 22

