import cPickle as pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  ## DEBUGGING:
  print 'Inside load_CIFAR_batch(', filename, ') ...'
  
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    ## Applying this type conversion to the entire matrix generates a memory
    ##  error on my 32-Gbyte system (with 32-bit Python).  
    ##  Doing it one element at a time worked.  Perhaps that's not necessary
    ##  after changing to 64-bit Python.
    
    # print 'Converting X dtype to float ...'
    # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
    # X[:] = X[:].astype("float")

    Y = np.array(Y)
    
#    print 'X shape: ', X.shape
#    print 'X dtype: ', X.dtype
    
#    print 'Y shape: ', Y.shape
#    print 'Y dtype: ', Y.dtype
    
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
