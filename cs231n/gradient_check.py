import numpy as np
from random import randrange

def eval_numerical_gradient(f, x):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  if (__debug__):
    print 'Inside eval_numerical_gradient(f, x==', x, ')...'
  #endif debugging
  
  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    x[ix] += h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] -= h # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    print ix, grad[ix]
    it.iternext() # step to next dimension

  return grad

def grad_check_sparse(f, x, analytic_grad, num_checks):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """
  h = 1e-5
  if (__debug__):
      print 'Inside grad_check_sparse(f, 2nd arg=x, ...)...:'
      print '  analytic_grad:  shape: ', analytic_grad.shape
      #print '  analytic_grad[9]: ', analytic_grad[9]
  #endif debugging
  x.shape
  if (__debug__):
    # Examine what happens at a particular element:
    ix = tuple([5, 974])
    print '                   ix == ', ix
    print '                 x[ix]== ', x[ix]
    
    x[ix] += h # increment by h
    print '              x[ix]+h == ', x[ix]
    xph = x[ix]
    print '  about to call f(x)==svn_loss_naive(W, ...) after tweaking W[5, 974] ...'
    fxph = f(x) # evaluate f(x + h)
    x[ix] -= 2 * h # increment by h
    print '            f(x[ix]+h)==', fxph
    print '              x[ix-h] == ', x[ix]
    xmh = x[ix]
    
    fxmh = f(x) # evaluate f(x - h)
    x[ix] += h # reset
    print '            f(x[ix]-h)==', fxmh
    print '     f(ix+h) - f(ix-h)==', (fxph - fxmh)
    print '(x[ix]+h) - (x[ix]-h) ==', (xph - xmh)
    print ' 2*h = ', 2*h
    
    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
    #print '... returning early from grad_check_sparse()'
    #return
  #endif debugging with element [2,2]
        
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])
    if (__debug__):
      print '   ix == ', ix
      print ' x[ix]== ', x[ix]
      #print '  analytic_grad[', ix[0], ']: ', analytic_grad[ix[0]]
      #print '  analytic_grad[', ix, ']: ', analytic_grad[ix]
    #endif debugging
    
    x[ix] += h # increment by h
    if (__debug__):
      print ' x[ix]+h == ', x[ix]
      xph = x[ix]
    #endif debugging
    if (__debug__):
        print ' about to call f(x)==svn_loss_naive(W,...) after tweaking W[', ix,']'
    #endif debugging
    fxph = f(x) # evaluate f(x + h)
        
    x[ix] -= 2 * h # increment by h
    if (__debug__):
      print ' f(x[ix]+h)==', fxph
      print '   x[ix-h] == ', x[ix]
      xmh = x[ix]
    #endif debugging
    
    fxmh = f(x) # evaluate f(x - h)
    x[ix] += h # reset
    if (__debug__):
      print ' f(x[ix]-h)==', fxmh
      print 'f(ix+h) - f(ix-h)==', (fxph - fxmh)
      print '(x[ix]+h) - (x[ix]-h) ==', (xph - xmh)
      print ' 2*h = ', 2*h
    #endif debugging
    
    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
  #endfor i

#end def grad_check_sparse( )

def grad_check_complete(f, W, analytic_grad, X_train, y_train):
  """
  return complete grids of:
      the derivative of f at all points of W
      the differences between analytic_grad and the derivative of f()
      the relative error
  Instead of recomputing the loss from scratch with every iteration,
    simply compute the CHANGE in loss due to the tweaks.  Assumes
    only the margins involved at the point being tweaked contribute.
  """
  h = 1e-5
  if (__debug__):
      print 'Inside grad_check_complete(f, 2nd arg=W, ...)...:'
  #endif debugging
  
  grad_numerical = np.zeros(W.shape)
  differences = np.zeros(W.shape)
  rel_errors = np.zeros(W.shape)
  delta = 1.0
  
  for i in xrange(W.shape[0]):
      if (__debug__):
          print 'Outer loop variable i==', i
      #endif debugging
      
      for j in xrange(W.shape[1]):
        # Capture the margin of loss computed for this point [i, j]
        if (j == y_train[i]): # correct class
          # The loss margin for the correct class is zero.
          margin_at_plus_h = 0.0
          margin_at_minus_h = 0.0
        else:
          # j != y_train[i] (not looking at correct class)
          # Now, compute the loss value at this point when the weight at this
          #   point is tweaked up & down.
          # After tweaking W, recompute scores = W.dot(X_train[:, i])
          #  
          original_weight = W[i, j]
          W[i, j] += h
          scores_at_plus_h = W.dot(X_train[:, i])  # A vector of scores for each class
          correct_class_score_at_plus_h = scores_at_plus_h[y_train[i]]
          W[i, j] -= 2*h
          scores_at_minus_h = W.dot(X_train[:, i])
          correct_class_score_at_minus_h = scores_at_minus_h[y_train[i]]
          # Now restore original weight value
          W[i, j] = original_weight
          margin_at_plus_h = scores_at_plus_h[i] - correct_class_score_at_plus_h + delta
          margin_at_minus_h = scores_at_minus_h[i] - correct_class_score_at_plus_h + delta
          # If margin is negative, set it to zero.
          margin_at_plus_h = (margin_at_plus_h >= 0)*margin_at_plus_h
          margin_at_minus_h = (margin_at_minus_h >= 0)*margin_at_minus_h
        #endif correct class -- else
        grad_numerical[i, j] = (margin_at_plus_h - margin_at_minus_h) / (2 * h)
        grad_analytic = analytic_grad[i, j]
        differences[i, j] = (grad_numerical[i, j] - grad_analytic)
        rel_errors[i, j] = abs(grad_numerical[i, j] - grad_analytic) / (abs(grad_numerical[i, j]) + abs(grad_analytic))
          
      #endfor j
  #endfor i
  
  return grad_numerical, differences, rel_errors
#end def grad_check_complete( )
