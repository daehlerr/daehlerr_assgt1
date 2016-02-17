import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  delta = 1.0 
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  
  print_flag = True
  
  if (__debug__):
      print 'Inside svm_loss_naive( )...'
      print '  W shape: ', W.shape
      print '  X shape: ', X.shape
      print '  y shape: ', y.shape
      print '  reg:     ', reg  
      print ' W[5, 974]== ', W[5,974]
  #endif debugging
  
  
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    if (__debug__ and (i==0)):
        print 'scores.shape: ', scores.shape
    #endif debugging
    
    # The score of image[i] for each class[j] (row of w).
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      """
      if j == y[i]:
        continue
      # margin = scores[j] - correct_class_score + 1 # note delta = 1
      margin = scores[j] - correct_class_score + delta
      """
      if (False and __debug__ and (i==1) and (j==1) ):
          print 'shape of (scores[:] - correct_class_score + delta): ', (scores[:] - correct_class_score + delta).shape
          print 'Question: Is this huge or small?'
      #endif debugging
      
      # The dimensions of W and dW are [class][pixel].  
      #  The dimensions of X are [pixel][image].
      #  For each pixel[:], we need to update the value of dW[class][pixel]
      #  by accumulating the contribution for each class [j] by each image [i].
      if (j == y[i]): # correct class
        # The loss margin for the correct class is zero, so no change in loss.
        if (__debug__ and print_flag):
            print 'Shape of np.sum( (scores[:] - correct_class_score + delta)>0)*X[:, i] ', (np.sum( (scores[:] - correct_class_score + delta)>0)*X[:, i]).shape
            print_flag = False  # So we don't repeat this message.
        #endif debugging
        
        if (__debug__ and (i<3)):
          print 'j==y[i==', i, ']==', y[i]
          print 'np.sum( (scores[:] - correct_class_score + delta)>0): ', np.sum( (scores[:] - correct_class_score + delta)>0)
          print '    ( (scores[j] - correct_class_score + delta) > 0): ', ( (scores[j] - correct_class_score + delta) > 0)
        #endif debugging
        
        """ 17 Feb 2016:  Andrew Barbarello doesn't have this part:
        dW[j, :] -= np.sum( (scores[:] - correct_class_score + delta)>0)*X[:, i]
        # But don't count the term for scores[correct class], if it contributed:
        dW[j, :] += ( (scores[j] - correct_class_score + delta) > 0)*X[:, i]
        """
      else:
        # j != y[i] (not looking at correct class)
        # margin = scores[j] - correct_class_score + 1 # note delta = 1
        margin = scores[j] - correct_class_score + delta
        if margin > 0:
          loss += margin
          dW[j, :] -= X[:, i]
          dW[y[i], :] -= X[:, i]
            # Andrew Barbarello adds this line.  Note that j is implicit in margin.
        #endif margin > 0

        if (__debug__ and (i<3)):
          print 'j==', j, ' !=y[i==', i, ']==', y[i]
          print 'margin == scores[j] - correct_class_score + delta: ', scores[j] - correct_class_score + delta
          print ' ( (scores[j] - correct_class_score + delta) > 0): ', ( (scores[j] - correct_class_score + delta) > 0)
          print ' The latter should be true only if margin > 0'
        #endif debugging
        
      #endif correct class -- else
    #endfor j in xrange(num_classes)
  #endfor i
  
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  dW += reg * W
    # Andrew Barbarello adds this line.

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  if (__debug__):
    print '     loss==', loss
    print '... leaving svm_loss_naive( )'
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass

  num_classes = W.shape[0]
  num_train = X.shape[1]
  delta = 1.0
    
  # Vectorize scores = W.dot(X[:, i])
  scores = W.dot(X[:])
  if (__debug__):
    #  print ' vectorized scores = W.dot(X[:]) shape: ', scores.shape
    pass
  #endif debugging
  i_range = xrange(num_train)
  j_range = xrange(num_classes)

  # Vectorize correct_class_score = scores[y[i]]
  correct_class_score = scores[y, i_range]
  #print 'Vectorized correct_class_score shape: ', correct_class_score.shape
  #print 'correct_class_score: ', correct_class_score
  
  # Vectorize margin = scores[j] - correct_class_score + delta
  #   Do it in stages.  First, compute values for all the elements
  #   using the formula for non-correct elements.  Then fill in
  #   zeros for the correct elements.
  margin = scores - correct_class_score + delta
  margin[y, i_range] = 0.0
  
  #  print 'Vectorized raw margin shape: ', margin.shape
  #print margin
  margin = (margin > 0)*margin
  # This applies max(margin, 0) to each element.
  # print 'Vectorized capped margin shape: ', margin.shape
  #print margin
    
  #loss = margin.sum(axis=0)
  loss = (margin.sum()/num_train) + 0.5 * reg * np.sum(W*W)
  # print 'Vectorized loss shape: ', loss.shape
  # print '                value: ', loss
  #print loss
  
  # Gradient:  
  #  For non-correct classification (non-vectorized form):
  #   dW[j, :] = -sum( (scores[j] - scores[y[i]] + delta) > 0)*X[:, i]
  #     (":" indicates pixel index)
  #     That is, count the number of score elements for which the 
  #     condition is true, then multiply that by X[:, i] (then subtract).
  #     Note that we've already computed (scores - correct_class_score + delta)
  #     with max(0, xxx) as margin[][].
  #  Vectorized form:
  # dW[:] = -(np.sum((margin >0).reshape(-1,1), axis=0))*(X[:].T)
  
  dW = -(margin >0).dot(X.T)
  dW /= num_train
  # print 'dW shape: ', dW.shape
  #print dW
  #print
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
