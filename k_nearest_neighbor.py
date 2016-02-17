import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    ## DEBUGGING: 
    print 'Inside compute_distances_two_loops() ...'
    print '     Shape of test images: ', X.shape
    print ' Shape of training images: ', X_train.shape
    return dists
    
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
        Train_vector = self.X_train[j][:]
        Test_vector = X[i][:]
        square_diff = np.sum((Train_vector - Test_vector)**2)
        dists[i,j] = np.sqrt(square_diff)
        # pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    ## DEBUGGING: 
    print 'Inside compute_distances_one_loop() ...'
    print '     Shape of test images: ', X.shape
    print ' Shape of training images: ', X_train.shape
    return dists
    
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i, :] = np.sqrt(np.sum((X[i][:] - self.X_train[:])**2))
      # pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    
    ## DEBUGGING: 
    print 'Inside compute_distances_no_loops() ...'
    print '     Shape of test images: ', X.shape
    print ' Shape of training images: ', X_train.shape
    return dists
    
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    
    # pass
    # Goal: To compute l2 (ell-two) norm of differences between pictures in
    #  a test collection and pictures in a training collection.
    #  ASSUME: Each image is a separate, single row of length
    #  width*height
    
    nbr_pixels_per_image = X.shape[1]
    
    ## DEBUGGING: 
    print 'Inside compute_distances_no_loops() ..."
    print '     Number of test images: ', num_test
    print ' Number of training images: ', num_train
    print 'Number of pixels per image: ', nbr_pixels_per_image
    return dists
    
    ##  We must compare only matching pixels:
    ##      Test_Image[i]pixel[x,y]
    ##  with Training_Image[j]pixel[x,y]  (The same x,y in both.)
    ##  
    ## Try reshaping Test as (nbr_test_images, 1, nbr_pixes_per_image)
    ##  then broadcast against Train.  The result should be
    ##  (nbr_test_images, nbr_train_images, nbr_pixels_per_image),
    ##  which is promising.
    
    XX = X.reshape(num_test,1,nbr_pixels_per_image)

    ## Broadcast difference:    
    diff = XX - self.X_train
    
    ## Broadcast product of array with itself squares each element.
    diff2 = diff*diff
    
    ## Now take dot product of each image-diff-squared with
    ##  an image-length vector of all ones:
    one_v = np.ones(nbr_pixels_per_image)
    sums_of_squares = diff2.dot(one_v)
  
    ## Then the ell-2 distance for each test/train pair is the square
    ##  root of the corresponding element of sums_of_squares:
    dists = np.sqrt(sums_of_squares)  
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # pass
      num_train = dists.shape[1]
      if (k > num_train):
          ## ERROR!  Bail out.
          print '*** ERROR: Inside predict_labels(self, dists, k)'
          print '***  number of neighbors k (', k, ') exceeds'
          print '***  number of training images (', num_train,').'
          print '*** ABORTING. ' 
          raise ValueError(
            'Value of k (%d) exceeds nbr of training images (%d).'
                      %  (k,                                num_train) )
      # endif number of neighbors (k) exceeds number of training images.

      if (k==1):
        # Since k is 1, we only need to find the closest match.  No need
        #  to do a full-blown sort.
        ## closest_y[0] = self.y_train[np.argmin(dists[i,:])]
        closest_y.append(self.y_train[np.argmin(dists[i,:])])
      else:
        # k > 1, so we should sort
        sorted_y = np.argsort(dists[i,:])
        for j in range(k):
            ## closest_y[j] = sorted_y[j]
            closest_y.append(sorted_y[j])
      # endif-else
      
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # pass
      if (k==1):
        # With only one to choose from, choose it!
        y_pred[i] = closest_y[0]
      else:
        # Create a histogram of closest_y[], then select
        #  the most frequent value.
        y_hist = np.histogram(closest_y, bins=range(num_train))
        # y_hist is an ordered pair of arrays.  The SECOND is the list
        #  of histogram bin margins, chosen to be the index values for 
        #  the training set.  The FIRST is the histogram, indicating the 
        #  frequency with with which each training image index value
        #  appears in closest_y[].
        #
        # If simply any most-frequent value would do, we could now
        #  grab y_hist[1][np.argmax(y_hist[0])], but we need to 
        #  look at all the labels that share the largest frequency.
        #
        # To examine the most frequent, we might sort the histogram by
        #  frequency (the first array), but that would break the
        #  alignment with the training set indices (the second array).
        #  Instead, let's hope k isn't too big and simply walk through
        #  the array closest_y.  We CAN use np.argmax() to grab the
        #  biggest value to begin with, and the length of its label.
        
        index_of_shortest_max_so_far = closest_y[np.argmax(y_hist[0])]
        shortest_label_so_far = self.y[index_of_shortest_max_so_far]
        shortest_len_so_far = len(shortest_label_so_far)
        nbr_with_max_freq = 1
        m = 1
        for n in range(len(y_hist[0])):
            if (y_hist[0][n] == nbr_with_max_freq):
                # Found another image with the same max frequency.
                m = m+1
                this_index = y_hist[1][n]
                this_label = self.y[this_index]
                this_length = len(this_label)
                if (this_length < shortest_len_so_far):
                    index_of_shortest_max_so_far = this_index
                    shortest_label_so_far = this_label
                    shortest_len_so_far = this_length
                #endif found a shorter one
            #endif found another with  max frequency.
        #endwhile walking through the histogram.

        y_pred[i] = index_of_shortest_max_so_far
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

