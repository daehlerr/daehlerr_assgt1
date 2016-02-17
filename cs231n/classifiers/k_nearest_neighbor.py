import numpy as np
import psutil

class KNearestNeighbor:
  """ a k classifier with L2 distance """

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
    self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if (__debug__):
        print "Inside KNearestNeighbor.train(self, X, Y) ..."
        print " self.y_train[0]==", self.y_train[0]
        print ' classes[self.y_train[0]]=="', self.classes[self.y_train[0]], '"'
    #endif debugging
    
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
    if (__debug__):
        print
        print 'Inside compute_distances_two_loops() ...'
        print '     Shape of test images: ', X.shape
        print ' Shape of training images: ', self.X_train.shape
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
   #endif debugging
    
    ## return dists
    
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
        if ( __debug__ and ((i==0) and (j==0)) ):
            print '    Shape of test vector = ', Test_vector.shape
            print 'Shape of training vector = ', Train_vector.shape
            print 'dists[0,0] = ', dists[i,j]
            print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
            print
        #endif 0,0

        # pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists
  ## end of compute_distances_two_loops(self, X)
  
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
    if (__debug__):
        print 
        print 'Inside compute_distances_one_loop() ...'
        print '     Shape of test images: ', X.shape
        print ' Shape of training images: ', self.X_train.shape
        nbr_pixels_in_image = X.shape[1]
        print '   Nbr of pixels in image: ', nbr_pixels_in_image
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
    #endif debugging
    
    # To sum the differences, we'll want to take a dot product with a vector
    #  with 1 for each pixel in an image.
    ones_vector = np.ones(nbr_pixels_in_image)
    
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      
      # My first attempt seems to grab the sum from ALL training images:
      # dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train[:])**2))
      #dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train[:])*(X[i] - self.X_train[:])))
      
      # Prepare to broadcast the current test image against each training image.
      Test_Image = X[i]
      Test_Image = Test_Image.reshape(1, nbr_pixels_in_image)
      
      # Broadcast the difference.  Each row of the result should be the pixel-by-pixel
      #  differences between the test image and each training image.
      Image_Diffs = Test_Image - self.X_train
 
      # Now square each pixel-difference by broadcast multiplication:
      Image_Diffs2 = Image_Diffs*Image_Diffs
           
      """
      ## DEBUGGING:
      if (__debug__ and (i==0) ):
        print
        print 'Shape of Image_Diffs2 for first test image: ', Image_Diffs2.shape
        print '  ==> This should be the same shape as the training array: ', self.X_train.shape
        print
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
      #endif i==0
      """
      
      # Now sum the squares for each image by matrix multiplication with ones_vector:
      Sums_of_squares = Image_Diffs2.dot(ones_vector)
      """
      if ( __debug__ and (i==0)):
        print 'Shape of Sums_of_squares: ', Sums_of_squares.shape
        print '  ==> Should be a column vector with one element for each training image.'
        print
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
      #endif i==0
      """
      # Now take the square root of each sum of squares:
      dists[i][:] = np.sqrt(Sums_of_squares)
    #endfor i
    """
    if (__debug__):   
        print 'Shape of dists = ', dists.shape
        print 'dists[0,0] = ', dists[0,0]
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
    #endif debugging
    """
      # pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists
  ## end compute_distances_one_loop(self, X)
  
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
    if (__debug__):
        print 
        print 'Inside compute_distances_no_loops() ...'
        print '     Shape of test images: ', X.shape
        print ' Shape of training images: ', self.X_train.shape
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
    #endif debugging
    ## return dists
    
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
    """
    if (__debug__):
        print 'Inside compute_distances_no_loops() ...'
        print '     Number of test images: ', num_test
        print ' Number of training images: ', num_train
        print 'Number of pixels per image: ', nbr_pixels_per_image
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
    #endif debugging
    """
    ## return dists
    
    ##  We must compare only matching pixels:
    ##      Test_Image[i]pixel[x,y]
    ##  with Training_Image[j]pixel[x,y]  (The same x,y in both.)
    ##  
    ## Try reshaping Test as (nbr_test_images, 1, nbr_pixes_per_image)
    ##  then broadcast against Train.  The result should be
    ##  (nbr_test_images, nbr_train_images, nbr_pixels_per_image),
    ##  which is promising.  (It generates the right results for
    ##  three test images against two training images.)
    
    #XX = X.reshape(num_test,1,nbr_pixels_per_image)
    XX = X
    """    
    ## DEBUGGING:
    if (__debug__):
        print '                     Test shape: ', XX.shape
        print '                 Training shape: ', self.X_train.shape
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
    #endif debugging
    """
    
    ## Broadcasting the arrays will multiply the amount of memory used by
    ##  a factor of (size of larger array)/(size of smaller array).
    ##  Do we have enough memory to accommodate that?
    ## Example:  If Test.shape==(TestRows, ImageSize) and Train.shape==(TrainRows,ImageSize),
    ##  then the broadcast will be with (TestRows,1,ImageSize)x(TrainRows,ImageSize)
    ##  ==> (TestRows, TrainRows, ImageSize), and the size will increase by a factor
    ##  of (TestRows * TrainRows)/(TestRows+TrainRows).  OOPS!  That won't work.
    ##
    ## Technique borrowed from Andrew Barbarello:
    ##  Notice that the el-2 norm of Vector1-Vector2 can be written as
    ##  sqrt( Vector1.dot(Vector1) -2*Vector1.dot(Vector2.T) + Vector2.dot(Vector2)
    ##  However, when "Vector1" and "Vector2" are actually arrays of vectors,
    ##  ".dot()" doesn't do what we want (it pulls in unwanted cross products
    ##  for Vector1.dot(Vector1) and for Vector2.dot(Vector2).
    ## We can accomplish what we want by combining broadcast (term-by-termm)
    ##  products and row-specific sums (np.sum(..., axis=1)).
    
    test2 = np.sum(XX*XX, axis=1)
    test2 = test2.reshape(-1,1) # prepare for broadcast sum below.
    
    #test2 = np.sum(XX**2, axis=1)
    train2 = np.sum(self.X_train*self.X_train, axis=1)
    #test_by_train = XX.dot(self.X_train.T).T
    test_by_train = self.X_train.dot(XX.T).T
    """
    if (__debug__):
        print '     Shape of test2 (sums of test pixels squared): ', test2.shape
        print 'Shape of train2 (same thing for training): ', train2.shape
        print '     Shape of test_by_train (cross-terms): ', test_by_train.shape
    #endif debugging
    """
     
    #sum_2 = test2 -2*test_by_train + train2
    sum_2a = test2 -2*test_by_train
    """
    if (__debug__):
        print 'Shape of sum_2a=test2 - 2*test_by_train: ', sum_2a.shape
    #endif debugging
    """
    
    #sum_2 = sum_2a + train2
    #  Python cannot broadcast sum_2a + train2 shapes: (5000,500)+(5000,)
    train2a = train2.reshape(1,-1)
    """
    if (__debug__):
        print 'Shape of train2a=train2.reshape(1,-1): ', train2a.shape
    #endif debugging
    """
    
    sum_2 = sum_2a + train2a
    """
    if (__debug__):
            print 'Shape of sum_2 = sum_2a + train2a: ', sum_2.shape
    #endif debugging
    
    memUsed = psutil.virtual_memory().used/(1024*1024)
    if (__debug__):
        print 'Memory (MB) used after computing the whole sum: ', memUsed
    #endif debugging
    """
        
    ## Then the ell-2 distance for each test/train pair is the square
    ##  root of the corresponding element of sums_of_squares:
    dists = np.sqrt(sum_2)  
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
  ## end compute_distances_no_loops(self, X)
  
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
    if (__debug__):
        print 
        print 'Inside predict_labels(self, dists, k==', k,') ...'
    #endif debugging

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
            #closest_y.append(sorted_y[j])
            closest_y.append(self.y_train[sorted_y[j]])
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
        
        if (__debug__ and ((k<6)  and (i<3)) ):
            print 'Inside predict_labels() len(y_hist)==', len(y_hist)
            print '                     len(closest_y)==', len(closest_y)
            # print '  y_hist[0]: ', y_hist[0]
            print '  np.argmax(y_hist[0])==', np.argmax(y_hist[0])
            print '  closest_y: ', closest_y
            print '  self.y_train[closest_y]==', self.y_train[closest_y]
            print '  self.classes[self.y_train[closest_y][0]]== "', self.classes[self.y_train[closest_y][0]], '"'
        #endif debugging
        
        
        index_of_shortest_max_so_far = np.argmax(y_hist[0])
        if (__debug__ and ((k<6)  and (i<3))):
            print 'Inside predict_labels() len(y_hist[0])==', len(y_hist[0])
            print '          index_of_shortest_max_so_far==', index_of_shortest_max_so_far
        #endif debugging
        
        max_freq = y_hist[0][index_of_shortest_max_so_far]
        
        """
        if (__debug__ and ((k<6) and (i<3)) ):
            print 'index_of_shortest_max_so_far==', index_of_shortest_max_so_far
        #endif debugging
        
        shortest_label_so_far = self.classes[self.y_train[index_of_shortest_max_so_far]]
        
        if (__debug__ and (k<4)):
            print ' shortest_label_so_far: "', shortest_label_so_far, '"'
        #endif debugging
        
        
        shortest_len_so_far = len(shortest_label_so_far)
        nbr_with_max_freq = 1
        if (__debug__):
            print '(Not) About to loop from n==1 to n==', len(y_hist[0])-1
        #endif debugging
        """  
        """This loop seems to be infinite.
        for n in range(1,len(y_hist[0])):
            if (y_hist[0][n] == max_freq):
                # Found another image with the same max frequency.
                nbr_with_max_freq += 1
                this_index = y_hist[1][n]
                this_label = self.classes[self.y_train[this_index]]
                this_length = len(this_label)
                if (this_length < shortest_len_so_far):
                    index_of_shortest_max_so_far = this_index
                    shortest_label_so_far = this_label
                    shortest_len_so_far = this_length
                #endif found a shorter one
            #endif found another with  max frequency.
        #endwhile walking through the histogram.
        """
        
        y_pred[i] = index_of_shortest_max_so_far
      #endif (k==1) -- else
      """
      if (__debug__ and (k<4)):
        print 'Leaving predict_labels() ...'
        print 'Memory used (MB): ', psutil.virtual_memory().used/(1024*1024)
        print
      #endif debugging
      """
    #endfor i
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################
    
    return y_pred
  ## end predict_labels()
