import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        
        class_data_set = dataset[np.where(dataset[:,-1].astype(float) == class_value)]
        self.prior = class_data_set.shape[0]/dataset.shape[0]
         
        temp_train_set = class_data_set[0]
        self.temp_mean = temp_train_set.mean()
        self.temp_std = temp_train_set.std()
        
        humi_train_set = class_data_set[1]
        self.humi_mean = humi_train_set.mean()
        self.humi_std = humi_train_set.std()
        
        print("for class ", class_value, " the prior is ", self.prior, 
              " temp_mean=", self.temp_mean, " temp_std=", self.temp_std,  
              " humi_mean=", self.humi_mean, " humi_std=", self.humi_std)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        p0=normal_pdf(x[0], self.temp_mean, self.temp_std)
        p1=normal_pdf(x[1], self.humi_mean, self.humi_std)
        return (p0 * p1)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ( self.get_instance_likelihood(x) * self.get_prior() )
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        
        self.class_value = class_value
        samples, features = dataset.shape
        self.d = features
        
        class_data_set = dataset[np.where(dataset[:,-1].astype(float) == class_value)]
        self.prior = class_data_set.shape[0]/dataset.shape[0]
        self.cov_matrix = np.cov(m=class_data_set, y=None, rowvar=False, bias=False, ddof=None, fweights=None, aweights=None)[:-1, :-1]
        self.cov_matrix_inv = np.linalg.inv(self.cov_matrix)
        self.cov_matrix_det = np.linalg.det(self.cov_matrix)
         
        temp_train_set = class_data_set[0]
        self.temp_mean = temp_train_set.mean()

        humi_train_set = class_data_set[1]
        self.humi_mean = humi_train_set.mean()
        
        
        self.mean_vector = np.array([self.temp_mean, self.humi_mean]) #self.mean_vector (2,)

        
        
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
       
        
        return (multi_normal_pdf(x[:-1], self.mean_vector, self.cov_matrix))
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ( self.get_instance_likelihood(x) * self.get_prior() )
    
    
def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    
    normal = 1 / (np.sqrt( 2 * np.pi * np.power(std, 2))) * np.exp( - np.power((x - mean), 2) / (2 * np.power(std, 2) ))
    return normal

                                                                            
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.        
    """
    dim = x.shape[0] - 1 # get the dimensions of the vector
    
    a = np.power((np.pi * 2), (- dim / 2))
    b = np.power(np.linalg.det(cov), -0.5)  
    
    c1 = ((x - mean).reshape(1, x.shape[0])).dot(np.linalg.inv(cov))
    c2 = c1.dot(np.vstack(x - mean))
    c3 = np.exp(-0.5 * c2)
    c=c3[0][0]
    
   
                                                                        
    multi_normal = a * b * c                                                      
    #print("multi_normal return: ", multi_normal)
    #print("a: ", a, " b: ", b, " c1: " ,c2, " c2: " ,c3, " c3: " ,c, " c: " ,c)
    #print("((x - mean).reshape(1, x.shape[0])): ", ((x - mean).reshape(1, x.shape[0])), " np.linalg.inv(cov): ", np.linalg.inv(cov), " np.vstack(x - mean): " ,np.vstack(x - mean)
   # print("((x - mean).reshape(1, x.shape[0])).shape: ", ((x - mean).reshape(1, x.shape[0])).shape, " np.linalg.inv(cov).shape: ", np.linalg.inv(cov).shape, " np.vstack(x - mean).shape: " ,np.vstack(x - mean).shape)
    #raise Exception
    return multi_normal


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        
        class_data_set = dataset[np.where(dataset[:,-1].astype(float) == class_value)]
        self.prior = (class_data_set.shape[0] + 1) / (dataset.shape[0] + 2)
         
        temp_train_set = class_data_set[0]
        self.temp_mean = temp_train_set.mean()
        self.temp_std = temp_train_set.std()
        
        humi_train_set = class_data_set[1]
        self.humi_mean = humi_train_set.mean()
        self.humi_std = humi_train_set.std()
        
        print("for class ", class_value, " the prior is ", self.prior, 
              " temp_mean=", self.temp_mean, " temp_std=", self.temp_std,  
              " humi_mean=", self.humi_mean, " humi_std=", self.humi_std)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        p0=normal_pdf(x[0], self.temp_mean, self.temp_std)
        p1=normal_pdf(x[1], self.humi_mean, self.humi_std)
        return (p0 * p1)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ( self.get_instance_likelihood(x) * self.get_prior() )

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        if(self.ccd1.get_instance_posterior(x) < self.ccd0.get_instance_posterior(x)):
            return 0
        else:
            return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
              
    inst_num, atrb_num = testset.shape
    
    error_cnt = 0
    for i in range (0, inst_num):
        pre = map_classifier.predict(testset[i,:])
        if(pre != testset[i,-1]):
            error_cnt += 1
    
    error_rate = (error_cnt/inst_num)
    accuracy = (1-error_rate)
    
    return accuracy

def normalize(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """
    
    ###########################################################################
    # TODO: Implement the normalization function.   
    # (x - mean) / (max - min)
    ###########################################################################
    xmean = np.mean(X, axis=0)
    
    xdenominator = np.max(X,axis=0) - np.min(X, axis=0)
    
    ydenominator = np.max(y, axis=0) - np.min(y, axis=0)
    
    X = (X - xmean) / (xdenominator)
    y = y / (ydenominator)    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y
