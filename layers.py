#!/opt/anaconda3/bin/python

# Loose basis: https://www.coursera.org/learn/convolutional-neural-networks/programming/4xt9A/convolutional-model-step-by-step/

# We need classes for: ConvLayer, PoolLayer, FlattenLayer, FCLayer, SoftMax/Output Layer
import numpy as np
import math
import time

class ConvLayer:
    """
    Class for a convolutional layer of a neural network.
    """
    def __init__(self, input, num_filters, filter_dim, pad = 0, stride = 1):
        """ Initialises a convolutional layer."""

        self.p = pad
        self.f = filter_dim
        self.s = stride
        self.n_C = num_filters
        self.input = np.asarray(input)
        self.W = np.random.random((self.f,self.f,self.input.shape[3],self.n_C))*0.01
        self.b = np.random.random((1,1,1,self.n_C))
        
    
    def zero_pad(self, x, mode = "ADD"):
        """
        Pad with zeros or unpad all images of the dataset X. The padding is applied
        to / removed from the height and width of an image.

        Arguments:
        input -- the input to the ConvLayer from the previous layer.
        
        Returns:
        Padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
        """
        if self.p == 0:
            return x

        if mode == "ADD":
            pad_dims = ((0,0), (self.p,self.p), (self.p, self.p), (0,0))
            x = np.pad(x, pad_dims)
            return x
        elif mode == "REMOVE":
            return x[:,self.p:-self.p,self.p:-self.p,:]
        else: 
            raise ValueError(f"Invalid mode: {mode}. Expected 'ADD' or 'REMOVE'.")  

        
        
    def single_conv(self, input_slice, current_f):
        """
        Perform one convolution step on a given slice and features.

        Arguments:
        input_slice -- slice of input data of shape (f, f, n_C_prev) 
        W -- a filter of weight params, shape (f, f, n_C_prev, n_C) 
        b -- bias parameters in a window, shape (1, 1, 1, n_C)

        Returns:
        Z -- a scalar, to be put into activation.
        """
        filter = self.W[:,:,:,current_f]
        Z = np.sum(np.multiply(input_slice, filter))
        + float(self.b[:,:,:,current_f])

        return Z
    
    def new_shape(self, x):
        """
        Find the output shape, using the input shape.

        Returns:
        out_h -- number of rows of layer output.
        out_w -- number of columns of layer output.
        """
        out_h = math.floor((x.shape[1] + 2 * self.p - self.f)/self.s) + 1
        out_w = math.floor((x.shape[2] + 2 * self.p - self.f)/self.s) + 1
        return [out_h, out_w]
    
    def select_slice(self, x, i, j, k):
        """
        Creates a slice of the input, ready to convolve with the filter.
        This prepares input for use in single_step function.
        Arguments:
        i -- the training example currently in use.
        j -- the current row of the convolution.
        k -- the current column of convolution.

        Returns:
        A slice of the input with dimensions equal to the filter.
        Dimensions (self.f,self.f, self.input.shape[3])

        """
        h1 = j*self.s
        h2 = h1 + self.f
        w1 = k*self.s
        w2 = w1 + self.f
        return x[i, h1:h2,w1:w2,:]

    def forward(self):       
        """
        Perform a forward pass through one convolutional layer.
        Returns:
        Z - the output, to be passed into activation.

        """
        m = self.input.shape[0]
        x = self.input
        out_h, out_w = self.new_shape(x)
        x = self.zero_pad(x, mode = "ADD")
        Z = np.zeros((m,out_h,out_w, self.n_C))
        for i in range(m):
            for j in range(out_h):  
                for k in range (out_w):
                    for l in range(self.n_C):
                        Z[i,j,k,l] = self.single_conv(self.select_slice(x, i, j, k),l)
                        
        return Z
    
    def backward(self, dZ, alpha):
        """
        Perform a backward pass through one convolutional layer.
        """
        m = self.input.shape[0]
        dA_prev = np.zeros(self.input.shape)
        dA_prev = self.zero_pad(dA_prev,mode="ADD")
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        x = self.zero_pad(self.input,mode="ADD")
        in_h, in_w = self.input.shape[1],self.input.shape[2]        
        for i in range(m):
            for j in range(in_h):  
                for k in range (in_w):
                    for l in range(self.n_C):

                        dA_prev_slice = self.select_slice(dA_prev,i,j,k)
                        dA_prev_slice += self.W[:,:,:,l] * dZ[i,j,k,l]
                        dW[:,:,:,l] += (self.select_slice(x,i,j,k))*dZ[i,j,k,l]
                        db[:,:,:,l] += dZ[i,j,k,l]
        
        self.W -= dW * alpha
        self.b -= db * alpha
        self.zero_pad(dA_prev, mode = "REMOVE")

            
        
        return dA_prev, dW, db
        

class PoolLayer:
    """
    Class for a pool layer of a neural network. Can be MAX or AVERAGE pool.
    """
    def __init__(self, input, stride, filter_width, mode = "MAX"):
        self.input = np.asarray(input)
        self.s = stride
        self.f = filter_width
        self.mode = mode

    
    # Functions: forward, max, average, backward,
    def select_slice(self, i, j, k, l):
            """
            Creates a slice of the input, ready to convolve with the filter.
            This prepares input for use in single_step function.
            Arguments:
            i -- the training example currently in use.
            j -- the current row of the convolution.
            k -- the current column of convolution.

            Returns:
            A slice of the input with dimensions equal to the filter.
            Dimensions (self.f,self.f, self.input.shape[3])

            """
            h1 = j * self.s
            h2 = h1 + self.f
            w1 = k * self.s
            w2 = w1 + self.f
            return self.input[i, h1:h2,w1:w2,l]
    
    def single_step(self, input_slice):
        if self.mode == "MAX":
            result = np.max(input_slice)
        elif self.mode == "AVERAGE":
            result = np.mean(input_slice)
        return result 

    def new_shape(self):
        out_h = math.floor((self.input.shape[1] - self.f)/self.s) + 1
        out_w = math.floor((self.input.shape[2] - self.f)/self.s) + 1
        return [out_h, out_w]
    
    def remove_pad(self):
        """
        Removes padding. 
        Arguments:
        input -- a padded activation.
        
        Returns:
        The activation with no padding, p = 0
        """
        return 

        
        

    def forward(self):
        """
        Performs a forward pool.
        """
        # How to do this?
        # Not a filter. Just pass a window and pick the max in each window
        # So, just take slices, move forward by stride, find max/average in each case
        m, n_C = self.input.shape[0], self.input.shape[3]
        out_h, out_w = self.new_shape()
        Z = np.zeros((m,out_h,out_w, n_C))
        for i in range(m):
                    for j in range(out_h):  
                        for k in range (out_w):
                            for l in range(n_C):
                                Z[i,j,k,l] = self.single_step(self.select_slice(i, j, k, l))
                                
        return Z
        

    def backward(self):
        pass
         

class FlattenLayer:
    """
    Class for a flatten layer of a neural network.
    """
    # This takes a convolution and flattens it out. that is it. so just a reshape 
    def __init__(self, input):
        self.input = np.asarray(input)


    def forward(self):
        reshape = (self.input.shape[0], int(self.input.size / self.input.shape[0]))
        return np.reshape(self.input, reshape)        

    def backward(self):
        pass


class FCLayer:
    """
    Class for a fully connected layer of a neural network.
    """
    def __init__(self):
        pass
    
    def activate(z ,function = "relu"):
        functions = {
        "relu": lambda z: np.maximum(0, z),
        "tanh": lambda z: np.tanh(z),
        "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
        "linear": lambda z: z,
        "leaky_relu": lambda z: np.maximum(0.1 * z, z)
        }
        if function not in functions:
            raise ValueError(f"Invalid function: {function}. 
                             Please inspect the permitted values.")
                
        return functions[function](z)

        
    def forward(self):
        pass
        
    def backward(self):
        pass


        

    









if __name__ == "__main__":
    """CONV TESTING"""
    # x = np.random.random((1,192,192,3))
    # convtest = ConvLayer(input=x,num_filters=3,filter_dim=3,pad=3,stride=1)
    # before = time.time()
    # test_z = convtest.forward()
    # dA_prev,dW,db = convtest.backward(test_z,alpha=0.01)
    # after = time.time()
    # print(f"Time taken: {after - before}")



    """POOL TESTING"""
    # x = np.random.random((1,3,3,3))
    # pooltest = PoolLayer(input = x, stride = 2, filter_width=2, mode = "AVERAGE")
    # print(f"input starts as: {pooltest.input}")

    # print(f"And input has shape: {pooltest.input.shape}")
    # print(f"So, first dim: {pooltest.input.shape[0]}")

    # before = time.time()
    # test = pooltest.forward()
    # after = time.time()
    # print(f"Our result is: {test}")
    # print(f"Result has shape: {test.shape}")
    # print(f"Time taken: {after - before}")

    """FLATTEN TESTING"""
    # x = np.random.random((1,3,3,3))
    # flattest = FlattenLayer(input=x)

    # test = flattest.forward()
    # print(f"Input is {flattest.input}")
    # print(f"And output is {test}")