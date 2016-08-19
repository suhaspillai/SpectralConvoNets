#Author : Suhas . Pillai

import numpy as np
from numpy import linalg as LA
import pdb
class layers:


    def affine_forward(x, w, b):
        
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
        We multiply this against a weight matrix of shape (D, M) where
        D = \prod_i d_i

        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)
          
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        out=x_new.dot(w)+b
        cache = (x, w, b)
        return out, cache


    def affine_backward(dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        dx, dw, db = None, None, None
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dx=np.zeros(x_new.shape)
        N =x.shape[0]
        db=db+np.sum(dout,axis=0)
        dx=dout.dot(w.T)
        dx=dx.reshape(x.shape) 
        dw=(x_new.T).dot(dout)
        return dx, dw, db

    def relu_forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
           
        out = None
        mask=np.zeros(x.shape)
        mask[x>0]=1
        out=x*mask
        cache = x
        return out, cache

    def relu_backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        mask=np.zeros(x.shape)
        mask[x>0]=1 
        dout=mask*dout
        dx=dout 
        return dx

    def spectralConvolution(self,Xtrain, U,W):
        '''
        The method is used to do forward propagation in spectral domain
        Xtrain :  N,c,w,h
        U : w*h,w*h        # tranformation matrix from spatial to spectral domain
        W : F,c,w,h

        return :
        X_ret (N,w,h,F) : X after convolution in spectral domain, bring it back to spatial domain
        to do pooling and passing it through non-linear activation function. 
        '''
        N = Xtrain.shape[0]       
        F,c,w,h = W.shape
        X_convo = np.zeros((N,F,c,w,h))
        X_convo_sum_chan= np.zeros((N,F,1,w,h))   # Sum across channels
        for i  in xrange(N):
            for j in xrange(F):
                X_convo[i,j] =  Xtrain[i] * W[j]
            #Summing it across channels
            for k in xrange (c):
                 X_convo_sum_chan[i,:,0,:,:] += X_convo[i,:,k,:,:]

        # Reshaping X_convo_sum_chan (N,F,w,h,c) to X_convo_sum_chan (N,w,h,F) as c will be 1

        X_convo_sum_chan = X_convo_sum_chan.reshape(N,F,w*h)
        
        # Going back to spatial domain
        
        X_ret = X_convo_sum_chan [:].dot(U)
        X_ret = X_ret.reshape(N,F,w,h)
                  
        cache={}
        cache['X_train']=Xtrain
        cache['W'] = W 
        cache['eigenvectors']=U
        return  X_ret,cache
    
        
    def spectralConvonlutionBackprop (self,dout,cache):
        '''
        The method performs backpropagation in spectral domain 

        Input:
        
        dout = Gradients from the forward layers  - shape (N,F,w,h)

        Output:
        dx -  gradients with respect to input X
        dw - gradients with respect to weights.
        
        '''
        
        X_train = cache['X_train']
        N,c,w,h = X_train.shape
        W = cache['W']
        F,c_f,w_f,h_f = W.shape               #(10, 1, 28, 28)  1->3
        W = W.reshape(F,c_f,w_f*h_f)     #(10, 1, 784)    1->3
        U = cache['eigenvectors']
        dx = np.zeros((N,c,w*h))            #(1000, 1, 784)   1->3
        dw = np.zeros((F,c_f, w_f*h_f))   #(10, 1, 784)   1->3
        
        mask = np.ones(c).reshape(1,c)
        for i in xrange(N):
            temp_train = X_train [i].reshape(c,w*h)   #(3*784)
            for j in xrange(F):
                dout_temp = dout[i,j,:,:]   # temp = 28*28
                dout_temp = dout_temp.reshape(w*h*c,1) # so now this will become 784 * 1 
                split_temp = mask * dout_temp        # This will make it  784 * 3
                dw[j,:]= dw[j,:] +   temp_train * split_temp.T   #  (3*784 times 3*784)This will  do it for all channels 
                dx [i,:] = dx[i,:] + W[j] * split_temp.T             # (3*784 times 3 *784)
            
                
        dx = np.dot(dx,U)           #same as U.T*dx.T     
        return  dx,dw
    
        
    def max_pool_forward_naive(x, pool_param):
        """
        A naive implementation of the forward pass for a max pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
        """
        out = None
        pool_height=pool_param['pool_height']
        pool_width=pool_param['pool_width']
        stride=pool_param['stride']
        N,C,H,W=x.shape
        pool_img_height=((H-pool_height)/stride)+1
        pool_img_width=((W-pool_width)/stride)+1
        out=np.zeros((x.shape[0],x.shape[1],pool_img_height,pool_img_width))
        for i in xrange(N):
          flag=1
          row_traverse=0
          for row_count in xrange(pool_img_height):
            column_traverse=0
            if flag==0:
              row_traverse=row_traverse+stride
            for column_count in xrange(pool_img_width):
              out[i,:,row_count,column_count]=np.amax(np.amax(x[i,:,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride],1),1)
              column_traverse=column_traverse+stride
              flag=0
        cache = (x, pool_param)
        return out, cache

    def max_pool_backward_naive(dout, cache):
        """
        A naive implementation of the backward pass for a max pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        x,pool_param=cache
        pool_height=pool_param['pool_height']
        pool_width=pool_param['pool_width']
        stride=pool_param['stride']
        N,C,H,W=x.shape
        pool_img_height=((H-pool_height)/stride)+1
        pool_img_width=((W-pool_width)/stride)+1
        dx=np.zeros((x.shape))
        temp=np.zeros((C,pool_height,pool_width))
        for i in xrange(N):
          flag=1
          row_traverse=0
        
          for row_count in xrange(pool_img_height):
            column_traverse=0
            if flag==0:
              row_traverse=row_traverse+stride
            for column_count in xrange(pool_img_width):
              flag=0
              temp=np.zeros((C,pool_height,pool_width))
              for channel in xrange(C):
                val=np.max(x[i,channel,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride])
                a_row,a_col=np.where(val==x[i,channel,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride])
              #temp=x[i,0,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride]
                temp[channel,a_row[0],a_col[0]]=dout[i,channel,row_count,column_count]
                dx[i,channel,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride]=temp[channel]
              column_traverse=column_traverse+stride
              
        return dx


    def softmax_loss(x, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
          for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
          0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

