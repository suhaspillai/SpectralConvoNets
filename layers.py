#Author : Suhas.Pillai

import numpy as np
from numpy import linalg as LA
import  pdb
import os
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import sys
sys.path.append('/home/sbp3624/scikit-cuda')
import skcuda.linalg as linalg
import skcuda.misc as m
import time
class layers:


    def affine_forward_cuda(self,x, w, b):
        
        """
        Computes the forward pass for an affine (fully-connected) layer on GPU.

        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)
          
        Returns a tuple of:
        - out_gpu: output, of shape (N, M)
        - cache: (x, w, b)
        """
        #pdb.set_trace()
        x_gpu=x
        w_gpu=w
        b_gpu=b
        out_gpu = None
        x_new=x_gpu.reshape(x_gpu.shape[0],np.prod(x_gpu.shape[1:]))
        dot_mat=linalg.dot(x_new,w_gpu)
        b_t=linalg.transpose(b_gpu)
        out_gpu=m.add(dot_mat,b_t)
        #give gpu arrays, so that it is easy for backward computation
        cache=(x_gpu,w_gpu,b_gpu)

        return out_gpu, cache


    def affine_forward_norm(self,x, w, b):

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
        #pdb.set_trace()
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        out=x_new.dot(w)+b.T    #change to match dimensions
        cache = (x, w, b)
        return out, cache




    def affine_backward_cuda(self,dout, cache):
        """
        Computes the backward pass for an affine layer on GPU.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx_gpu: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        
        dout_gpu=dout
        x_gpu,w_gpu,b_gpu=cache
        x_new=x_gpu.reshape(x_gpu.shape[0],np.prod(x_gpu.shape[1:]))
        dx, dw, db = None, None, None
        N =x_gpu.shape[0]
        dx_gpu=linalg.dot(dout_gpu,linalg.transpose(w_gpu))
        dx_gpu=dx_gpu.reshape(x_gpu.shape)
        x_new_t=linalg.transpose(x_new)
        dw_gpu=linalg.dot(x_new_t,dout_gpu)
        dw=dw_gpu.get()
        db=np.zeros(b_gpu.shape)
        db=db.astype(np.float32)
        db=np.sum(dout_gpu.get(),0)
        db=db.reshape(db.shape[0],1) 
        return dx_gpu, dw, db



    def affine_backward_norm(self,dout, cache):
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
        #pdb.set_trace()
        x, w, b = cache
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        dx, dw, db = None, None, None
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dx=np.zeros(x_new.shape)
        N =x.shape[0]
        db=db+(np.sum(dout,axis=0).reshape(b.shape[0],b.shape[1]))    #  change to match dimesions summing 
        dx=dout.dot(w.T)
        dx=dx.reshape(x.shape)
        dw=(x_new.T).dot(dout)
        return dx, dw, db



    def relu_forward_cuda(self,x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs) on GPU.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out_gpu: Output, of the same shape as x
        - cache: x
        """
          
        out = None
        x_gpu=x
        mask=np.zeros(x_gpu.shape)
        mask[x_gpu.get()>0]=1
	mask=mask.astype(np.float32)
        mask_gpu=gpuarray.to_gpu(mask) 
        
        out_gpu= linalg.multiply(mask_gpu,x_gpu)
        cache = x_gpu,mask_gpu
        return out_gpu, cache

    def relu_forward(self,x):
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



    def relu_backward_cuda(self,dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs) on GPU.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx_gpu: Gradient with respect to x
        """
        dx_gpu,(x_gpu,mask_gpu) = None, cache
	dout_gpu=dout
        dx_gpu=linalg.multiply(mask_gpu,dout_gpu) 
        return dx_gpu



    def relu_backward(self,dout, cache):
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


   
 

    def spectralForwardprop_cuda(self,X_train,W,b,U,i_U):
        '''
        The method performs Spectral Forward Propagation on GPU 
        '''      
        X_train_gpu=X_train
        W_gpu=W
        b_gpu=b
        U_gpu=U 
        i_U_gpu=i_U
        N,img_b_c,img_b_w,img_b_h = X_train_gpu.shape
        X_train_gpu = X_train_gpu.reshape(X_train_gpu.shape[0]*img_b_c,img_b_w * img_b_h)
        X_freq_basis_gpu = linalg.dot(X_train_gpu,U_gpu)
        X_freq_basis_gpu = X_freq_basis_gpu.reshape(N,img_b_c,img_b_w * img_b_h)  
        F,c,w,h = W_gpu.shape  # same dimesnions as  X-train (2,3,10)
        W_gpu=W_gpu.reshape(F,c,w*h)
        X_convo = np.zeros((N,F,1,w*h))
        
        for i in xrange(N):
            for j in xrange(F):
                X_convo[i,j]= np.sum((X_freq_basis_gpu[i] * W_gpu[j]).get(),0)+b[j].get()
        
        X_convo=X_convo.reshape(N*F,w*h)
        X_convo=X_convo.astype(np.float32)
        X_convo_gpu=gpuarray.to_gpu(X_convo)
        X_ret_gpu = linalg.dot(X_convo_gpu,i_U_gpu)
        X_ret_gpu = X_ret_gpu.reshape(N,F,img_b_w,img_b_h)
        X_train_gpu=X_train_gpu.reshape(N,img_b_c,img_b_w,img_b_h)
        W_gpu=W_gpu.reshape(F,c,w,h)
        cache=(X_train_gpu,W_gpu,b_gpu,U_gpu,i_U_gpu)
        return X_ret_gpu,cache



    
  
    def spectralConvolutionBackprop_Final_cuda(self,dout,cache):
        '''
        Backward Propagation in spectral domain on GPU 
        '''
        
        dout_gpu=dout
        X_train_gpu,W_gpu,b_gpu,U_gpu,i_U_gpu = cache
        dout_freq_basis_gpu= dout_gpu.reshape(dout_gpu.shape[0]*dout_gpu.shape[1],dout_gpu.shape[2]*dout_gpu.shape[3])     # 5 * 3 * 25
        dout_freq_basis_gpu=linalg.dot(dout_freq_basis_gpu,U_gpu) 
        dout_freq_basis_gpu=dout_freq_basis_gpu.reshape(dout_gpu.shape[0],dout_gpu.shape[1],U_gpu.shape[1])
        N,img_b_c,img_b_w,img_b_h = X_train_gpu.shape
        X_train_gpu = X_train_gpu.reshape(N*img_b_c,img_b_w * img_b_h) #(5,3,25)
        
        # First compute gradients with respect to x
        #X_freq_basis = np.dot(X_train,U)      # ----> 5 * 3 * 25    *  25 * 9  = 5 * 3 * 9
        X_freq_basis_gpu=linalg.dot(X_train_gpu,U_gpu)
        X_freq_basis_gpu =  X_freq_basis_gpu.reshape(N,img_b_c,U_gpu.shape[1]) 
        F,c_f,w_f,h_f = W_gpu.shape

        W_gpu = W_gpu.reshape(F,c_f,w_f*h_f)    # (3,3,9)
        dw_gpu = np.zeros((F,c_f,w_f,h_f))
        dw_gpu=dw_gpu.astype(np.float32)
        dw_gpu=gpuarray.to_gpu(dw_gpu)    
        db=np.zeros(b_gpu.shape)
        db=db.astype(np.float32)
        dx_b = np.zeros(X_freq_basis_gpu.shape)     # (5 * 3 *9) This  is for gradient with respect to x in spectral domain

        dx_b=dx_b.astype(np.float32)
        dx_b_gpu=gpuarray.to_gpu(dx_b)

        for i in xrange(N):
            for j in xrange(F):  
                dx_b_gpu[i]=m.add(dx_b_gpu[i],m.multiply(W_gpu[j],dout_freq_basis_gpu[i,j])) #using miscellaneous object m because it has broadcasting 

         # Now reshape go back to spactial domain.
        dx_b_gpu=dx_b_gpu.reshape(dx_b_gpu.shape[0]*dx_b_gpu.shape[1],dx_b_gpu.shape[2])
        dx_b_gpu=linalg.dot(dx_b_gpu,i_U_gpu)
        dx_b_gpu=dx_b_gpu.reshape(N,img_b_c,img_b_w,img_b_h) 
        dw_gpu = dw_gpu.reshape(F,c_f,w_f*h_f)
        for i in xrange(F):
            for j in xrange(N):
                dw_gpu[i] = m.add(dw_gpu[i], m.multiply(X_freq_basis_gpu[j], dout_freq_basis_gpu[j][i]))
                #temp= m.multiply(X_freq_basis_gpu[j],dout_freq_basis_gpu[j,i])                 
                db[i] = db[i]+ m.sum(dout_freq_basis_gpu[j][i]).get()

        dw_gpu = dw_gpu.reshape(F,c_f,w_f,h_f)
        dw=dw_gpu.get()
        #As dx_gpu wiil be used for backpropagation ,so no need to copy it to the host side.
        return dx_b_gpu,dw,db



    
        
    def max_pool_forward_naive(self,x, pool_param):
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

    def max_pool_forward_cuda(self,x_gpu, pool_param):
        """
        A implementation of the forward pass for a max pooling layer on GPU.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out_gpu: Output data
        - cache: (x, pool_param)
        """
        out = None
        x=x_gpu.get()   # This ahs to be changed for efficiency
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
        out=out.astype(np.float32) 
        out_gpu=gpuarray.to_gpu(out)
        cache = (x, pool_param)
        return out_gpu, cache


    def max_pool_backward_naive(self,dout, cache):
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
                temp[channel,a_row[0],a_col[0]]=dout[i,channel,row_count,column_count]
                dx[i,channel,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride]=temp[channel]
              column_traverse=column_traverse+stride
        
        return dx



    def max_pool_backward_cuda(self,dout_gpu, cache):
        """
        An implementation of the backward pass for a max pooling layer on GPU.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx_gpu: Gradient with respect to x
        """
        dx=None
        dout = dout_gpu.get()
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
                temp[channel,a_row[0],a_col[0]]=dout[i,channel,row_count,column_count]
                dx[i,channel,row_traverse:row_traverse+stride,column_traverse:column_traverse+stride]=temp[channel]
              column_traverse=column_traverse+stride

        dx=dx.astype(np.float32)
        dx_gpu=gpuarray.to_gpu(dx)
        return dx_gpu

    if (os.path.isfile('probs.txt')):
        os.remove('probs.txt')
    
    def softmax_loss(self,x, y):
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
        f_open = open('prob.txt','a')
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
       
        probs /= np.sum(probs, axis=1, keepdims=True)
        f_open.write(str(probs))
  
	N = x.shape[0]
    
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

