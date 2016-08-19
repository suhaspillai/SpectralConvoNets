#Author Suhas.Pillai

import numpy as np
from layers_utils import *
from layers import *
from loadData import *
import os
from numpy import linalg as LA
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import sys
sys.path.append('/home/sbp3624/scikit-cuda')
import skcuda.linalg as linalg
import skcuda.misc as m


class SpectralConvoNet:
    '''
    Class initiliaze the filters for spectral domain. 
    '''
    
    count_check=None
    def  __init__(self):
        self.count_check=0
        pass
    
    if (os.path.isfile('file_gradient.txt')):
        os.remove('file_gradient.txt')  

    
    def  two_layer_convonet_model(self,X,model,U,U_2,pool_param,y=None,reg=0.0):
        '''
        The function calls spectral layers wuth pooling and fully connected layers.
        X -  train / val /  test data.
        model - weights for training.
        U - Transformation matrix for going from spatial domain to spectral domain for 1st layer, calculated using laplacian.
        U_2 - Transformation matrix for going from spatial domain to spectral domain for 1st layer, calculated using laplacian.
        pool_param -  dictionary containing pooling paramters
        reg - regularization.

        Output -
        loss - loss
        grads -  gradients for all the layers.
        
        '''
        
        l_util = layers_utils()       # For convonet sub models
        data_obj = data_adj()     # For adjacency matrix
        layer =layers()                 # For individual layers
        W1,W2,W3,W4,b1,b2,b3,b4 = model['W1'],model['W2'],model['W3'],model['W4'],model['b1'],model['b2'],model['b3'],model['b4']
        N,C,WW,HH = X.shape
        U=U.astype(np.float32)
        U_2=U_2.astype(np.float32)
        
        # put everything on gpu

        W1_gpu=gpuarray.to_gpu(W1)
        W2_gpu=gpuarray.to_gpu(W2)
        W3_gpu=gpuarray.to_gpu(W3)
        W4_gpu=gpuarray.to_gpu(W4)
        b1_gpu=gpuarray.to_gpu(b1)
        b2_gpu=gpuarray.to_gpu(b2)
        b3_gpu=gpuarray.to_gpu(b3)
        b4_gpu=gpuarray.to_gpu(b4)
        U_gpu=gpuarray.to_gpu(U)
        U2_gpu=gpuarray.to_gpu(U_2)         
        X_gpu=gpuarray.to_gpu(X)  

        #------------------------------------Doing Forward Pass --------------------------------#
        
        out_CRP_CR_gpu, cache_CRP_CR = l_util.Conv_Relu_Pool_Conv_Relu_Forward(X_gpu,W1_gpu,W2_gpu,b1_gpu,b2_gpu,U_gpu,U2_gpu,pool_param)
        out_affine_gpu,cache_affine = layer.affine_forward_cuda(out_CRP_CR_gpu,W3_gpu,b3_gpu)
        scores_gpu, cache_frwd = layer.affine_forward_cuda(out_affine_gpu,W4_gpu,b4_gpu)
        scores=scores_gpu.get()

        if y is None:
            return scores

        data_loss,dscores = layer.softmax_loss(scores,y)
        
    
        # -----------------------------Doing the backward pass-----------------------------------#
        
        dscores=dscores.astype(np.float32)
        dscores_gpu=gpuarray.to_gpu(dscores)
        da4_gpu,dW4,db4 = layer.affine_backward_cuda(dscores_gpu,cache_frwd)
        da3_gpu,dW3,db3 = layer.affine_backward_cuda(da4_gpu,cache_affine)
        da1,dW1,db1,dW2,db2=l_util.Convo_Relu_Pool_Conv_Relu_Backward(da3_gpu,cache_CRP_CR)
        dW1 +=  reg*W1
        dW2 += reg*W2
        dW3 +=reg*W3
        dW4 += reg * W4
        reg_loss = 0.5 * reg * sum(np.sum(W*W) for W in [W1,W2,W3,W4])
        
        loss = data_loss +reg_loss
        grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2, 'W3':dW3,'b3':db3, 'W4':dW4,'b4':db4}
        
        return loss, grads



    def interpolation(self,num_filters,C,filter_size,filter_size_new):
        '''
        The method initializes weight matrix in spectreal domain, use linear interpolation
        to transform matrix from spatial domain to spectral domain.

        Input

        num_filters - Total number of filetrs.
        C - channels
        filter_size - original size of the filters.
        filter_size_new - The size of filters in spectral domain after interpolation.

        Output
        W_disp - weights in spectral domain.
        
        ''' 
        W1 = 0.001 * np.random.randn(num_filters,C,filter_size,filter_size)   # check weight_scale later on
        #filter_size_new=32
        N=filter_size_new
        W1_disp=np.zeros((W1.shape[0],W1.shape[1],N,N))

        min_arr=np.zeros((W1.shape[0],W1.shape[1],W1.shape[2]))
        max_arr=np.zeros((W1.shape[0],W1.shape[1],W1.shape[2]))
        f_first_min=np.zeros(W1.shape[1])
        f_first_max=np.zeros(W1.shape[1])
        f_last_min=np.zeros(W1.shape[1])
        f_last_max=np.zeros(W1.shape[1])
        first_row=np.zeros((W1.shape[1],N))
        last_row = np.zeros((W1.shape[1],N))



        for sample in xrange(W1.shape[0]):
                for channel in xrange(W1.shape[1]):
                        min_arr[sample,channel]=np.min(W1[sample,channel],1)
                        max_arr[sample,channel]=np.max(W1[sample,channel],1)
                        f_first_min[channel] = np.min(min_arr[sample,channel])
                        f_first_max[channel] = np.max(min_arr[sample,channel])
                        f_last_min[channel] = np.min(max_arr[sample,channel])
                        f_last_max[channel] = np.max(max_arr[sample,channel])

                for i in xrange(f_first_min.shape[0]):
                        first_row[i] = np.linspace(f_first_min[i],f_first_max[i],N)
                        last_row[i]= np.linspace(f_last_min[i],f_last_max[i],N)
                for c in xrange(W1.shape[1]):
                        for k in xrange(N):
                                W1_disp[sample,c,k,:]=np.linspace(first_row[c,k],last_row[c,k],N)
        W1_disp=np.asarray(W1_disp,np.float32)
        W1_disp.reshape(W1.shape[0],W1.shape[1], filter_size_new, filter_size_new)
        return W1_disp


    
    def initialize_parameters(self,weight_scale =1e-3,bias_scale=0, input_shape=(3,32,32),num_classes=10,num_filters_1=32,num_filters_2=64,org_filter_size=3):
        C,WW,HH=input_shape
        model={}
	#model['W1']= W1_disp.reshape(W1.shape[0],W1.shape[1], filter_size_new, filter_size_new)
        model['W1']= self.interpolation(num_filters_1,C,org_filter_size,32)
	#pdb.set_trace()
 	model['b1'] = bias_scale * np.asarray(np.random.randn(num_filters_1,1),np.float32)
        model['W2']= self.interpolation(num_filters_2,num_filters_1,org_filter_size,16)
        model['b2']= bias_scale * np.asarray(np.random.randn(num_filters_2,1),np.float32)
        model['W3'] = 0.001* np.asarray(np.random.randn(num_filters_2*HH/2*WW/2,1000),np.float32)
        model['b3'] = bias_scale * np.asarray(np.random.randn(1000,1),np.float32)
        model['W4'] = 0.001* np.asarray(np.random.randn(1000,num_classes),np.float32)
        model['b4'] = bias_scale * np.asarray(np.random.randn(num_classes,1),np.float32)
        return model
        
        
        
        

