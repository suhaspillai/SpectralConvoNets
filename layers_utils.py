#Author : Suhas.Pillai

import numpy as np
from layers import *
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import sys
sys.path.append('/home/sbp3624/scikit-cuda')
import skcuda.linalg as linalg
import skcuda.misc as m


class layers_utils:
    
    '''
    The class is used to create different deep networks by stacking layers together.
    '''
    
    def __init__(self):
       pass
        

    def  Convo_Relu_Forward(self,x,w,b,U):
        l_obj = layers()
        out_spectral,spectral_conv_cache = l_obj.spectralForwardprop_cuda(x,w,b,U,linalg.transpose(U))
        out_relu, relu_cache = l_obj.relu_forward_cuda(out_spectral)
        cache = (spectral_conv_cache,relu_cache)
        return out_relu,cache
        
    def Convo_Relu_Backward(self,dout,cache):
        """
        Backward pass
        """
        l_obj = layers()
        spectral_conv_cache, relu_cache = cache

        da = l_obj.relu_backward_cuda(dout,relu_cache)
        dx,dw,db = l_obj.spectralConvolutionBackprop_Final_cuda(da,spectral_conv_cache)

        return dx.get(),dw,db

    def  Convo_Relu_Pool_Forward(self,x,w,b,U,pool_param):
        l_obj = layers()
        out_spectral,spectral_conv_cache = l_obj.spectralForwardprop_cuda(x,w,b,U,linalg.transpose(U))
        out_relu, relu_cache = l_obj.relu_forward_cuda(out_spectral)
        #ouit_relu_cpu=out_relu.get()
        out_pool,pool_cache=l_obj.max_pool_forward_cuda(out_relu,pool_param)
        cache = (spectral_conv_cache,relu_cache,pool_cache)
        return out_pool,cache

    def Convo_Relu_Pool_Backward(self,dout,cache):
        """
        Backward pass
        """
        l_obj = layers()
        spectral_conv_cache, relu_cache = cache

        da = l_obj.relu_backward_cuda(dout,relu_cache)
        dx,dw,db = l_obj.spectralConvolutionBackprop_Final_cuda(da,spectral_conv_cache)

        return dx.get(),dw,db
 

    def Conv_Relu_Pool_Conv_Relu_Forward(self,x,w1,w2,b1,b2,U,U2,pool_param):
        '''
        The method is deep network in Convo->Relu->Conv->Relu
        '''
        
        l_obj = layers()
        out_spectral_1,spectral_conv_cache_1 = l_obj.spectralForwardprop_cuda(x,w1,b1,U,linalg.transpose(U))
        out_relu_1, relu_cache_1 = l_obj.relu_forward_cuda(out_spectral_1)
        #ouit_relu_cpu=out_relu.get()
        out_pool_1,pool_cache_1=l_obj.max_pool_forward_cuda(out_relu_1,pool_param)

        out_spectral_2,spectral_conv_cache_2 = l_obj.spectralForwardprop_cuda(out_pool_1,w2,b2,U2,linalg.transpose(U2))
        out_relu_2, relu_cache_2 = l_obj.relu_forward_cuda(out_spectral_2)
        cache = (spectral_conv_cache_1,relu_cache_1,pool_cache_1,spectral_conv_cache_2,relu_cache_2) 
        return out_relu_2,cache 
        


    def Convo_Relu_Pool_Conv_Relu_Backward(self,dout,cache):
        """
        Backward pass for deep network in Convo->Relu->Conv->Relu
        """
        #pdb.set_trace()
        l_obj = layers()
        #spectral_conv_cache, relu_cache = cache
        spectral_conv_cache_1,relu_cache_1,pool_cache,spectral_conv_cache_2,relu_cache_2=cache       
        da = l_obj.relu_backward_cuda(dout,relu_cache_2)
        dx_2,dw_2,db_2 = l_obj.spectralConvolutionBackprop_Final_cuda(da,spectral_conv_cache_2)
        da_pool=l_obj.max_pool_backward_cuda(dx_2, pool_cache)
        da = l_obj.relu_backward_cuda(da_pool,relu_cache_1)
        dx_1,dw_1,db_1 = l_obj.spectralConvolutionBackprop_Final_cuda(da,spectral_conv_cache_1)
      

        return dx_1.get(),dw_1,db_1,dw_2,db_2

    def Convo_Relu_pool_Backward(self,dout,cache):
        """
        Backward pass
        """
        l_obj = layers()
        spectral_conv_cache, relu_cache,pool_cache = cache
        da_pool = max_pool_backward_naive(dout, pool_cache)     
        da_relu = l_obj.relu_backward(da_pool,relu_cache)
        dx,dw,db = l_obj.spectralConvolutionBackprop_Final(da_relu,spectral_conv_cache)

        return dx,dw,db
    

