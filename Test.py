#Author : Suhas.Pillai

import numpy as np
from layers import *
from gradient_check import *
from loadData import *
from layers_utils import *
from classifier_trainer import *
from SpectralConvoNet import *
from data_utils import *
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import sys
sys.path.append('/home/sbp3624/scikit-cuda') # Give path of skit cuda
import skcuda.linalg as linalg
import pdb
import cPickle as cp



def main():
    '''
    This the main function, which initializes weights and calls training function.
    '''
    print 'Start training !!!! \n'
    layers_obj= layers()
    data_obj = data_adj()
    linalg.init()          # initialization required for GPU computation
    #----------------------------- Loading CIFAR data-------------------------#  
    X_train,y_train,X_test,y_test=load_CIFAR10('/home/sbp3624/Graph_CNN_python_copy/GraphCNN_python/cifar-10-batches-py')
    num_training = 8000     # Training split
    num_validation =2000    # Testing Split 
    num_test=1000           #Test split
    mask = range(num_training, num_training+num_validation)
 
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask =range(num_training)
    X_train = X_train[mask]
    y_train=y_train[mask]
    mask = range(num_test) 
    X_test = X_test[mask]
    y_test=y_test[mask]
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    
    N = X_train.shape[0]
    mean = np.sum(X_train,axis=0)/N
    X_train = X_train - mean
    X_val =  X_val-mean
    X_test = X_test-mean
    print (X_train.shape,X_val.shape,X_test.shape,y_train.shape,y_val.shape,y_test.shape)
    
    #Convert all to float32 type.
    X_train=X_train.astype(np.float32)
    X_val=X_val.astype(np.float32)
    X_test=X_test.astype(np.float32)
    
    train_obj = ClassifierTrainer()
    spectral_convo_obj = SpectralConvoNet()
    model =spectral_convo_obj .initialize_parameters() 
    adj_matrix = data_obj.adjacencyMat(X_train)

    #Calculating Laplacian
    #Laplacaian for 1st layer
    dict_laplacian = data_obj.getLaplcaian(adj_matrix,"Norm")
    E_val,E = LA.eig(dict_laplacian)
    U = np.zeros(E.shape)
    for i_iter in xrange(U.shape[0]):
      for j_iter in xrange(U.shape[1]):
        U[i_iter,j_iter] = E[i_iter,j_iter].real    


    # Laplacian for 2nd Layer  
    X_adj_mat_2=np.zeros((1,1,16,16))
    adj_matrix_2=data_obj.adjacencyMat(X_adj_mat_2)
    dict_laplacian_2 = data_obj.getLaplcaian(adj_matrix_2,"Norm")
    E_val_2,E_2 = LA.eig(dict_laplacian_2)
    U_2 = np.zeros(E_2.shape)
    for i_iter in xrange(U_2.shape[0]):
      for j_iter in xrange(U_2.shape[1]):
        U_2[i_iter,j_iter] = E_2[i_iter,j_iter].real


    # Pool Parameters.
    pool_param ={}
    pool_param['pool_height']=2
    pool_param['pool_width']=2
    pool_param['stride']=2
    
    
    f_loss=open("f_loss_kgcoe-2.txt","wb")
    f_train_acc=open("f_train_acc_kgcoe-2.txt","wb")
    f_val_acc =open("f_val_acc_kgcoe-2.txt","wb")
    
    #model = cp.load(open('param_save_50'))   # For loading saved model (Pretrained Model)

    # Calling the training function   
    best_model, loss_history, train_acc_history, val_acc_history = train_obj.train(X_train,y_train,X_val,y_val,model,pool_param,spectral_convo_obj.two_layer_convonet_model,U,U_2,reg=0.00334207889236,momentum=0.9,learning_rate=0.000229671465228,update="Adam",batch_size=200,num_epochs=50,verbose=True)
    f_loss.write("\n*******************************************************\n") 
    f_loss.write(str(loss_history))
    f_val_acc.write(str(val_acc_history))
    f_train_acc.write(str(train_acc_history))
    
    # Dumping learned parameters
    f_parameters = open("parameters_wb_two_layer.txt","wb")
    cp.dump(best_model,f_parameters) 
    f_loss.close()
    f_train_acc.close()
    f_val_acc.close()
    f_parameters.close()
    print "Done"
    
if __name__=="__main__":
    main()
