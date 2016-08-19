import numpy as np
import gzip
import cPickle as cp
import pdb
class data_adj:
    
    
    def __init__(self):
        pass
   
    def loadData(self, width,height,data):
        N = len(data[0])
        
        X_train = np.zeros((N,width*height))
        y_train= np.zeros((N,1))
        
        for i in xrange(N):
            X_train[i] = data[0][i]
            y_train[i] = data[1][i]
        
        return X_train,y_train
        
    def adjacencyMat(self, X_train):
        
        #dict_adjmat={}
        N,C,W,H = X_train.shape
        count = 0
        adj_matrix = np.zeros((W*H,W*H))
                              
       
        for i in xrange(W):
            #pdb.set_trace()
                     # have to strech/unfold later on
            for j in xrange(W):
                adj_mat = np.zeros((W,H))
                if i==0 :
                    if j==0:
                        adj_mat[i,j+1] = 1
                        adj_mat[i+1,j] = 1
                        adj_mat[i+1,j+1] = 1
                            
                    elif j==W-1:
                        adj_mat[i+1][j-1] = 1
                        adj_mat[i][j-1] = 1
                        adj_mat[i+1][j] = 1
                            
                    else:
                        adj_mat[i][j-1] = 1
                        adj_mat[i][j+1] = 1
                        adj_mat[i+1][j] = 1
                        adj_mat[i+1][j-1]=1
                        adj_mat[i+1][j+1]=1
                               
                elif i ==H-1:
                    if j==0:
                        adj_mat[i-1][j] = 1
                        adj_mat[i-1][j+1] = 1
                        adj_mat[i][j+1] = 1
                            
                    elif j==W-1:
                        adj_mat[i][j-1] = 1
                        adj_mat[i-1][j] = 1
                        adj_mat[i-1][j-1] = 1
                            
                    else:
                        adj_mat[i][j-1] = 1
                        adj_mat[i][j+1] = 1
                        adj_mat[i-1][j] = 1
                        #addded 
                        adj_mat[i-1][j-1]=1 
                        adj_mat[i-1][j+1]=1
                         
                                
                elif j==0 and i>0 and i < W-1:
                    adj_mat[i-1][j] = 1
                    adj_mat[i-1][j+1] = 1
                    adj_mat[i][j+1] = 1
                    adj_mat[i+1][j+1] = 1
                    adj_mat[i+1][j] = 1
                        
                            
                elif j==H-1 and i>0 and i < W-1:
                    adj_mat[i-1][j] = 1
                    adj_mat[i-1][j-1] = 1
                    adj_mat[i][j-1] = 1
                    adj_mat[i+1][j-1] = 1
                    adj_mat[i+1][j] = 1
                        
                else:
                    adj_mat[i][j-1] = 1
                    adj_mat[i][j+1] = 1
                    adj_mat[i-1][j] = 1
                    adj_mat[i+1][j] = 1
                    adj_mat[i-1][j-1] = 1
                    adj_mat[i+1][j-1] = 1
                    adj_mat[i-1][j+1] = 1
                    adj_mat[i+1][j+1] = 1

                adj_matrix[count] = np.ravel(adj_mat)
                    #print (adj_image)
                    
                count = count+1     
                  
        return adj_matrix
    
    
    def getLaplcaian(self,dict_adj,typeLaplacian):
        X = dict_adj
        W = X.shape[0]
        H = X.shape[0] 
        Unitary_mat = np.ones((W,1))
        dict_laplacian=np.zeros((W,H))
        #print (dict_laplacian.shape)
        D_vect = X.dot(Unitary_mat)
        D = np.zeros(X.shape)       
        for j in xrange(W):
            D[j,j] = D_vect[j]
                
        if typeLaplacian=="UNorm":
            dict_laplacian[i] = D-X
        else:
            for j in xrange(W):
                D[j,j] = 1/np.sqrt(D[j,j])

            I = np.eye(D.shape[0])
            temp = D.dot(X)
            dict_laplacian= I - temp.dot(D)
            
            '''Vectorized way to get D
            I = np.eye(D.shape[0])
            D[I==1] = D_vect
            D[I==1] = 1/np.sqrt(D_vect)
            '''
            
        return dict_laplacian

        '''
        def getLaplcaian(self,dict_adj,laplacianType):
        
        N = len(dict_adj)
        print "size %d" % (N)
        Unitary_mat = np.ones(N)
        dict_laplacian=np.zeros(N,dict_adj[0].shape) #shape of the laplacian
        for i in xrange(N):
            X = dict_adj[i]
            D_vect = X.dot(Unitary_mat)
            D = np.zeros(X.shape)
            for j in xrange(N):
                D[j,j] = D_vect[j]

            
            
            dict_laplacian[i] = I-(1/np.sqrt(D))*X*(1/np.sqrt(D)) #normalized laplacian
        
        return dict_laplacian 

        
        return dict_laplacian 
        '''
        
        
        
    
    
        
                
                                    
   
        
    
        
    
 
