# Author : Suhas. Pillai
import numpy as np
import pdb
import time
import cPickle as cp

class ClassifierTrainer:
    '''
    This is where all the parameters are updated
    '''
    
    def __init__(self):
        self.step_cache={} #use for stroing momentum velocities
        self.m={}     
        self.v={}

    def train(self,X,y,X_val,y_val,model,pool_param,loss_function,U,U_2,reg=0.0,learning_rate=1e-2,momentum=0,
              learning_rate_decay=0.95,update="momentum",sample_batches=True,
              num_epochs=30,batch_size=100,acc_frequency=None,verbose=False):
        '''
        The method is used to train the network ,along with updating parameters with SGD.

        Input 
        X -  Train data
        y - Train ground truth
        X_val - Validation data
        y_val - validation ground truth data
        loss_function :  calling training function.
        U -  Laplacian Matrix for layer 1
        U_2 -  Laplacian matrix for layer 2
        reg - regularization
        learning_rate - learning ratre
        update - can be sgd/momentum/rmsprop/adam
        num_epochs - number of epochs to run.
        

        output:
        best_models :  Model parameters with best validation accuaracy
        -loss_history: List containing the value of the loss function at each iteration.
        -train_acc_history: List storing training set accuracy.
        -val_acc_history: List storing the validation set accuracy.
         '''    
         
        beta1=0.9
        beta2=0.999
        N=X.shape[0]
 
        if sample_batches:
            iterations_per_epoch=N/batch_size
        else:
            iterations_per_epoch=1

        num_iters=num_epochs*iterations_per_epoch
        epoch=0
        best_val_acc=0.0
        best_model={}
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        m=0
        v=0
        count_iter = 0
        for it in xrange(num_iters):
            if it%10==0: print 'starting iteration', it

            #get batch of data 
            if sample_batches:
                batch_mask=np.random.choice(N,batch_size)
                X_batch=X[batch_mask]
                y_batch=y[batch_mask]
            else:
                # no SGD, using full gradient
                X_batch=X
                y_batch=y

            cost,grads=loss_function(X_batch,model,U,U_2,pool_param,y_batch,reg )
            print ("Loss is :  %f") % (cost)

            # Break if loss is nan
            if (str(cost)=="nan") or (str(cost)=="inf"):
	      print "nan or inf value break out\n"
              break

            loss_history.append(cost)

            #--------------------perform parameter update----------------------------#
        
            for p in model:
                
                #pdb.set_trace() 
                if update=="sgd":
                    dx=-learning_rate*grads[p]
                elif update=="momentum":
                    if not p in self.step_cache:
                        self.step_cache[p]=np.zeros(grads[p].shape)
                    
                    dx=np.zeros_like(grads[p])
                    dx=momentum*self.step_cache[p]-learning_rate*grads[p]
                    self.step_cache[p]=dx
                    
                elif update=="rmsprop":  
                    #print "Inside RMS prop \n"                                      
                    decay_rate=0.99 #can keep this as an option
                    if not p in self.step_cache:
                        self.step_cache[p]=np.zeros(grads[p].shape)
                        dx=np.zeros_like(grads[p])
                    dx=grads[p]
                    self.step_cache[p]=decay_rate*self.step_cache[p]+(1-decay_rate)*dx**2
                    dx=-(learning_rate*dx)/(np.sqrt(self.step_cache[p])+1e-8)   
                elif update=="Adam":
                    #print "Inside Adam \n"
                    if p not in self.m:
                      self.m[p]=np.zeros(grads[p].shape)
                      self.v[p]=np.zeros(grads[p].shape)
                      dx=np.zeros_like(grads[p])
                    dx= grads[p]                     
                    self.m[p]= beta1*self.m[p] + (1-beta1) * dx
                    self.v[p]= beta2 * self.v[p] + (1-beta2) * (dx**2)
                    dx = -(learning_rate * self.m[p]) / (np.sqrt(self.v[p]) + 1e-8)   

                else:
                    raise ValueError('Unrecognized update type %s' % update)


                model[p]+=dx

            '''    
                elif update=="rmsprop":
                     
                    self.step_cache[p]=np.zeros(grads[p].shape)
                    decay_rate=0.99 #can keep this as an option
                    if not p in self.step_cache:
                        dx=np.zeros_like(grads[p])
                    dx=grads[p]
                    step_cache[p]=decay_rate*step_cache[p]+(1-decay_rate)*dx**2
                    dx=-learning_rate*dx/(np.sqrt(step_cache+1e-8))
            '''
            print "\n"
            first_it=(it==0)
            epoch_end=(it+1) % iterations_per_epoch==0
            acc_check=(acc_frequency is not None and it % acc_frequency==0)

            if first_it or epoch_end or acc_check:
                if it>0 or epoch_end:
                    learning_rate*=learning_rate_decay
                    epoch+=1

             #evaluate train accuracy

            if N>1000:
                 train_mask=np.random.choice(N,1000)
                 X_train=X[train_mask]
                 y_train=y[train_mask]
            else:
                X_train=X
                y_train=y

            score_train = loss_function(X_train,model,U,U_2,pool_param)
            y_pred_train=np.argmax(score_train,axis=1)
            train_acc=np.mean(y_train==y_pred_train)
            train_acc_history.append(train_acc)
            print ("\nTraining acc is  : %f") % (train_acc)
            # ------------------saving parameters after every 50 iterations ------------------------#
            if (it%50==0):
              file_save_param=open("param_save_50","wb")
              cp.dump(model,file_save_param)
              file_save_param.close()

            score_val=loss_function(X_val,model,U,U_2,pool_param)

            y_pred_val=np.argmax(score_val,axis=1)
            val_acc=np.mean(y_pred_val==y_val)
            val_acc_history.append(val_acc)
            print ("\nValidation accuracy is : %f")  % (val_acc)
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                best_model={}
                for p in model:
                    best_model[p]=model[p].copy()


        if verbose:
           print'\nfinished optimization .best validation accuracy %f' % (best_val_acc,)
       
        return best_model,loss_history, train_acc_history, val_acc_history

     
         
                       
                 
             
    
    
