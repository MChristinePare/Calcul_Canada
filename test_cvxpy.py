import numpy as np 
import cvxpy as cp 
#import random 
#import pdb 

def layer(inputs, seq_length): 

    U_0=np.random.rand(13,2)
    W_0=np.random.rand(2,2)
    D_2_0=np.random.rand(13,2)
    b_0=np.random.rand(1,2)

    alpha=0.5

    Z=[]

    z_t=None
    X_t_1=None
    for t in range(seq_length): 

            #Select inputs for timestep t. shape (batch_size, input_shape)
            X_t=cp.reshape(inputs[t], shape=(1, 13))
            #Compute hidden state at t. 
            
            reluinput=X_t@U_0 + (z_t@W_0 if z_t is not None else 0) + (X_t_1@D_2_0 if X_t_1 is not None else 0) + (b_0 if b_0 is not None else 0) 
           

            zeros_n=reluinput*alpha
            relu_inter=cp.vstack([reluinput,zeros_n])
            relu=cp.max(relu_inter,axis=0)
            z_t=cp.reshape(relu, shape=(1, 2))
    
            Z.append(z_t)

            #save previous inputs 
            X_t_1=X_t

            #output shape : (seq_length, batch_size, nhidden_units) 
            
    return Z

#Data 
seq_length=36 

#Variables 
inputs={}
for t in range(seq_length):
    inputs[t]=cp.Variable((1,13))


var=layer(inputs, seq_length)

#constraints 
constraints=[]


for t in range(15):
     constraints+=[var[t] <= 100] 

obj=0

prob=cp.Problem(cp.Minimize(obj),constraints)    
prob.solve(solver="GUROBI", verbose=True)


