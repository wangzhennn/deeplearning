#--------------- 1.1 --------------%
#3 layers: input/hidden/output layer
#--------------- 1.2 forward propagation --------------%
#- multiply -> add process
#- dot product
#- forward propagation for one data point at a time
#- output is the prediction for that data point

import numpy as np
input_data-np.array([2,3]) #input layer
weights={'node_0':np.aray([1,1]), #hidden layer
         'node_1':np.aray([-1,1]),
         'output':np.aray([2,-1])}
node_0_value=(input_data*weights['node_0']).sum() #output layer
node_1_value=(input_data*weights['node_1']).sum()
hidden_layer_values=np.array([node_0_value,node_1_value])
print(hidden_layer_values)
output=(hidden_layer_values*weights['output']).sum()
print(output)
