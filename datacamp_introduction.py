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

#--------------- 1.3 activation functions --------------%
#activation functions allow models to capture the non-linear functions

#tanh activation function

#ReLU (Rectified Linear Activation)

import numpy as np
input_data-np.array([2,3])
weights={'node_0':np.aray([3,3]), 
         'node_1':np.aray([1,5]),
         'output':np.aray([2,-1])}
node_0_input=(input_data*weights['node_0']).sum()
node_0_output=np.tanh(node_0_input)
node_1_input=(input_data*weights['node_1']).sum()
node_1_output=np.tanh(node_1_input)
hidden_layer_outputs=np.array([node_0_output,node_1_output])
output=(hidden_layer_output*weights['output']).sum()
print(output)

#=====example=======%
#The Rectified Linear Activation Function
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input) #注意
    # Return the value just calculated
    return(output)
# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input) #如果是tanh，使用np.tach
# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])
# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()
# Print model output
print(model_output)
