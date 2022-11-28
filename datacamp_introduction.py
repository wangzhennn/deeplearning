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

#======example======%
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

#======example======%
# Applying the network to many observations/rows of data
# Define predict_with_network()
def predict_with_network(input_data_row, weights):
    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    # Return model output
    return(model_output)
# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))
# Print results
print(results)

#--------------- 1.4 deeper networks --------------%
#modeler doean't need to specify the interactions
#when you train the model, the neural network gets weights that find the relevant patterns to make better predictions

# Multi-layer neural networks
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)
    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)
    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)
    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)
    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()   
    # Return model_output
    return(model_output)
output = predict_with_network(input_data)
print(output)

#--------------- 2.1 the need for optimization --------------%
#the change in weights can improve the model for the data point
#loss function: agregates errors in predictions from many data points into single number. measure of model's predictive performance

# squared error

# gradient descent
#- start at random point
#- until you are somewhere flat: find the slope, take a step downhill

#======Example======%
# Coding how weight changes affect accuracy
# The data point you will make a prediction for
input_data = np.array([0, 3])
# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }
# The actual target value, used to calculate the error
target_actual = 3
# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)
# Calculate error: error_0
error_0 = model_output_0 - target_actual
# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }
# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)
# Calculate error: error_1
error_1 = model_output_1 - target_actual
# Print error_0 and error_1
print(error_0)
print(error_1)

#======Example======%
#Scaling up to multiple data points
from sklearn.metrics import mean_squared_error
# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []
# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))
# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)
# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)
# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)

#--------------- 2.2 Gradient descent --------------%























