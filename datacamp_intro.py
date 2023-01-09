#--------------- 1.1 --------------%
#neural networks acount for interactions really well
#3 layers: input/hidden/output layer
#--------------- 1.2   --------------%
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
#activation functions: allow models to *capture the non-linear functions*
#激活函数的作用就是把输入节点的加权和转化后输出
#相反的是identity function

#tanh activation function: s-shaped function: 双曲正切函数

#ReLU (Rectified Linear Activation):神经网络激活函数

import numpy as np
input_data-np.array([2,3])
weights={'node_0':np.aray([3,3]), 
         'node_1':np.aray([1,5]),
         'output':np.aray([2,-1])}
node_0_input=(input_data*weights['node_0']).sum()
node_0_output=np.tanh(node_0_input) #不同点在于这里，使用了激活函数对input进行转化
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
#representation learning: deep networks internally build representations of patterns in the data; partially replace the need for feature engineering; subsequent layers build increasingly sophisticated representations of raw data

#DL
#modeler doesn't need to specify the interactions
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
#loss function: aggregates errors in predictions from many data points into single number; measure of model's predictive performance; lower loss function value means a better model; the goal is to find the weights that give the lowest value for the loss function

# squared error

# gradient descent梯度下降
#- start at random point
#- draw the tangent line to the curve at the current point-->derivative
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
#if the slope is positive
#going opposite the slope means moving to lower numbers
#subtract the slop from the current value
#too big a steop might lead us astray

#solution: [learning rate]--update each weight by subtracting learning rate*slope

#💡slope calculation, need to multiply:
#- slope of the loss function w.r.t value at teh node we feed into --> slope of mean-squared loss function w.r.t prediction=2*(predicted value -actual value)=2*error
#- the value of the node that feeds into our weight
#- slope of the activition function w.r.t value we feed into

#code to calculate slopes and update weights
import numpy as np
weights = np.arrary([1,2])
input_data=np.array([3,4])
target=6
learning_rate=0.01
preds=(weights*input_data).sum()
error=preds-target
print(error)
-----
gradient=2*input_data*error
gradient
-----
weights_updated=weights-learning_rate*gradient
preds_updates=(weights_updated*input_data).sum()
error_updated=pred_updated-target
print(error_updated)

#===example===%
n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

#--------------- 2.3/2.4 backpropagation --------------%
#to calculate the slopes we need to optimize more complex DL models
#allows gradient descent to update all weights in neural network (by gettting gradients for all weights)
#comes from chain rule of calculus
#important to understand the process, but you will generally use a library that implements this

#💡backpropagation process
#trying to estimate the slope of the loss function w.r.t each weight
#do forward propagation to calculate predictions and errors
#use backward propagation to calculate the slope of the loss function with respect to each weight
#gradients for weight is product of: 
         # 1) node value feeding into that weight
         # 2) slope of activation function for the node being fed into
         # 3) slope of loss function w.r.t output node
#multiply the slope by the learning rate and subtract that from the current weights
#keep going with the cycle until we get to a flat part

#💡stochastic gradient descent
#it is common to calculate slopes on only a subset of the data (a batch)
#use a different batch of data to calculate the next update
#start over from the beginning once all data is used
#each time through the training data is called an epoch
#when slopes are calculated on one batch at a time: stochastic gradient descent

#--------------- 3.1 creating a Keras model --------------%
#model building steps
# 1) specify architecture
# 2) compile the model: specify the loss function and some details about how optimization works
# 3) fit the model
# 4) use the model for prediction

# 1) model specification
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

predictors=np.loadtxt('prefictors_data.csv',delimiter=',')
n_cols=predictors.shape[1]

model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
#which means it has n_cols items in each row of data, and any number of rows of data are acceptable as inputs
model.add(Dense(100,activation='relu'))
model.add(Dense(1))


#dense layer: all of the nodes in the previous layer connect to all of the nodes in the current layer

#===example===%
# Import necessary modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50,activation='relu',input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32,activation='relu'))

# Add the output layer
model.add(Dense(1))

#--------------- 3.2 compiling and fitting a model --------------%
#why need to compile the model
#-specify the optimizer: many options and mathematically complex, 'Adam' is usually a good choice
#-loss function: 'mean_squared_error' common for regression

n_cols=predictors.shape[1]
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')

#compiling and fitting a model
# 1) applying backpropagation and gradient descent with your data to update the weights
# 2) scaling data before fitting can ease optimization

n_cols=predictors.shape[1]
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(predictors,target)

#--------------- 3.3 classification models --------------%

#‘categorical_crossentropy' loss function
#similar to log loss: lower is better
#add metrics=['accuracy'] to compile step for easy-to-understand diagnostics
#output layer has separate node for each possible outcome and uses 'softmax' activation

from tensorflow.keras.utils import to_categorical

data=pd.read_csv('basketball_shot_log.csv')
predictors=data.drop(['shot_result'],axis=1).values
target=to_categorical(data['shot_result'])

model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(predictors,target)

#--------------- 3.4 using models --------------%

#saving, reloading and using the model
from tensorflow.models import load_model
model.save('model_file.h5')
my_model=load_model('model_file.h5')
predictions=my_model.predict(data_to_predict_with)
probability_ture=predictions[:,1]
my_model.summary()












