import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/mpaul/projects/mpaul/mai_prep/interview_preparation/topics/deep_learning/dl_concepts/src/insurance_data.csv')
print(df.head())


X = df[['age', 'affordibility']]
y = df['bought_insurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['age'] = X_train['age'] / 100
X_test['age'] = X_test['age'] / 100

_model= False
if _model:
    model = keras.Sequential([
        keras.layers.Dense(1, 
                            input_shape=(2,), 
                            activation='sigmoid',
                            kernel_initializer='ones',
                            bias_initializer='zeros')
    ])

    model.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])

    model.summary()
    model.fit(X_train, y_train, epochs=100)
    model.evaluate(X_test, y_test)

    print(model.predict(X_test))
    print(y_test)

    coeff, bias = model.get_weights()
    print(coeff)
    print(bias)


# ========================= User Defined Prediction Function =========================
def prediction_function(age, affordibility):
    weighted_sum = coef[0]*age + coef[1]*affordibility + intercept
    z = sigmoid(weighted_sum)
    return z

# print(prediction_function(X_test['age'], X_test['affordibility']))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    loss = -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))
    return loss

def gradient_descent(age, affordibility, y_true, epochs, loss_threshold):
    w1 = w2 = 1
    bias = 0
    rate = 0.5
    n = len(age)
    
    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordibility + bias
        y_predicted = sigmoid(weighted_sum)
        loss = log_loss(y_true, y_predicted)
        print(loss)
        w1d = (1/n)*np.dot(np.transpose(age),(y_predicted - y_true)) 
        w2d = (1/n)*np.dot(np.transpose(affordibility),(y_predicted - y_true)) 
        bias_d = np.mean(y_predicted-y_true)
        w1 = w1 - rate * w1d
        w2 = w2 - rate * w2d
        bias = bias - rate * bias_d
        print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
        
        if loss <= loss_threshold:
            print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            break
    return w1, w2, bias

gradient_descent(X_train['age'],X_train['affordibility'],y_train,1000, 0.4631)

# print(model.predict(X_test))

# print(model.get_weights())