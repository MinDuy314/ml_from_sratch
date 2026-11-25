import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-np.pi, np.pi, 100) #create 100 evenly-space number between -pi and pi
ys = np.sin(xs) #Target curve


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

h1,b1,w1 = (0.01576408772634988, 0.8346478258693997, 0.13464782586939533)
h2,w2,b2 = (0.006985773778613827,-1.0930142262213844,3)
h3,w3,b3 = (0.3775885758704721,-1.4929021661605685,0.5)
b_out = 0.0

lr = 0.01


for epoch in range(5000):
    y_preds =[]
    for x,y_true in zip(xs,ys):
        sig1 = sigmoid(w1*x + b1)
        sig2 = sigmoid(w2*x + b2)
        sig3 = sigmoid(w3*x + b3)

        s1 = h1*sig1
        s2 = h2*sig2
        s3 = h3*sig3
        y_pred = s1 + s2 + s3 + b_out    

        error = np.mean(y_pred - y_true)

        h1 -= lr*error*sig1
        h2 -= lr*error*sig2
        h3 -= lr*error*sig3
        blame1 = error * h1 * sig1 * (1 - sig1)
        blame2 = error * h2 * sig2 * (1 - sig2)
        blame3 = error * h3 * sig3 * (1 - sig3)
        w1 -= lr*blame1*x 
        w2 -= lr*blame2*x
        w3 -= lr*blame3*x  
        b1 -= lr*blame1
        b2 -= lr*blame2
        b3 -= lr*blame3
        b_out -= lr*error
        y_preds.append(y_pred)


plt.plot(xs,ys)
plt.plot(xs,y_preds)
plt.show()

