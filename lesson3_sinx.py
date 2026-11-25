import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-np.pi, np.pi, 100) #create 100 evenly-space number between -pi and pi
ys = np.sin(xs) #Target curve


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

h1,h2,h3 = np.random.randn(3) #create an array of 3 random float number  
w1,w2,w3 = 1.0,4.0,5.0
b1,b2,b3 = 0.3,1.5,-1.0

y_preds = []
lr = 0.01

for epoch in range(100):
    for x,y_true in zip(xs,ys):

        s1 = h1*sigmoid(w1*x + b1)
        s2 = h2*sigmoid(w2*x + b2)
        s3 = h3*sigmoid(w3*x + b3)
        y_pred = s1+s2+s3

        error = np.mean(y_pred - y_true)

        h1 = np.mean(h1 -lr*error)
        h2 = np.mean(h2 -lr*error)
        h3 = np.mean(h3 -lr*error)
        w1 = np.mean(w1 -lr*error)
        w2 = np.mean(w2 -lr*error)
        w3 = np.mean(w3 -lr*error)
        b1 = np.mean(b1 -lr*error)
        b2 = np.mean(b2 -lr*error)
        b3 = np.mean(b3 -lr*error)
    y_preds.append(y_pred)


plt.plot(xs,ys)
plt.plot(xs,y_preds)
plt.show()

