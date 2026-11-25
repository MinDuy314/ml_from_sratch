import numpy as np

#hidden line y = 4x^2 + 3x + 1 / At the beginning i use y = 3x+2
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [6.0, 19.0, 40.0, 69.0, 106.0]

g = 3.9966059492109958 #for a linear function, i dont need a third weight
w = 1.0228961534476646 #First ly i set it to 0 and keep update with the last w and b value of the last epoch
b = 0.9685288608734758
lr = 0.0001

for epoch in range (9000):
    for x,y_true in zip(xs,ys):
        y_pred =  g*x**2 + w * x + b

        error = y_pred - y_true
        g = g - error * lr * x**2    
        w = w - error * lr * x    
        b = b - error *lr
print(f"setup: lr = {lr},epochs = {epoch}")
print(f"error: {error},(g,w,b):{g},{w},{b}")


