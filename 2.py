import numpy as np

X=np.array([1,2,3,4])
Y=np.array([2,4,6,8])

w=0.0

def forward(x):
    return w*x

def loss(y,y_predicted):
    return ((y-y_predicted)**2).mean()

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate=0.01
n_iters=10

for epoch in range(n_iters):
    y_pred=forward(X)

    l=loss(Y,y_pred)

    dw=gradient(X,Y,y_pred)

    w-= learning_rate*dw

    if epoch%1==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training f(5) = {forward(5):.3f}')