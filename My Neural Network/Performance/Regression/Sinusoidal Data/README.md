# Regression on sinusoidal data

  Data includes f(x) = sin(x/10) from x = -50 to x = 30 (200 pts)

## Files:  
#### - Final Predicted Curve.png:  
Shows the Neural Network's predicted curve after training.

#### - RMSE Graph.png:  
Shows RMSE recorded throughout training. (numbers on x-axis represent the iteration at which RMSE is recorded, not epoch)

#### - Training.wmv:
Shows the Neural Network's predicted curve throughout training.

## Hyperparameters:  
- Architecture: (1,4,4,2,1)
- Activation Function: Hidden - tanh, Output - tanh
- Epochs: 25000
- Learning Rate: 0.005
- Momentum Enabled? : True
- Momentum Rate(beta): 0.9
- Batc Size: 10
