# Binary Classification on scikit's make_moons dataset

Dataset includes points belonging to one of two classes that are in the shape of 2 arcs(/'moons') in opposite directions.  
Data noise can be adjusted along with number of data points. Scikit Docs: [sklearn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)  

## Files:  
#### - Dataset and Predictions.png:  
Visualisation of dataset, along with Neural Network's predictions for points in the dataset, and general prediction for points around the input space.

#### - RMSE Graph.png:  
Shows RMSE recorded throughout training. (numbers on x-axis represent the iteration at which RMSE is recorded, not epoch)

#### - Result Accuracy Graph.png:
Shows percentage of points correctly classified throughout training, for both training and validation data. (numbers on x-axis represent the iteration at which RMSE is recorded, not epoch)

#### - Training.wmv:
Shows the Neural Network's changes in prediction for points around the input space.

## Hyperparameters:  
- Architecture: (2,4,3,1)
- Activation Function: Hidden - ReLu, Output - atanScaled
- Epochs: 1000
- Learning Rate: 0.02
- Momentum Enabled? : True
- Momentum Rate(beta): 0.9
- Batc Size: 7
