'''Creates 2-label data for classification.
Types: circular (inside outside radius) & moon-shaped'''

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
import numpy as np
import random

np.random.seed(0)

cm1 = mcol.LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
cnorm = mcol.Normalize(vmin = 0, vmax = 1)
cpick = cm.ScalarMappable(norm = cnorm, cmap = cm1)
cpick.set_array([])

def rect(distance,angle):
    '''convert polar coordinates to rectangular and translate by (x,y)'''
    return np.array([X + np.cos(angle)*distance, Y + np.sin(angle)*distance])

k = 10 # Circular data radius
fg = None
pointSeperation = 5 # Seperation of data from circle radius
X, Y = -2, 4

ct = 1
noise = 5
pointCount = 100

def createData(data = 'circular'):
    '''data type: 'circular' or 'moons'(scikit) '''
    random.seed(0)
    np.random.seed(0)
    global X, Y
    if data.lower() == 'circular':
        #Creates data of 2 types: inside circle and outside circle
        halfPointCount = pointCount//2
        theta = np.linspace(0, 2*np.pi, halfPointCount)
        aDist = k - pointSeperation + np.random.uniform(-noise, noise, size = halfPointCount)
        bDist = k + pointSeperation + np.random.uniform(-noise, noise, size = halfPointCount)

        A = rect(aDist, theta + np.random.rand(halfPointCount)).T
        B = rect(bDist, theta + np.random.rand(halfPointCount)).T

        allPoints = np.vstack((A,B))
        allLabels = np.array([0]*len(A) + [1]*len(B))
    else:
        from sklearn.datasets import make_moons
        allPoints, allLabels = make_moons(n_samples = pointCount, noise = noise/100)
        allPoints *= 5
        allPoints *= np.array([[2,1]]*len(allPoints))
    
    indices = np.arange(0, len(allPoints), 1)
    np.random.shuffle(indices)

    return allPoints[indices], allLabels[indices]

titles = ['DataSet', 'Predicted(Discrete)', 'Predicted(Actual)', 'Prediction Space']

def plotPoints(p, r, show = True, maxR = 2, maxC = 2):
    global fg
    if not fg:
        fg = plt.figure(figsize=(8, 8))
    global ct
    print('Plotting points')
    ax = fg.add_subplot(maxR, maxC, ct)
    ax.set_title(titles[ct-1])
    ct += 1
    ax.scatter(p[:,0], p[:,1], c = cpick.to_rgba(r))
    if show:
        plt.show()

if __name__ == "__main__":
    dataPoints, dataLabels = createData('circular')
    plotPoints(dataPoints, dataLabels, True, 1, 1)
