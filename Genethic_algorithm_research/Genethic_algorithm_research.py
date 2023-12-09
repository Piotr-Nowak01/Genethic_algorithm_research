
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def GeneratePoints(XminRange=0,XmaxRange=100,YminRange=0,YmaxRange=100,NumberOfPoints=50):
    Points = []
    Matrix = []
    Xdiff=XmaxRange-XminRange+1
    Ydiff=YmaxRange-YminRange+1
    for x in range(NumberOfPoints*2):
        Points.append(0)
    for x in range(Xdiff*Ydiff):
        Matrix.append(0)
    Matrix=np.array(Matrix)
    Matrix=Matrix.reshape(Xdiff,Ydiff)
    Points = np.array(Points)
    Points= Points.reshape(NumberOfPoints,2)
    for i in range(NumberOfPoints):
        x = random.randint(XminRange,XmaxRange)
        y = random.randint(YminRange,YmaxRange)
        while Matrix[x][y]==1:
            x = random.randint(XminRange,XmaxRange)
            y = random.randint(YminRange,YmaxRange)
        Matrix[x][y]=1
        Points[i][0]=x
        Points[i][1]=y
    return Points
def CalculateDistance(Points, FirstIndex, SecondIndex):
    return math.sqrt(pow(Points[FirstIndex][0]-Points[SecondIndex][0],2)+pow(Points[FirstIndex][1]-Points[SecondIndex][1],2))
def VisualizePath(Order,GraphTitle, XminRange=0,XmaxRange=100,YminRange=0,YmaxRange=100):
    margin=3
    plt.xlim(Xminrange - margin, Xmaxrange + margin)
    plt.ylim(Yminrange - margin, Ymaxrange + margin)
    plt.grid()
    x,y = zip(*Order)
    plt.plot(x,y,marker="o",markersize=5,markeredgecolor="black", markerfacecolor="black")
    plt.title(GraphTitle)
    plt.show()
    