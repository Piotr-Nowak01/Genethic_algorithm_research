
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
    plt.xlim(XminRange - margin, XmaxRange + margin)
    plt.ylim(YminRange - margin, YmaxRange + margin)
    plt.grid()
    x,y = zip(*Order)
    plt.plot(x,y,marker="o",markersize=5,markeredgecolor="black", markerfacecolor="black")
    plt.title(GraphTitle)
    plt.show()
def Matrix_Distances(Order):
    size=len(Order)
    Matrix=[]
    for i in range(size*size):
        Matrix.append(0)
    Matrix=np.array(Matrix)
    Matrix=Matrix.reshape((size,size))
    for k in range(size):
        for j in range(size):
            if j==k:
                Matrix[k][j]=0.0
            elif Matrix[k][j]==0.0:
                Matrix[k][j]=CalculateDistance(Order,k,j)
                Matrix[j][k]=Matrix[k][j]
    return Matrix
def objective_function(Matrix,population):
    result=0.0
    size=len(Matrix)-1
    for point in range(size):
        result+=Matrix[population[point]][population[point+1]]
    result+=Matrix[population[-1]][population[0]]
    return result
def starting_population(Order,population_size=50):
    number_of_cities=len(Order)
    population=[]
    for subject in range(population_size):
        route=np.random.permutation(number_of_cities)
        population.append(route)
    return population
def tournament_selection(Matrix,population,tournament_size=5):
    participants=random.sample(population,tournament_size)
    winner=participants[0]
    for participant in participants:
        if objective_function(Matrix,participant)<objective_function(Matrix,winner):
            winner=participants
    return winner
def selection(Matrix,population,number_of_parents,tournament_size=5):
    parents = []
    for i in range(number_of_parents):
        parents.append(tournament_selection(Matrix,population,tournament_size))
    return parents
def crossover(Matrix,parent):
    person_size=len(parent)
    crossover_point1=crossover_point2=0
    while crossover_point1==crossover_point2:
        crossover_point1=np.random.randint(0,person_size)
        crossover_point2=np.random.randint(0,person_size)
    child = []
    PointsDone = []
    iter=crossover_point1-1
    for _ in range(person_size):
        child.append(0)
    for i in range(crossover_point1,crossover_point2):
        child[i]=parent[i]
        PointsDone.append(parent[i])
    while iter>=0:
        min=999999
        for point in range(person_size):
            if point not in PointsDone:
                if Matrix[child[iter+1]][point]<min:
                    pos=point
                    min=Matrix[child[iter+1]][point]
        child[iter]=pos
        PointsDone.append(pos)
        iter-=1
    iter=crossover_point2
    while iter<person_size:
        min=999999
        for point in range(person_size):
            if point not in PointsDone:
                if Matrix[child[iter-1]][point]<min:
                    pos=point
                    min=Matrix[child[iter-1]][point]
        child[iter]=pos
        PointsDone.append(pos)
        iter+=1
    return child
def mutation(child,mutation_probability=25,mutation_type=1):
    mutate = np.random.randint(0,100)
    if mutate<mutation_probability:
        point1=point2=0
        person_size=len(child)
        if mutation_type==1:
            while point1==point2:
                point1=np.random.randint(0,person_size)
                point2=np.random.randint(0,person_size)
            child[point1],child[point2]=child[point2],child[point1]
    return child
def GeneticAlgorithm(Order,number_of_generations=1000,tournament_size=5,population_size=50,number_of_parents=25,mutation_probability=25,mutation_type=1):
    if population_size!=50:
        number_of_parents=population_size/2
    population=starting_population(Order,population_size)
    best_individual=population[0]
    Matrix=Matrix_Distances(Order)
    for i in range(number_of_generations):
        print(len(population))
        parents=selection(Matrix,population,tournament_size,number_of_parents)
        childs=[]
        new_kids=[]
        for parent in parents:
            childs.append=crossover(Matrix,parent)
        for child in childs:
            new_kids.append(mutation(child,mutation_probability,mutation_type))
        population=parents+new_kids
    for individual in population:
        if objective_function(Matrix,individual)<objective_function(Matrix,best_individual):
            best_individual=individual
    return best_individual
#===========================
Order = []
Order1=[]
Points = GeneratePoints()
population = []
for i in range(50):
        Order.append((Points[i][0], Points[i][1]))
        population.append(i)
matrix=Matrix_Distances(Order)
VisualizePath(Order,"Starting "+str(objective_function(matrix,population)))
result=GeneticAlgorithm(Order)
for i in range(len(Order)):
    Order1.append(Order[result[i]])
matrix=Matrix_Distances(Order)
print(len(result))
VisualizePath(Order1,"Final "+str(objective_function(matrix,result)))