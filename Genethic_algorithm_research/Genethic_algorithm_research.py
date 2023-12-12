
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
            winner=participant
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
        crossover_point1=np.random.randint(1,person_size-1)
        crossover_point2=np.random.randint(1,person_size-1)
    if crossover_point1>crossover_point2:
        crossover_point1,crossover_point2=crossover_point2,crossover_point1  
    
    
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
    #print(child)
    #input()
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
    best_individual1=best_individual=population[0]
    Matrix=Matrix_Distances(Order)
    for i in range(number_of_generations):
        print("Working. Current generation: "+str(i+1))
        parents=selection(Matrix,population,number_of_parents,tournament_size)
        childs=[]
        new_kids=[]
        for parent in parents:
            childs.append(crossover(Matrix,parent))
        for child in childs:
            new_kids.append(mutation(child,mutation_probability,mutation_type))
        population=parents+new_kids
        for individual in population:
            if objective_function(Matrix,individual)<objective_function(Matrix,best_individual1):
                best_individual1=individual 
    for individual in population:
        if objective_function(Matrix,individual)<objective_function(Matrix,best_individual):
            best_individual=individual
    return (best_individual,best_individual1)
def NearestNeighbourAlgorithm(Order, starting_point=0):
    matrix = Matrix_Distances(Order)
    Done=1 
    size=len(Order)
    starting_order=[]
    for i in range(size):
        starting_order.append(i)
    final_order =[]
    final_order.append(starting_point)
    currently_do=starting_point
    while Done < size:
        min = 999999
        for point in starting_order:
            if point not in final_order:
                if matrix[currently_do][point]<min:
                    pos=point
                    min=matrix[currently_do][point]
        final_order.append(pos)
        Done+=1
        currently_do=pos
    return final_order
#===========================
def testowanie():
    Order = []
    Order1=[]
    Order2=[]
    Order3=[]
    Points = GeneratePoints()
    population = []
    for i in range(50):
            Order.append((Points[i][0], Points[i][1]))
            population.append(i)
    matrix=Matrix_Distances(Order)
    result=GeneticAlgorithm(Order)
    result1=result[0]
    result2=result[1]
    result3=NearestNeighbourAlgorithm(Order)
    for i in range(len(Order)):
        Order1.append(Order[result1[i]])
        Order2.append(Order[result2[i]])
        Order3.append(Order[result3[i]])
    Order.append(Order[0])
    Order1.append(Order1[0])
    Order2.append(Order2[0])
    Order3.append(Order3[0])
    VisualizePath(Order,"Starting "+str(objective_function(matrix,population)))
    VisualizePath(Order1,"Final, last gen "+str(objective_function(matrix,result1)))
    VisualizePath(Order2,"Final, whole "+str(objective_function(matrix,result2)))
    VisualizePath(Order3,"Nearest Neighbour "+str(objective_function(matrix,result3)))
testowanie()