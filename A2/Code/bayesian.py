#BAYESIAN CLASSIFIER - Team - 10
#Amogh Patil - EE19B134
#Rishabh Adiga - EE19B135

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve

#Returns the Gausian outputs(of a given mean and covariance matrix) for all data points in a 2D space
#X - feature 1 of vectors in a 2D space
#Y - feature 2 of vectors in a 2D space
def PDF(X,Y,u,E):
    _E = np.linalg.inv(E)
    _X = X - u[0]
    _Y = Y - u[1] 
    exp_arg = -1/2 * (_E[0][0] * _X * _X + (_E[0][1] + _E[1][0]) * _X * _Y + _E[1][1] * _Y * _Y)
    return (1/(2*np.pi*np.sqrt(np.linalg.det(E)))) * np.exp(exp_arg)

#Discrimant function value for a given class for all data points in a 2D space
def discrimant(E,u,P,_X,_Y):
    E_inv = np.linalg.inv(E)
    W = 1/2 * E_inv
    w = np.dot(E_inv,u)
    w0 = -1/2*np.dot(np.dot(u.T,E_inv),u) - 1/2 * np.log(np.linalg.det(E)) + np.log(P)
    g_x = -(W[0][0] * _X * _X + (W[0][1] + W[1][0]) * _X * _Y + W[1][1] * _Y * _Y) + (w[0] * _X + w[1] * _Y) + w0
    return g_x

#Returns the Mean vector
def mean(x):
    return sum(x)/len(x)

#Returns the Covariance Matrix
def covariance(x,mean):
    out = np.zeros((len(x[0]),len(x[0])))
    for v in x:
        vec = (v-mean)
        vec = vec.reshape((len(vec),1))
        out += np.dot((vec),vec.T)
    return 1/(len(x)-1) * out

#For Case 3, finding a sigma that represents a covariance matrix
def sigma(x,mean):
    out = np.sum((x-mean)*(x-mean))
    return 1/(len(x)-1) * out

#Classifies the test data
#Discrimant value is found for each input vector for all three classes
#The input vector is classified into the class having the largest Discriminant Value
def classification(means,covs,tst_data,P):
    X,Y,T = tst_data[:,0],tst_data[:,1],tst_data[:,2]
    Z1 = discrimant(covs[0],means[0],P[0],X,Y)
    Z2 = discrimant(covs[1],means[1],P[1],X,Y)
    Z3 = discrimant(covs[2],means[2],P[2],X,Y)
    classify = np.array([Z1,Z2,Z3]).argmax(0)
    return(classify,(len(T)-np.count_nonzero(classify-T+1))/len(T) * 100)

#Plots the PDF, Decision Boundary, Contour Plots, Confusion Matrix
def plots(means,covs,class1,class2,class3,data,tst_data,P,Title,mins,maxs,flag=0):
    #Finding the range of train data for plotting
    
    #Defining the 3D space
    n = 350
    x1 = np.linspace(mins[0],maxs[0],n)
    x2 = np.linspace(mins[1],maxs[1],n)
    X,Y = np.meshgrid(x1,x2)

    #Plotting Probability Density functions for various Classes
    plt.figure(figsize=(6,6))
    Z1 = PDF(X,Y,means[0],covs[0])      #P(x|C1)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z1,label="Class 1",color='red',edgecolor='None')

    Z2 = PDF(X,Y,means[1],covs[1])      #P(x|C2)
    ax.plot_surface(X, Y, Z2,label="Class 2",color='blue',edgecolor='None')

    Z3 = PDF(X,Y,means[2],covs[2])      #P(x|C3)
    ax.plot_surface(X, Y, Z3,label="Class 3",color='green',edgecolor='None')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("PDF")
    ax.set_title(Title)
    plt.show()

    #Contour Plots
    plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.contour(X,Y,Z1)
    ax.contour(X,Y,Z2)
    ax.contour(X,Y,Z3)

    
    #Decision Boundary Plot
    Z1 = Z1 * P[0]      #P(C1|x)
    Z2 = Z2 * P[1]      #P(C2|x)
    Z3 = Z3 * P[2]      #P(C3|x)
    classify = 1+np.array([Z1,Z2,Z3]).argmax(0)
    
    c = ax.contourf(X, Y, classify,cmap='viridis',levels=[0,1,2,3])
    plt.colorbar(c)
    ax.set_title(Title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    #Scatter Plot
    if flag == 0:       #For train Data
        ax.scatter(class1[:,0],class1[:,1],facecolors='none',edgecolors='r',label='class1')
        ax.scatter(class2[:,0],class2[:,1],facecolors='none',edgecolors='b',label='class2')
        ax.scatter(class3[:,0],class3[:,1],facecolors='none',edgecolors='g',label='class3')
    elif flag == 1:     #For Development Data/Test Data, To visualise how well the model is doing
        tst_class1,tst_class2,tst_class3=[],[],[]
        for i in range(len(tst_data)):
            if tst_data[i][2]==1:
                tst_class1.append(tst_data[i][:2])
            if tst_data[i][2]==2:
                tst_class2.append(tst_data[i][:2])
            if tst_data[i][2]==3:
                tst_class3.append(tst_data[i][:2])
        tst_class1=np.array(tst_class1)
        tst_class2=np.array(tst_class2)
        tst_class3=np.array(tst_class3)
        ax.scatter(tst_class1[:,0],tst_class1[:,1],facecolors='none',edgecolors='r',label='class1')
        ax.scatter(tst_class2[:,0],tst_class2[:,1],facecolors='none',edgecolors='b',label='class2')
        ax.scatter(tst_class3[:,0],tst_class3[:,1],facecolors='none',edgecolors='g',label='class3')
    plt.legend(loc ="lower right")

    #Eigen Vectors
    e,V = np.linalg.eig(covs[0])
    V = V*e
    u = means[0]
    origin = np.array([u,u]).T
    plt.quiver(*origin,*V,angles='xy',color=['black'],scale_units='xy', scale=1, linewidths=0.25)

    e,V = np.linalg.eig(covs[1])
    V = V*e
    u = means[1]
    origin = np.array([u,u]).T
    plt.quiver(*origin,*V,angles='xy',color=['black'],scale_units='xy', scale=1, linewidths=0.25)

    e,V = np.linalg.eig(covs[2])
    V = V*e
    u = means[2]
    origin = np.array([u,u]).T
    plt.quiver(*origin,*V,angles='xy',color=['black'],scale_units='xy', scale=1, linewidths=0.25)
    plt.axis('scaled')
    plt.show()

    #CONFUSION Matrix
    classify,accuracy = classification(means,covs,tst_data,P)
    conf_matrix = np.zeros((3,3))
    for i in range(len(tst_data)):
        conf_matrix[int(tst_data[i][2]-1)][classify[i]] += 1
    #print(conf_matrix)
    #Plotting the Confusion Matrix
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(Title, fontsize=12)
    plt.show()

    #Displaying Accuracy of the Model on Test Data
    print(Title," on Test Data :",accuracy,"%")

#Receiver Operating Characteristic Curve and Detection Error Tradeoff Curve
def ROC_DET(means,cov,tst_data,P):
    #ROC
    S_list = []
    plt.figure(figsize=(5,5))
    for case_no in range(1,6):  #For Loop for all cases
        covs = cov[case_no-1]  
        S = []
        S.append(PDF(tst_data[:,0],tst_data[:,1],means[0],covs[0]) * P[0])
        S.append(PDF(tst_data[:,0],tst_data[:,1],means[1],covs[1]) * P[1])
        S.append(PDF(tst_data[:,0],tst_data[:,1],means[2],covs[2]) * P[2])
        S = np.array(S)
        S_list.append(S.flatten())

        S = S.T
        Scores = sorted(S.flatten())   #Scores are sorted for thresholding
        
        TPR = [0]*len(Scores)   
        FPR = [0]*len(Scores) 
        count=0
        for threshold in Scores:
            TP,FP,TN,FN = 0,0,0,0
            for i in range(len(tst_data)):
                for j in range(3):
                    if S[i][j] >= threshold:        #Classifying As Positive
                        if tst_data[i][2] == j+1: TP+=1
                        else:   FP+=1
                    else:
                        if tst_data[i][2] == j+1: FN+=1
                        else:   TN+=1
            TPR[count] = TP/(TP+FN)     #True Positive Rate
            FPR[count] = FP/(FP+TN)     #False Positive Rate

            count+=1
        plt.plot(FPR,TPR,label="Case "+str(case_no))
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic curve")
    plt.legend()
    plt.show()

    
    #DET
    #Makes use of an inbuilt library - sklearn.metrics.det_curve
    #y_true - 1D array of length(no of class * len(dev_data))
    y_true = []
    for i in range(3):
        for j in range(len(tst_data)):
            if tst_data[j][2] == i+1:
                y_true.append(1)
            else:
                y_true.append(0)

    plt.figure(figsize=(8,5))    
    for i in range(5):
        S = S_list[i]
        y_true = np.array(y_true)
        fpr, fnr, thresholds = det_curve(y_true, S)
        plt.plot(fpr,fnr,label="Case "+str(i+1))
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve")
    plt.show()
  
#Calcuting Covariance Matrix for Case 1 and Case 4
def case_1_4(data,tst_data,class1,class2,class3,means,P):
    # CASE 1
    # Unbiased Average of the Covariance Matrices for different Classes
    cov1 = covariance(class1[:,:2],means[0])
    cov2 = covariance(class2[:,:2],means[1])
    cov3 = covariance(class3[:,:2],means[2])
    cov = 1/(len(data)-3) * (cov1 * (len(class1)-1) + cov2 * (len(class2)-1) + cov3 * (len(class3)-1))
    covs1 = [cov,cov,cov]

    # CASE 4
    # Naive bayes - Nulling the non-diagonal elements
    cov = np.diag(np.diag(cov))
    covs2 = [cov,cov,cov]

    return(covs1,covs2)

#Calcuting Covariance Matrix for Case 3
def case_3(data,tst_data,class1,class2,class3,means,P):
    cov1 = sigma(class1[:,:2],means[0]) * np.diag([1,1])
    cov2 = sigma(class2[:,:2],means[1]) * np.diag([1,1])
    cov3 = sigma(class3[:,:2],means[2]) * np.diag([1,1])
    covs = [cov1,cov2,cov3]
    
    return(covs)


#Calcuting Covariance Matrix for Case 2 and Case 5
def case_2_5(data,tst_data,class1,class2,class3,means,P):
    #CASE 2
    cov1 = covariance(class1[:,:2],means[0])
    cov2 = covariance(class2[:,:2],means[1])
    cov3 = covariance(class3[:,:2],means[2]) 
    covs1 = [cov1,cov2,cov3]

    #CASE 5
    # Naive bayes - Nulling the non-diagonal elements
    cov1 = np.diag(np.diag(cov1))
    cov2 = np.diag(np.diag(cov2))
    cov3 = np.diag(np.diag(cov3))
    covs2 = [cov1,cov2,cov3]

    return(covs1,covs2)

def main(data,tst_data,name_dataSet):
    #Segregating the Training data into respective Classes
    class1,class2,class3=[],[],[]
    for i in range(len(data)):
        if data[i][2]==1:
            class1.append(data[i][:2])
        if data[i][2]==2:
            class2.append(data[i][:2])
        if data[i][2]==3:
            class3.append(data[i][:2])
    class1=np.array(class1)
    class2=np.array(class2)
    class3=np.array(class3)

    #Calculating Mean vectors for each class
    mean1 = mean(class1[:,:2])
    mean2 = mean(class2[:,:2])
    mean3 = mean(class3[:,:2])
    means = [mean1,mean2,mean3]

    #Probability of each Class
    P1 = len(class1)/len(data)
    P2 = len(class2)/len(data)
    P3 = len(class3)/len(data)
    P = [P1,P2,P3]
    
    #Calculating the Covariance Matrices of eeach class for each Case
    cov = [[]]*5
    cov[2]=case_3(data,tst_data,class1,class2,class3,means,P)
    cov[0],cov[3]=case_1_4(data,tst_data,class1,class2,class3,means,P)
    cov[1],cov[4]=case_2_5(data,tst_data,class1,class2,class3,means,P)

    #Finds the range of values of each feature
    mins = [min(min(data[:,0])-3,mean1[0]-4*np.sqrt(cov[1][0][0][0]),mean2[0]-4*np.sqrt(cov[1][1][0][0]),mean3[0]-4*np.sqrt(cov[1][2][0][0])),min(min(data[:,1])-3,mean1[1]-4*np.sqrt(cov[1][0][1][1]),mean2[1]-4*np.sqrt(cov[1][1][1][1]),mean3[1]-4*np.sqrt(cov[1][2][1][1]))]
    maxs = [max(max(data[:,0])+3,mean1[0]+4*np.sqrt(cov[1][0][0][0]),mean2[0]+4*np.sqrt(cov[1][1][0][0]),mean3[0]+4*np.sqrt(cov[1][2][0][0])),max(max(data[:,1])+3,mean1[1]+4*np.sqrt(cov[1][0][1][1]),mean2[1]+4*np.sqrt(cov[1][1][1][1]),mean3[1]+4*np.sqrt(cov[1][2][1][1]))]

    #Plotting of the PDF for each CASE
    Title = "Case 1: Bayes with Covariance same for all classes " + name_dataSet
    plots(means,cov[0],class1,class2,class3,data,tst_data,P,Title,mins,maxs)
    Title = "Case 2: Bayes with Covariance different for all classes " + name_dataSet
    plots(means,cov[1],class1,class2,class3,data,tst_data,P,Title,mins,maxs)
    Title = "Case 3: Naive Bayes with C = "+r'$\sigma^2$'+"I " + name_dataSet
    plots(means,cov[2],class1,class2,class3,data,tst_data,P,Title,mins,maxs)
    Title = "Case 4: Naive Bayes with C same for all classes " + name_dataSet
    plots(means,cov[3],class1,class2,class3,data,tst_data,P,Title,mins,maxs)
    Title = "Case 5: Naive Bayes with C different for all classes " + name_dataSet
    plots(means,cov[4],class1,class2,class3,data,tst_data,P,Title,mins,maxs)
    
    ROC_DET(means,cov,tst_data,P)
    print()

#Reading Data from files
def Linear_Sep_Data():
    with open("Data/LinearlySeperable/trian.txt") as f:
        train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Data/LinearlySeperable/dev.txt") as f:
        tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
    tst_data = np.array(tst_data)
    main(train_data,tst_data,"(Lin_Sep)")

#Reading Data from files
def Non_Linear_Sep_Data():
    with open("Data/NonLinearlySeperable/trian.txt") as f:
        train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Data/NonLinearlySeperable/dev.txt") as f:
        tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
    tst_data = np.array(tst_data)
    main(train_data,tst_data,"(Non_Lin_Sep)")

#Reading Data from files
def Real_Data():
    with open("Data/RealData/trian.txt") as f:
        train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Data/RealData/dev.txt") as f:
        tst_data = [[float(val) for val in line.strip().split(',')] for line in f]
    tst_data = np.array(tst_data)
    main(train_data,tst_data,"(Real)")

if __name__ == "__main__":
    Linear_Sep_Data()
    Non_Linear_Sep_Data()
    Real_Data()