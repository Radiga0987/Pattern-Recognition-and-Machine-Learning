#REGRESSION MODELS,Team 10
#Amogh Patil - EE19B134
#Rishabh Adiga - EE19B135

#Importing required librarires
import matplotlib.pyplot as plt
import numpy as np

#Loading the data
data_1d_train=np.loadtxt('Data/1d_team_10_train.txt')
data_1d_dev=np.loadtxt('Data/1d_team_10_dev.txt')

data_2d_train=np.loadtxt('Data/2d_team_10_train.txt')
data_2d_dev=np.loadtxt('Data/2d_team_10_dev.txt')

#Function for finding weights for polynomial of order m using normal equation
def poly_reg_1D(data,m,lamda=0):
    x=data[:,0]
    y=data[:,1]
    phiT=[]
    #Phi matrix is constructed
    for i in range(m+1):
        phiT.append(x**i)

    #Normal equation used below to obtain best weights for the polynomial
    phiT_phi=np.dot(phiT,np.transpose(phiT)) + np.diag([lamda]*(m+1))
    weights=np.linalg.solve(phiT_phi,np.dot(phiT,y))

    return weights

def poly_reg_2D(data,n,lamda=0):
    x1=data[:,0]
    x2=data[:,1]
    y=data[:,2]
    phiT=[]
    #Phi matrix is constructed
    for count in range(n+1):
        for i in range(count+1):
            phiT.append((x1**(count-i))*(x2**i))

    #Normal equation used below to obtain best weights for the polynomial
    phiT_phi=np.dot(phiT,np.transpose(phiT))
    inv=np.linalg.inv(phiT_phi + np.diag([lamda]*len(phiT_phi)))
    w=np.dot(np.dot(inv,phiT),y)

    #Weights put into 2D matrix for ease of use
    c=np.zeros((n+1,n+1))
    len_w=0
    for count in range(n+1):
        for i in range(count+1):
            if len_w<len(w):
                c[count-i][i]=w[len_w]
                len_w+=1
                
            else:
                break
    return c

x=data_1d_train[:,0]
y=data_1d_train[:,1]

#Regression on 1D dataset
#Plot of the approximated functions obtained using training datasets of different sizes
w_n=[]
data_1d_n=[]  #This list will contain datasets with different sizes for experiments

for i in [10,20,50,200]:    #Data is sampled uniformly into sets of different sizes
    random_indices = np.random.choice(data_1d_train.shape[0], size=i, replace=False)
    data_1d_n.append(data_1d_train[random_indices, :])
    w_n.append(poly_reg_1D(data_1d_n[-1],7))   #Weights for these different dataset sizes are obtained

def plot_1D_different_n():
    x_=np.linspace(0.05, 4.9,num=1000)
    #We get the predictions from different models constructed above for different dataset sizes
    y1=np.poly1d(w_n[0][::-1])(x_)
    y2=np.poly1d(w_n[1][::-1])(x_)
    y3=np.poly1d(w_n[2][::-1])(x_)
    y4=np.poly1d(w_n[3][::-1])(x_)

    #Plotting the prediction curves for different models and their corresponding training data
    figure, axis = plt.subplots(2, 2,figsize=(10,7))
    for ax in axis.flatten():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axis[0, 0].plot(x_, y1)
    axis[0, 0].scatter(data_1d_n[0][:,0],data_1d_n[0][:,1],s=25,color='none',edgecolor='red')
    axis[0, 0].set_title("$N=10$")
    axis[0, 1].plot(x_, y2)
    axis[0, 1].scatter(data_1d_n[1][:,0],data_1d_n[1][:,1],s=25,color='none',edgecolor='red')
    axis[0, 1].set_title("$N=20$")
    axis[1, 0].plot(x_, y3)
    axis[1, 0].scatter(data_1d_n[2][:,0],data_1d_n[2][:,1],s=25,color='none',edgecolor='red')
    axis[1, 0].set_title("$N=50$")
    axis[1, 1].plot(x_, y3)
    axis[1, 1].scatter(data_1d_n[3][:,0],data_1d_n[3][:,1],s=25,color='none',edgecolor='red')
    axis[1, 1].set_title("$N=200$")
    figure.tight_layout()
    figure.suptitle("1D dataset: Models for varying dataset sizes")
    figure.subplots_adjust(top=0.91)
    plt.show()



#Plot of the approximated functions obtained for different model complexities
def plot_1D_different_m():
    data=data_1d_n[2]     #Here we just take the dataset with number of data points as 50
    w_m=[]                #And observe the models characteristics for different model complexities
    for i in [0,1,4,10]:  #Weights for the different models are stored
        w_m.append(poly_reg_1D(data,i))  #Data for all models is fixed

    x_=np.linspace(0, 4.9,num=1000)
    y0=np.poly1d(w_m[0][::-1])(x_)
    y1=np.poly1d(w_m[1][::-1])(x_)
    y2=np.poly1d(w_m[2][::-1])(x_)
    y3=np.poly1d(w_m[3][::-1])(x_)

    #Plotting the prediction curves for the models with different complexity(polynomial order)
    figure, axis = plt.subplots(2, 2,figsize=(10,7))
    for ax in axis.flatten():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axis[0, 0].plot(x_, y0)
    axis[0, 0].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[0, 0].set_title("$M=0$")
    axis[0, 1].plot(x_, y1)
    axis[0, 1].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[0, 1].set_title("$M=1$")
    axis[1, 0].plot(x_, y2)
    axis[1, 0].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[1, 0].set_title("$M=4$")
    axis[1, 1].plot(x_, y3)    
    axis[1, 1].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[1, 1].set_title("$M=10$")
    figure.tight_layout()
    figure.suptitle("1D dataset: Models for varying model complexity but fixed dataset")
    figure.subplots_adjust(top=0.91)
    plt.show()



#Plot of the approximated functions obtained for different values of regularization parameter
def plot_1D_different_lambda():
    data=data_1d_n[2]       #Here we just take the dataset with number of data points as 50
    w_l=[]                  #And observe the models characteristics for different regularisation values
    for i in [0,1e-8,1,10]: #Weights for the different models are stored
        w_l.append(poly_reg_1D(data,10,i))  #Data and polynomial order are fixed

    x_=np.linspace(0.05, 4.88,num=1000)
    y1=np.poly1d(w_l[0][::-1])(x_)
    y2=np.poly1d(w_l[1][::-1])(x_)
    y3=np.poly1d(w_l[2][::-1])(x_)
    y4=np.poly1d(w_l[3][::-1])(x_)

    #Plotting the prediction curves for the models with different regularisation
    figure, axis = plt.subplots(2,2,figsize=(10,7))
    for ax in axis.flatten():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axis[0, 0].plot(x_, y1)
    axis[0, 0].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[0, 0].set_title(r'$\lambda$= 0')
    axis[0, 1].plot(x_, y2)
    axis[0, 1].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[0, 1].set_title(r'$\lambda$= 1e-8')
    axis[1, 0].plot(x_, y3)
    axis[1, 0].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[1, 0].set_title(r'$\lambda$= 1')
    axis[1, 1].plot(x_, y4)
    axis[1, 1].scatter(data[:,0],data[:,1],s=25,color='none',edgecolor='red')
    axis[1, 1].set_title(r'$\lambda$= 10')
    figure.tight_layout()
    figure.suptitle("1D dataset: Models for varying "+ r"$\lambda$"+" but fixed dataset and order")
    figure.subplots_adjust(top=0.91)
    plt.show()

#Finding the best model for the 1D dataset for which we will see the error on development and train data
def rms_error_1D(data,w):  # Function that returns rms error for a given model on any data input
    x=data[:,0]
    y=data[:,1]
    p=np.poly1d(w[::-1]) 

    E_w=(sum((p(x)-y)**2))/2
    return ((2*E_w)/len(x))**0.5

def find_best_model_1D(): #Function for finding the best model for the 1D data provided
    train_rms_errors_m=[]
    test_rms_errors_m=[]
    for m in range(12): #Finding rms errors on train and development data for different model complexities
        w=poly_reg_1D(data_1d_train,m)
        train_rms_errors_m.append(rms_error_1D(data_1d_train,w))
        test_rms_errors_m.append(rms_error_1D(data_1d_dev,w))

    #Plotting rms error vs model complexity(order of polynomial)
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(0,11,12),train_rms_errors_m,'.-',label="training data error")
    plt.plot(np.linspace(0,11,12),test_rms_errors_m,'.-',label="development data error")
    plt.xlabel("Order of polynomial (m)")
    plt.ylabel("RMS Error")
    plt.title("1D Dataset:Observing rms errors to find out best polynomial order")
    plt.legend()
    plt.show()

    # Finding rms error for different regularisation parameter values for the best polynomial order
    train_rms_errors_l=[]
    test_rms_errors_l=[]
    for l in np.logspace(-15,0,100):
        w=poly_reg_1D(data_1d_train,10,l)
        train_rms_errors_l.append(rms_error_1D(data_1d_train,w))
        test_rms_errors_l.append(rms_error_1D(data_1d_dev,w))

    #Plotting rms error vs lambda(regularisation parameter)
    plt.figure(figsize=(8,6))
    plt.plot(np.logspace(-15,0,100),train_rms_errors_l,'.-',label="training data error")
    plt.plot(np.logspace(-15,0,100),test_rms_errors_l,'.-',label="development data error")
    plt.xscale("log")
    plt.xlabel("Regularisation parameter (Lambda)")
    plt.ylabel("RMS Error")
    plt.title("1D dataset:Finding best regularisation parameter(m=10)")
    plt.legend()
    plt.show()



##########################################################################

#Regression on 2D Dataset
x1=data_2d_train[:,0]
x2=data_2d_train[:,1]
y=data_2d_train[:,2]

x1_ = np.linspace(-1, 1,1000)
x2_ = np.linspace(-1, 1,1000)
x1v, x2v = np.meshgrid(x1_, x2_)

#Plot of the approximated functions obtained using training datasets of different sizes
W_n=[]
data_2d_n=[]  #This list will contain datasets with different sizes for experiments

for i in [50,200,500,1000]:    #Data is sampled uniformly into sets of different sizes
    random_indices = np.random.choice(data_2d_train.shape[0], size=i, replace=False)
    data_2d_n.append(data_2d_train[random_indices, :])
    W_n.append(poly_reg_2D(data_2d_n[-1],6))   #Weights for these different sizes are obtained

def plot_2D_different_n():
    #We get the predictions from different models constructed above for different dataset sizes
    y_1=np.polynomial.polynomial.polyval2d(x1v, x2v , W_n[0])
    y_1=y_1.reshape(1000,1000)
    y_2=np.polynomial.polynomial.polyval2d(x1v, x2v , W_n[1])
    y_2=y_2.reshape(1000,1000)
    y_3=np.polynomial.polynomial.polyval2d(x1v, x2v , W_n[2])
    y_3=y_3.reshape(1000,1000)
    y_4=np.polynomial.polynomial.polyval2d(x1v, x2v , W_n[3])
    y_4=y_4.reshape(1000,1000)

    #Plotting the prediction curves for different models and their corresponding training data
    fig = plt.figure(figsize=(11,8))
    axis=[]
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_1, color="blue")
    ax.scatter3D(data_2d_n[0][:,0], data_2d_n[0][:,1], data_2d_n[0][:,2],c='red')
    ax.set_title("n=50")
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_2, color="blue")
    ax.scatter3D(data_2d_n[1][:,0], data_2d_n[1][:,1], data_2d_n[1][:,2],c='red')
    ax.set_title("n=200")
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_3, color="blue")
    ax.scatter3D(data_2d_n[2][:,0], data_2d_n[2][:,1], data_2d_n[2][:,2],c='red')
    ax.set_title("n=500")
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_4, color="blue")
    ax.scatter3D(data_2d_n[3][:,0], data_2d_n[3][:,1], data_2d_n[3][:,2],c='red')
    ax.set_title("n=1000")
    for ax in axis:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
    fig.suptitle("2D dataset: Models for varying dataset sizes")
    fig.subplots_adjust(top=0.91)
    plt.show()


#Plot of the approximated functions obtained for different model complexities
def plot_2D_different_m():
    data=data_2d_n[2]     #Here we just take the dataset with number of data points as 50
    w_m=[]                #And observe the models characteristics for different model complexities
    for i in [0,1,4,10]:  #Weights for the different models are stored
        w_m.append(poly_reg_2D(data,i))   #Data for all models is fixed

    y_1=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[0])
    y_1=y_1.reshape(1000,1000)
    y_2=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[1])
    y_2=y_2.reshape(1000,1000)
    y_3=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[2])
    y_3=y_3.reshape(1000,1000)
    y_4=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[3])
    y_4=y_4.reshape(1000,1000)

    #Plotting the prediction curves for the models with different complexity(polynomial order)
    fig = plt.figure(figsize=(11,8))
    axis=[]
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_1, color="blue")
    ax.scatter3D(data[:,0], data[:,1], data[:,2],c='red')
    ax.set_title("m=0")
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_2, color="blue")
    ax.scatter3D(data[:,0], data[:,1], data[:,2],c='red')
    ax.set_title("m=1")
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_3, color="blue")
    ax.scatter3D(data[:,0], data[:,1], data[:,2],c='red')
    ax.set_title("m=4")
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    axis.append(ax)
    ax.plot_surface(x1v, x2v, y_4, color="blue")
    ax.scatter3D(data[:,0], data[:,1], data[:,2],c='red')
    ax.set_title("m=10")
    for ax in axis:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
    fig.suptitle("2D dataset: Models for varying model complexity but fixed dataset")
    fig.subplots_adjust(top=0.91)
    plt.show()

#Finding the best model for the 2D dataset for which we will see the error on development and train data
def rms_error_2D(data,c):  # Function that returns rms error for a given model on any data input
    x1=data[:,0]
    x2=data[:,1]
    y=data[:,2]
    y_model=np.polynomial.polynomial.polyval2d(x1, x2 , c)

    E_w=(sum((y_model-y)**2))/2
    return ((2*E_w)/len(x))**0.5

def find_best_model_2D(): #Function for finding the best model for the 2D data provided
    train_rms_errors_m=[]
    test_rms_errors_m=[]
    for m in range(13): #Finding rms errors on train and development data for different model complexities
        c=poly_reg_2D(data_2d_train,m)
        train_rms_errors_m.append(rms_error_2D(data_2d_train,c))
        test_rms_errors_m.append(rms_error_2D(data_2d_dev,c))

    #Plotting rms error vs model complexity(order of polynomial)
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(0,12,13),train_rms_errors_m,'.-',label="training data error")
    plt.plot(np.linspace(0,12,13),test_rms_errors_m,'.-',label="development data error")
    plt.xlabel("Order of polynomial (m)")
    plt.ylabel("RMS Error")
    plt.title("2D dataset:Observing rms errors to find out best polynomial order")
    plt.legend()
    plt.show()

    # Finding rms error for different regularisation parameter values for the best polynomial order
    train_rms_errors_l=[]
    test_rms_errors_l=[]
    for l in np.logspace(-12,1,100):
        c=poly_reg_2D(data_2d_train,9,l)
        train_rms_errors_l.append(rms_error_2D(data_2d_train,c))
        test_rms_errors_l.append(rms_error_2D(data_2d_dev,c))

    #Plotting rms error vs lambda(regularisation parameter)
    plt.figure(figsize=(8,6))
    plt.plot(np.logspace(-12,1,100),train_rms_errors_l,'.-',label="training data error")
    plt.plot(np.logspace(-12,1,100),test_rms_errors_l,'.-',label="development data error")
    plt.xscale("log")
    plt.xlabel("Regularisation parameter (Lambda)")
    plt.ylabel("RMS Error")
    plt.title("2D dataset:Finding best regularisation parameter")
    plt.legend()
    plt.show()

#Plotting the best models
def plot_best_fit():
    #1D Best model for Least square regression
    x=data_1d_dev[:,0]
    y=data_1d_dev[:,1]
    w1=poly_reg_1D(data_1d_train,10)
    p1=np.poly1d(w1[::-1])
    plt.plot(np.linspace(0.05, 4.9,num=1000),p1(np.linspace(0.05, 4.9,num=1000)),label="m=10")
    plt.scatter(x,y,s=25,color='orange',label="dev data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Best fit for 1D dataset least squares regression")
    plt.show()
    print("RMS error for best model on 1D dev data for Least square regression =",rms_error_1D(data_1d_dev,w1))

    #1D Best model for ridge regression
    x=data_1d_dev[:,0]
    y=data_1d_dev[:,1]
    w2=poly_reg_1D(data_1d_train,10,5e-6)
    p1=np.poly1d(w2[::-1])
    plt.plot(np.linspace(0.05, 4.9,num=1000),p1(np.linspace(0.05, 4.9,num=1000)),label="m=10,"+r'$\lambda$=5e-6')
    plt.scatter(x,y,s=25,color='orange',label="dev data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Best fit for 1D dataset ridge regression")
    plt.show()
    print("RMS error for best model on 1D dev data for ridge regression =",rms_error_1D(data_1d_dev,w2))

    #Scatter plot of target output on the x-axis and model output on the y-axis
    y1=p1(x)
    x_=data_1d_train[:,0]
    y_=data_1d_train[:,1]
    y2=p1(x_)

    figure, axis = plt.subplots(1,2,figsize=(10,4))
    for ax in axis.flatten():
        ax.set_xlabel('target output')
        ax.set_ylabel('predictions')
    axis[0].plot(y,y1,'.',color='orange')
    axis[0].set_title("target output is dev data")
    axis[1].plot(y_,y2,'.')
    axis[1].set_title("target output is train data")
    figure.suptitle("1D data")
    plt.show()



    #2D Best model for Least square regression
    x1=data_2d_dev[:,0]
    x2=data_2d_dev[:,1]
    y=data_2d_dev[:,2]

    c1=poly_reg_2D(data_2d_train,9,1e-5)
    y_model=np.polynomial.polynomial.polyval2d(x1v, x2v , c1)
    y_model=y_model.reshape(1000,1000)
    fig = plt.figure(figsize =(10, 6))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(x1v,x2v ,y_model,color='blue',label="m=9")
    ax.scatter3D(x1, x2, y,c='red',label="dev data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.title("Best fit for 2D dataset Least square regression")
    plt.show()
    print("RMS error for best model on 2D dev data for Least square regression =",rms_error_2D(data_2d_dev,c1))

    #2D Best model for ridge regresssion
    x1=data_2d_dev[:,0]
    x2=data_2d_dev[:,1]
    y=data_2d_dev[:,2]

    c2=poly_reg_2D(data_2d_train,9,1e-5)
    y_model=np.polynomial.polynomial.polyval2d(x1v, x2v , c2)
    y_model=y_model.reshape(1000,1000)
    fig = plt.figure(figsize =(10, 6))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(x1v,x2v ,y_model,color='blue',label="m=9,"+r'$\lambda$=1e-5')
    ax.scatter3D(x1, x2, y,c='red',label="dev data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.title("Best fit for 2D dataset ridge regresssion")
    plt.show()
    print("RMS error for best model on 2D dev data for Ridge regression =",rms_error_2D(data_2d_dev,c2))

    #Scatter plot of target output on the x-axis and model output on the y-axis
    y1=np.polynomial.polynomial.polyval2d(x1, x2 , c2)
    x1_=data_2d_train[:,0]
    x2_=data_2d_train[:,1]
    y_=data_2d_train[:,2]
    y2=np.polynomial.polynomial.polyval2d(x1_, x2_ , c2)

    figure, axis = plt.subplots(1,2,figsize=(10,4))
    for ax in axis.flatten():
        ax.set_xlabel('target output')
        ax.set_ylabel('predictions')
    axis[0].plot(y,y1,'.',color='orange')
    axis[0].set_title("target output is dev data")
    axis[1].plot(y_,y2,'.')
    axis[1].set_title("target output is train data")
    figure.suptitle("2D data")
    plt.show()



if __name__ == "__main__":
    #EXPERIMENTS
    #Calling all 1D dataset functions(If required,specific functions can be commented out)
    plot_1D_different_n()
    plot_1D_different_m()
    plot_1D_different_lambda()

    #Calling all 2D dataset functions(If required,specific functions can be commented out)
    plot_2D_different_n()
    plot_2D_different_m()

    #Finding the best models using method specified in the functions
    find_best_model_1D()
    find_best_model_2D()

    #Plotting the best models for both datasets along with dev data and rms errors
    plot_best_fit()