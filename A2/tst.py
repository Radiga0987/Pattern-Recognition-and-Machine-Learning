import matplotlib.pyplot as plt
import numpy as np

data_1d_train=np.loadtxt('1d_team_10_train.txt')
data_1d_dev=np.loadtxt('1d_team_10_dev.txt')

data_2d_train=np.loadtxt('2d_team_10_train.txt')
data_2d_dev=np.loadtxt('2d_team_10_dev.txt')

def poly_reg_1D(data,n,lamda=0):
    x=data[:,0]
    y=data[:,1]
    phiT=[]
    
    for i in range(n+1):
        phiT.append(x**i)
    
    #print(np.linalg.cond(phiT))
    phiT_phi=np.dot(phiT,np.transpose(phiT))
    inv=np.linalg.inv(phiT_phi + np.diag([lamda]*(n+1)))
    weights=np.dot(np.dot(inv,phiT),y)

    """"
    if n>=12:
        c = np.linalg.inv(np.linalg.cholesky(phiT_phi+ np.diag([1e-10]*(n+1))))
        inv2 = np.dot(c.T,c)
        weights=np.dot(np.dot(inv2,phiT),y)"""

    return weights

def poly_reg_2D(data,n,lamda=0):
    x1=data[:,0]
    x2=data[:,1]
    y=data[:,2]
    phiT=[]
    
    for count in range(n+1):
        for i in range(count+1):
            phiT.append((x1**(count-i))*(x2**i))

    phiT_phi=np.dot(phiT,np.transpose(phiT))
    inv=np.linalg.inv(phiT_phi + np.diag([lamda]*len(phiT_phi)))
    w=np.dot(np.dot(inv,phiT),y)

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

w=poly_reg_1D(data_1d_train,13)
#print(w)

p1=np.poly1d(w[::-1])
plt.scatter(x,y,s=25,color='orange')
plt.plot(np.linspace(0, 4.9,num=1000),p1(np.linspace(0, 4.9,num=1000)))
plt.show()


x1=data_2d_train[:,0]
x2=data_2d_train[:,1]
y=data_2d_train[:,2]
n=4
c=poly_reg_2D(data_2d_train,n)
x1_ = np.linspace(-1, 1,1000)
x2_ = np.linspace(-1, 1,1000)
x1v, x2v = np.meshgrid(x1_, x2_)
y_model=np.polynomial.polynomial.polyval2d(x1v, x2v , c)
y_model=y_model.reshape(1000,1000)
fig = plt.figure(figsize =(10, 6))
ax = plt.axes(projection ='3d')
# Creating plot
ax.plot_surface(x1v,x2v ,y_model,color='gray')
ax.scatter3D(x1, x2, y,c='red')
# show plot
plt.grid()
plt.show()
"""

"""
#Plot of the approximated functions obtained using training datasets of different sizes
data_1d_n=[]
w_n=[]
for i in [10,20,50,100,200]:
    random_indices = np.random.choice(data_1d_train.shape[0], size=i, replace=False)
    data_1d_n.append(data_1d_train[random_indices, :])
    w_n.append(poly_reg_1D(data_1d_n[-1],6))

x_=np.linspace(0, 4.9,num=1000)
y10=np.poly1d(w_n[0][::-1])(x_)
y20=np.poly1d(w_n[1][::-1])(x_)
y50=np.poly1d(w_n[2][::-1])(x_)
y100=np.poly1d(w_n[3][::-1])(x_)
y200=np.poly1d(w_n[4][::-1])(x_)

figure, axis = plt.subplots(2, 3,figsize=(13,7))
figure.delaxes(axis[1][2])
axis[0, 0].plot(x_, y10)
axis[0, 0].scatter(data_1d_n[0][:,0],data_1d_n[0][:,1],s=25,color='none',edgecolor='red')
axis[0, 0].set_title("$N=10$")
axis[0, 1].plot(x_, y20)
axis[0, 1].scatter(data_1d_n[1][:,0],data_1d_n[1][:,1],s=25,color='none',edgecolor='red')
axis[0, 1].set_title("$N=20$")
axis[0, 2].plot(x_, y50)
axis[0, 2].scatter(data_1d_n[2][:,0],data_1d_n[2][:,1],s=25,color='none',edgecolor='red')
axis[0, 2].set_title("$N=50$")
axis[1, 0].plot(x_, y100)
axis[1, 0].scatter(data_1d_n[3][:,0],data_1d_n[3][:,1],s=25,color='none',edgecolor='red')
axis[1, 0].set_title("$N=100$")
axis[1, 1].plot(x_, y200)
axis[1, 1].scatter(data_1d_n[4][:,0],data_1d_n[4][:,1],s=25,color='none',edgecolor='red')
axis[1, 1].set_title("$N=200$")
plt.show()



#Plot of the approximated functions obtained for different model complexities
data_50=data_1d_n[2]
w_m=[]
for i in [0,1,4,10]:
    w_m.append(poly_reg_1D(data_50,i))

x_=np.linspace(0, 4.9,num=1000)
y0=np.poly1d(w_m[0][::-1])(x_)
y1=np.poly1d(w_m[1][::-1])(x_)
y4=np.poly1d(w_m[2][::-1])(x_)
y10=np.poly1d(w_m[3][::-1])(x_)
print(w_m[3])

figure, axis = plt.subplots(2, 2,figsize=(10,7))
axis[0, 0].plot(x_, y0)
axis[0, 0].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[0, 0].set_title("$M=0$")
axis[0, 1].plot(x_, y1)
axis[0, 1].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[0, 1].set_title("$M=1$")
axis[1, 0].plot(x_, y4)
axis[1, 0].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[1, 0].set_title("$M=4$")
axis[1, 1].plot(x_, y10)    
axis[1, 1].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[1, 1].set_title("$M=10$")
plt.show()



#Plot of the approximated functions obtained for different values of regularization parameter
data_50=data_1d_n[4]
w_l=[]
for i in [0,1e-8,5e-6,1,10]:
    w_l.append(poly_reg_1D(data_50,10,i))

x_=np.linspace(0, 4.9,num=1000)
y_inf=np.poly1d(w_l[0][::-1])(x_)
y_12=np.poly1d(w_l[1][::-1])(x_)
y_6=np.poly1d(w_l[2][::-1])(x_)
y_1=np.poly1d(w_l[3][::-1])(x_)
y_0=np.poly1d(w_l[4][::-1])(x_)


figure, axis = plt.subplots(2, 3,figsize=(10,7))
figure.delaxes(axis[1][2])
axis[0, 0].plot(x_, y_inf)
axis[0, 0].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[0, 0].set_title("$M=0$")
axis[0, 1].plot(x_, y_12)
axis[0, 1].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[0, 1].set_title("$M=1$")
axis[0, 2].plot(x_, y_6)
axis[0, 2].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[0, 2].set_title("$M=4$")
axis[1, 0].plot(x_, y_1)
axis[1, 0].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[1, 0].set_title("$M=10$")
axis[1, 1].plot(x_, y_0)
axis[1, 1].scatter(data_50[:,0],data_50[:,1],s=25,color='none',edgecolor='red')
axis[1, 1].set_title("$M=10$")
plt.show()

#Finding the best model for the 1D dataset for which we will see the error on development and train data
def rms_error_1D(data,w):
    x=data[:,0]
    y=data[:,1]
    p=np.poly1d(w[::-1]) 

    E_w=(sum((p(x)-y)**2))/2
    return ((2*E_w)/len(x))**0.5

train_rms_errors_m=[]
test_rms_errors_m=[]
for m in range(12):
    w=poly_reg_1D(data_1d_train,m,1 if m>=12 else 0)
    train_rms_errors_m.append(rms_error_1D(data_1d_train,w))
    test_rms_errors_m.append(rms_error_1D(data_1d_dev,w))

plt.figure(figsize=(8,6))
plt.plot(np.linspace(0,11,12),train_rms_errors_m,'.-',label="training data error")
plt.plot(np.linspace(0,11,12),test_rms_errors_m,'.-',label="development data error")
plt.legend()
plt.show()

train_rms_errors_l=[]
test_rms_errors_l=[]
for l in np.logspace(-15,0,100):
    w=poly_reg_1D(data_1d_train,10,l)
    train_rms_errors_l.append(rms_error_1D(data_1d_train,w))
    test_rms_errors_l.append(rms_error_1D(data_1d_dev,w))

plt.figure(figsize=(8,6))
plt.plot(np.logspace(-15,0,100),train_rms_errors_l,'.-',label="training data error")
plt.plot(np.logspace(-15,0,100),test_rms_errors_l,'.-',label="development data error")
plt.xscale("log")
plt.legend()
plt.show()






#2D
#Plot of the approximated functions obtained using training datasets of different sizes
data_2d_n=[]
w_n=[]
x1=data_2d_train[:,0]
x2=data_2d_train[:,1]
y=data_2d_train[:,2]
for i in [50,200,500,1000]:
    random_indices = np.random.choice(data_2d_train.shape[0], size=i, replace=False)
    data_2d_n.append(data_2d_train[random_indices, :])
    w_n.append(poly_reg_2D(data_2d_n[-1],6))

x1_ = np.linspace(-1, 1,1000)
x2_ = np.linspace(-1, 1,1000)
x1v, x2v = np.meshgrid(x1_, x2_)
y_1=np.polynomial.polynomial.polyval2d(x1v, x2v , w_n[0])
y_1=y_1.reshape(1000,1000)
y_2=np.polynomial.polynomial.polyval2d(x1v, x2v , w_n[1])
y_2=y_2.reshape(1000,1000)
y_3=np.polynomial.polynomial.polyval2d(x1v, x2v , w_n[2])
y_3=y_3.reshape(1000,1000)
y_4=np.polynomial.polynomial.polyval2d(x1v, x2v , w_n[3])
y_4=y_4.reshape(1000,1000)

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_1, color="blue")
ax.set_title("n=50")
ax = fig.add_subplot(2, 2, 2, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_2, color="blue")
ax.set_title("n=200")
ax = fig.add_subplot(2, 2, 3, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_3, color="blue")
ax.set_title("n=500")
ax = fig.add_subplot(2, 2, 4, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_4, color="blue")
ax.set_title("n=1000")
plt.show()


#Plot of the approximated functions obtained for different model complexities
data_500=data_2d_n[2]
w_m=[]
for i in [0,1,4,10]:
    w_m.append(poly_reg_2D(data_500,i))

y_1=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[0])
y_1=y_1.reshape(1000,1000)
y_2=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[1])
y_2=y_2.reshape(1000,1000)
y_3=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[2])
y_3=y_3.reshape(1000,1000)
y_4=np.polynomial.polynomial.polyval2d(x1v, x2v , w_m[3])
y_4=y_4.reshape(1000,1000)

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_1, color="blue")
ax.set_title("m=0")
ax = fig.add_subplot(2, 2, 2, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_2, color="blue")
ax.set_title("m=1")
ax = fig.add_subplot(2, 2, 3, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_3, color="blue")
ax.set_title("m=4")
ax = fig.add_subplot(2, 2, 4, projection='3d')
surf = ax.plot_surface(x1v, x2v, y_4, color="blue")
ax.set_title("m=10")
plt.show()

#Finding the best model for the 2D dataset for which we will see the error on development and train data
def rms_error_2D(data,c):
    x1=data[:,0]
    x2=data[:,1]
    y=data[:,2]
    y_model=np.polynomial.polynomial.polyval2d(x1, x2 , c)

    E_w=(sum((y_model-y)**2))/2
    return ((2*E_w)/len(x))**0.5


train_rms_errors_m=[]
test_rms_errors_m=[]
for m in range(20):
    c=poly_reg_2D(data_2d_train,m)
    train_rms_errors_m.append(rms_error_2D(data_2d_train,c))
    test_rms_errors_m.append(rms_error_2D(data_2d_dev,c))

plt.figure(figsize=(8,6))
plt.plot(np.linspace(0,19,20),train_rms_errors_m,'.-',label="training data error")
plt.plot(np.linspace(0,19,20),test_rms_errors_m,'.-',label="development data error")
plt.legend()
plt.show()

train_rms_errors_l=[]
test_rms_errors_l=[]
for l in np.logspace(-15,1,100):
    c=poly_reg_2D(data_2d_train,7,l)
    train_rms_errors_l.append(rms_error_2D(data_2d_train,c))
    test_rms_errors_l.append(rms_error_2D(data_2d_dev,c))

plt.figure(figsize=(8,6))
plt.plot(np.logspace(-15,1,100),train_rms_errors_l,'.-',label="training data error")
plt.plot(np.logspace(-15,1,100),test_rms_errors_l,'.-',label="development data error")
plt.xscale("log")
plt.legend()
plt.show()