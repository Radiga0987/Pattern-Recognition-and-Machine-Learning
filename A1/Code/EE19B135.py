#Assignment on decomposition/factorization of an image 
#Rishabh Adiga - EE19B135
#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

Im = img.imread("11.jpg")    #Representing the image as a matrix
Im=Im.astype("float64")      #All elements converted to float64 to increase precision
plt.imshow(Im,cmap="gray")   #Viewing the original image
plt.title("Original Image") 
plt.show()

#Defining various functions to be used for our experiments
def EVD(img):
    """Function that performs Eigen value decomposition on input image
        img=X*V*X_inv

    Inputs:
        img ([Matrix]): image to perform EVD on.

    Returns:
        X,V,X_inv ([Matrices]): The 3 matrices that img is decomposed to.
    """
    e_vals,e_vecs=np.linalg.eig(img) #Computing eigen values and eigen vectors of img

    mag=[abs(v) for v in e_vals]     #The eigen values and corresponding vectors are sorted according to magnitude
    mag,e_vals,e_vecs=[list(v) for v in zip(*sorted(zip(mag,e_vals,np.transpose(e_vecs)),reverse=True))]

    X=np.transpose(e_vecs)      #X is formed using the eigen vectors of img              
    X_inv=np.linalg.inv(X)      #X_inv is the inverse of X
    V=np.diag(e_vals)           #V is a diagnol matrix with sorted eigen values on diagnol 

    return X,V,X_inv

def SVD(img):
    """Function that performs Singular value decomposition on input image
        img=U*S*V_t

    Inputs:
        img ([Matrix]): image to perform SVD on.

    Returns:
        U,S,V_t ([Matrices]): The 3 matrices that img is decomposed to.
    """
    imgt_img=np.dot(np.transpose(img),img)    #imgt refers to transpose of img matrix and imgt_img=imgt*img
    e_vals,e_vecs=np.linalg.eig(imgt_img)     #Computing eigen values and eigen vectors of img_transpose*img 

    mag=[abs(v) for v in e_vals]                #The eigen values and corresponding vectors are sorted according to magnitude
    mag,e_vals,e_vecs=[list(v) for v in zip(*sorted(zip(mag,e_vals,np.transpose(e_vecs)),reverse=True))]

    #Due to eigen vector sign correspondance problems,U matrix is computed using eigen vectors of imgt*img instead of img*imgt
    U=np.dot(img,np.transpose(e_vecs))/np.sqrt(e_vals)   
    S=np.diag(np.sqrt(e_vals))  #S is a diagnol matrix with sqaure root sorted eigen values on diagnol
    V_t=np.conj(e_vecs)         #V_t is the conjugate transpose of eigen vectors of imgt*img (transpose already done while sorting)

    return U,S,V_t

def recon_img_k(A,B,C,K=256):
    """This function is used to get the reconstructed image by considering only top K eigen values

    Inputs:
        A,B,C ([Matrices]): The 3 matrices that are multiplied to get original img (img=A*B*C).
        K (int, optional): K is the number of eigen values to be considered for reconstruction of the image.Defaults to 256.

    Returns:
        reconstructed_image ([Matrix]): Image formed by using top K eigen values.
    """
    B_k=B.copy()
    for i in range(K,256):  #We make all the eigen values on the diagnol from the Kth row to 256 as 0
        B_k[i][i]=0         #This will ensure that we are considering only contribution of 1st to Kth eigen value and vectors in reconstruction
    #Now we reconstruct the image using B_k by doing A*B_k*C
    reconstructed_image=np.absolute(np.dot(A,np.dot(B_k,C)))

    return reconstructed_image

def frob_norm_plot(img,A,B,C,s="EVD",bool=True):
    """This function is used for plotting the frobenius norm of the error matrix vs 
    K(top K eigen values considered) which varies from 1 to 256.

    Args:
        img ([Matrix]): image being used.
        A,B,C ([Matrices]): The 3 matrices that are multiplied to get original img (img=A*B*C).
        s ([str]): Takes value as "EVD" or "SVD" for plot labelling
        bool ([boolean]):True if plot is needed else False
    """
    Frobenius_norms_K=[]       #List to store the norm for all 256 values of K
    for K in range(1,257):     #K varies from 1 to 256
        error_img=abs(img-recon_img_k(A,B,C,K))     #Kth error matrix calculated using original img and reconstructed img
        frob_norm=np.linalg.norm(error_img, 'fro')  #Kth frobenius norm is calculated
        Frobenius_norms_K.append(frob_norm)         #Added to the list of norms
    #Now we plot frobenius norm for error image vs K
    if bool:
        plt.figure()
        plt.plot(np.linspace(1, 256,num=256),Frobenius_norms_K)
        plt.xlabel("K (No. of Eigen values used)") 
        plt.ylabel("Frobenius norm") 
        plt.title("Frobenius norm plot of error images for "+s)
        plt.show()

    return Frobenius_norms_K





#Various experiments performed below using all the functions defined above

#Expt1-Performing EVD on the image
X,V,X_inv=EVD(Im)  # 3 matrices from EVD are obtained here
#Viewing reconstructed image for K=220,124,47,5
#Image for K=220
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(X,V,X_inv,220),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=220,EVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,220)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=220,EVD")
plt.show()

#Image for K=124
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(X,V,X_inv,124),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=124,EVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,124)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=124,EVD")
plt.show()

#Image for K=47
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(X,V,X_inv,47),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=47,EVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,47)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=47,EVD")
plt.show()

#Image for K=5
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(X,V,X_inv,5),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=5,EVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,5)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=5,EVD")
plt.show()

#Frobenius norm of error image for EVD is plotted
x=frob_norm_plot(Im,X,V,X_inv,"EVD")




#Expt2-Performing SVD on the image
U,S,V_t=SVD(Im)  # 3 matrices from EVD are obtained here
#Viewing reconstructed image and error image for K=220,124,47,5
#Image for K=220
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(U,S,V_t,220),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=220,SVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,220)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=220,SVD")
plt.show()

#Image for K=124
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(U,S,V_t,124),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=124,SVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,124)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=124,SVD")
plt.show()

#Image for K=47
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(U,S,V_t,47),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=47,SVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,47)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=47,SVD")
plt.show()

#Image for K=5
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(recon_img_k(U,S,V_t,5),cmap="gray", vmin=0, vmax=255)
plt.title("Image for K=5,SVD")
fig.add_subplot(1, 2, 2)
plt.imshow(abs(recon_img_k(X,V,X_inv,5)-Im),cmap="gray", vmin=0, vmax=255)
plt.title("Error image for K=5,SVD")
plt.show()

#Frobenius norm of error image for SVD is plotted
y=frob_norm_plot(Im,U,S,V_t,"SVD")

#Viewing frobenius norm vs K for both EVD and SVD in one plot
plt.figure()
plt.plot(np.linspace(1, 256,num=256),x,label="EVD")
plt.plot(np.linspace(1, 256,num=256),y,label="SVD")
plt.xlabel("K (No. of Eigen values used)") 
plt.ylabel("Frobenius norm") 
plt.title("Comparing frobenius norm plot of error images")
plt.legend()
plt.show()



#Expt3-Condition number of the matrices for EVD and SVD
"""We can find the condition number using np.linalg.cond() which is a numpy 
function or we can just divide the largest eigen value by the smallest eigen value.
Let us assume A=XVX_inv"""
#We use The Bauer–Fike Theorem to understand which method is better.
"""This theorem says that the relative variation in eigen values is 
proportional to condition number of X when there is a small change in A.
For a better understanding refer to the report """
#Condition num is applied on matrix X for EVD
Cond_num_1=np.linalg.cond(X)
print("The condition number for the matrix in EVD = ",Cond_num_1)  #Condition no is printed

#Condition num is applied on matrix U for SVD
Cond_num_2=np.linalg.cond(U)
print("The condition number for the matrix in SVD = ",Cond_num_2)  #Condition no is printed




#Expt4-Performing EVD on RBG image (256x256x3)
rgb = img.imread("color.jpg")    #Representing the image as a matrix
rgb = rgb.astype("float64")   #All elements converted to float64 to increase precision

#Obtaining the 3 256x256 matrices that form the rgb image
Im_1=rgb[:,:,0]
Im_2=rgb[:,:,1]
Im_3=rgb[:,:,2]
#Now we apply EVD on each of these 3 matrices
X1,V1,X1_inv=EVD(Im_1)
X2,V2,X2_inv=EVD(Im_2)
X3,V3,X3_inv=EVD(Im_3)
#Using these, we can reconstruct the 256x256x3 by applying np.dstack on the 3 256x256 recconstructed matrices

#Image for K=200
Recon_img=np.dstack((recon_img_k(X1,V1,X1_inv,200),recon_img_k(X2,V2,X2_inv,200),recon_img_k(X3,V3,X3_inv,200)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=200,EVD")
plt.show()

#Image for K=100
Recon_img=np.dstack((recon_img_k(X1,V1,X1_inv,100),recon_img_k(X2,V2,X2_inv,100),recon_img_k(X3,V3,X3_inv,100)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=100,EVD")
plt.show()

#Image for K=25
Recon_img=np.dstack((recon_img_k(X1,V1,X1_inv,25),recon_img_k(X2,V2,X2_inv,25),recon_img_k(X3,V3,X3_inv,25)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=25,EVD")
plt.show()
#Finding the frobenius norm of the error matrices for each of the 3 reconstructed matrices
f1=frob_norm_plot(Im_1,X1,V1,X1_inv,bool=False)
f2=frob_norm_plot(Im_2,X2,V2,X2_inv,bool=False)
f3=frob_norm_plot(Im_3,X3,V3,X3_inv,bool=False)
frob_rgb_evd=[f1[i] + f2[i] + f3[i] for i in range(len(f1))]    
#The corresponding values are summed up




#Expt5-Performing SVD on RBG image (256x256x3)
#SVD is applied on each of the 3 matrices
U1,S1,V1_t=SVD(Im_1)
U2,S2,V2_t=SVD(Im_2)
U3,S3,V3_t=SVD(Im_3)
#Using these, we can reconstruct the 256x256x3 by applying np.dstack on the 3 256x256 recconstructed matrices

#Image for K=200
Recon_img=np.dstack((recon_img_k(U1,S1,V1_t,200),recon_img_k(U2,S2,V2_t,200),recon_img_k(U3,S3,V3_t,200)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=200,SVD")
plt.show()

#Image for K=100
Recon_img=np.dstack((recon_img_k(U1,S1,V1_t,100),recon_img_k(U2,S2,V2_t,100),recon_img_k(U3,S3,V3_t,100)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=100,SVD")
plt.show()

#Image for K=25
Recon_img=np.dstack((recon_img_k(U1,S1,V1_t,25),recon_img_k(U2,S2,V2_t,25),recon_img_k(U3,S3,V3_t,25)))
Recon_img=Recon_img/Recon_img.max()
plt.imshow(Recon_img)
plt.title("Rgb image for K=25,SVD")
plt.show()
#Finding the frobenius norm of the error matrices for each of the 3 reconstructed matrices
f1=frob_norm_plot(Im_1,U1,S1,V1_t,bool=False)
f2=frob_norm_plot(Im_2,U2,S2,V2_t,bool=False)
f3=frob_norm_plot(Im_3,U3,S3,V3_t,bool=False)
frob_rgb_svd=[f1[i] + f2[i] + f3[i] for i in range(len(f1))]
#The corresponding values are summed up

#Finally, plotting frobenius norm vs K for both EVD and SVD in one plot
plt.figure()
plt.plot(np.linspace(1, 256,num=256),frob_rgb_evd,label="EVD")
plt.plot(np.linspace(1, 256,num=256),frob_rgb_svd,label="SVD")
plt.xlabel("K (No. of Eigen values used)") 
plt.ylabel("Frobenius norm") 
plt.title("Comparing frobenius norm plot of error images of the rgb imgs")
plt.legend()
plt.show()