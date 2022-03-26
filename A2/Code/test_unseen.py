#Importing required librarires
import matplotlib.pyplot as plt
import numpy as np
from regression import poly_reg_1D,poly_reg_2D,rms_error_1D,rms_error_2D

#Loading the data
data_1d_train=np.loadtxt('Data/1d_team_10_train.txt')
data_1d_unseen=np.loadtxt('Data/1d_team_10_dev.txt')

data_2d_train=np.loadtxt('Data/2d_team_10_train.txt')
data_2d_unseen=np.loadtxt('Data/2d_team_10_dev.txt')

def unseen_1d(data_unseen):
    w=poly_reg_1D(data_1d_train,10,5e-6)
    rmserror_1d_unseen=rms_error_1D(data_unseen,w)
    print("RMS error on unseen data for 1D=",rmserror_1d_unseen)

def unseen_2d(data_unseen):
    c=poly_reg_2D(data_2d_train,9,1e-5)
    rmserror_2d_unseen=rms_error_2D(data_unseen,c)
    print("RMS error on unseen data for 2D=",rmserror_2d_unseen)

unseen_1d(data_1d_unseen)
unseen_2d(data_2d_unseen)