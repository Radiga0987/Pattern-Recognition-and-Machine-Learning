EE19B135 Rishabh Adiga
EE19B134 Amogh Patil

The "Code" folder contains GMM.py ,DTW.py and the HMM-Code folder which contains 2 important files(train_test_model.py and HMM_Symbol_Generation.py).

The K means and GMM part of this assignment is present in GMM.py and can be directly run and results will be obtained.

For DTW ,DTW.py needs to be run.This code takes a long time to run without using Numba.But I have added Numbas njit module to make it run within 2 minutes.So for this , one needs to pip install numba or conda install numba.It can be run without doing this too but lines 5 and 23 needs to be commented out and the code will be slow.(For more info, refer to comments inside DTW.py)

For HMM, all the required code is inside HMM-Code folder which contains 2 important files(train_test_model.py and HMM_Symbol_Generation.py) and it has to Necessarily Be Tested On A Linux Machine(IMPORTANT).
HMM_Symbol_Generation.py is the file that makes the symbol sequences for all the train and test examples using K means.IMPORTANT - The symbol sequences have already been generated and are present in Digits_Symbol_Data and Letters_Symbol_Data folder for the 2 datasets,therefore running this file is not necessary!
To train the HMMs and test the dev data,only train_test_model.py file should be run and the results will be shown as plots and data on the terminal.

Please maintain the same folder and file structure while running the codes.
