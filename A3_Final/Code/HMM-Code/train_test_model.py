import subprocess
import numpy as np
import matplotlib.pyplot as plt

#Function to plot confusion matrix
def Confusion_matrix(predicted_classes,Title = "Confusion Matrix"):
    conf_matrix = np.zeros((len(predicted_classes),len(predicted_classes)))
    for cls in range(len(predicted_classes)):
        for val in predicted_classes[cls]:
            conf_matrix[cls][val] += 1

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


#Function to plot ROC and DET curves
from sklearn.metrics import det_curve
def ROC_DET(S,ground_truth,label ="HMM ROC curve"):
    #ROC
    S_list = []
    plt.figure(figsize=(6,5))
    for case_no in range(1,2):  #For Loop for all cases
        Scores = sorted(S.flatten())   #Scores are sorted for thresholding
        S_list.append(S.flatten())

        TPR = [0]*len(Scores)   
        FPR = [0]*len(Scores) 
        count=0
        for threshold in Scores:
            TP,FP,TN,FN = 0,0,0,0
            for i in range(len(ground_truth)):
                for j in range(3):
                    if S[i][j] >= threshold:        #Classifying As Positive
                        if ground_truth[i] == j+1: TP+=1
                        else:   FP+=1
                    else:
                        if ground_truth[i] == j+1: FN+=1
                        else:   TN+=1
            TPR[count] = TP/(TP+FN)     #True Positive Rate
            FPR[count] = FP/(FP+TN)     #False Positive Rate

            count+=1
        plt.plot(FPR,TPR,label=label)
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic curve")
    plt.legend()
    plt.show()

    #DET
    y_true = []
    for i in range(5):
        for j in range(len(ground_truth)):
            if ground_truth[j] == i+1:
                y_true.append(1)
            else:
                y_true.append(0)

    plt.figure(figsize=(6,5))    
    for i in range(1):
        S = S_list[i]
        y_true = np.array(y_true)
        fpr, fnr, thresholds = det_curve(y_true, S)
        plt.plot(fpr,fnr,label=label)
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve")
    plt.show()





#Training and Testing HMM model for spoken digits
digits = [1,2,5,9,'z']
states = [35,30,50,45,40]  #[30,30,40,60,40] Number of states per digit used
symbols = 44               #Number of symbols used 
#Training HMM models for all classes using the cpp code provided and subprocesses library for easily using terminal commands
for cls in range(1,len(digits)+1): #This loop is for training of the HMM and trained models are present in Digits_Symbol_Data/train
    output=subprocess.run(["./train_hmm", "Digits_Symbol_Data/train/train_sequence"+str(cls)+"_digits.hmm.seq", "1234",str(states[cls-1]),str(symbols),"0.01"], stdout = subprocess.PIPE,universal_newlines = True).stdout
    print("HMM trained for class "+str(cls)+" and dumped in train folder of Digits_Symbol_Data")

predicted_classes = []  #Here we are testing on the dev data
scores =  [[] for i in range(5)]    #This scores 2d list is used for ROC and DET
for cls in range(1,len(digits)+1):  #This loop is for testing of the HMM on the trained models
    alpha_all = []
    for tst_cls in range(1,len(digits)+1):
        output=subprocess.run(["./test_hmm", "Digits_Symbol_Data/dev/test_sequence"+str(cls)+"_digits.hmm.seq", "Digits_Symbol_Data/train/train_sequence"+str(tst_cls)+"_digits.hmm.seq.hmm"], stdout = subprocess.PIPE,universal_newlines = True).stdout
        alpha_all.append(np.loadtxt('alphaout'))
        scores[tst_cls-1] += np.loadtxt('alphaout').tolist()
    
    predicted_classes.append((np.argmax(np.array(alpha_all),axis = 0)).tolist())  #Prediction for each dev sample is obtained

scores =np.array(scores).T
ground_truth = [1]*12 + [2]*12 + [3]*12 + [4]*12 + [5]*12  #This is the actual classes which is used to find ROC


#Finding what percentage of test images are correctly classified
count_correct = 0
count_dev_files = 0
for i in range(len(predicted_classes)):
    for j in  predicted_classes[i]:
        count_dev_files += 1
        if j == i:
            count_correct += 1
Accuracy = (count_correct/count_dev_files)*100

#Printing Accuracy and predictions and plotting confusion matrix and ROC,DET
print("Accuracy of HMM(Isolated digits) = ",Accuracy,"%")
print("The class predictions by the HMM for Isolated digits is ",predicted_classes)
Confusion_matrix(predicted_classes,"HMM on Isolated Digits")
ROC_DET(scores,ground_truth,"HMM on Isolated Digits")





######################################################################


#Training and Testing HMM model for Telugu characters
letters = ['a','bA','chA','lA','tA']
states =   [11,11,13,13,11]  #[10,12,14,14,12] Number of states per letter used
symbols = 10                 #Number of symbols used 

#Training HMM models for all classes using the cpp code provided and subprocesses library for easily using terminal commands
for cls in range(1,len(letters)+1): #This loop is for training of the HMM and trained models are present in Digits_Symbol_Data/train
    output=subprocess.run(["./train_hmm", "Letters_Symbol_Data/train/train_sequence"+str(cls)+"_letters.hmm.seq", "1234",str(states[cls-1]),str(symbols),"0.01"], stdout = subprocess.PIPE,universal_newlines = True).stdout
    print("HMM trained for class "+str(cls)+" and dumped in train folder of Letters_Symbol_Data")

predicted_classes = [] #Here we are testing on the dev data
scores =  [[] for i in range(5)]     #This scores 2d list is used for ROC and DET
for cls in range(1,len(letters)+1):  #This loop is for testing of the HMM on the trained models
    alpha_all = []
    for tst_cls in range(1,len(letters)+1):
        output=subprocess.run(["./test_hmm", "Letters_Symbol_Data/dev/test_sequence"+str(cls)+"_letters.hmm.seq", "Letters_Symbol_Data/train/train_sequence"+str(tst_cls)+"_letters.hmm.seq.hmm"], stdout = subprocess.PIPE,universal_newlines = True).stdout
        alpha_all.append(np.loadtxt('alphaout'))
        scores[tst_cls-1] += np.loadtxt('alphaout').tolist()
    
    predicted_classes.append((np.argmax(np.array(alpha_all),axis = 0)).tolist())  #Prediction for each dev sample is obtained

scores =np.array(scores).T
ground_truth = [1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20    #This is the actual classes which is used to find ROC

#Finding what percentage of test images are correctly classified
count_correct = 0
count_dev_files = 0
for i in range(len(predicted_classes)):
    for j in  predicted_classes[i]:
        count_dev_files += 1
        if j == i:
            count_correct += 1
Accuracy = (count_correct/count_dev_files)*100


#Printing Accuracy and predictions and plotting confusion matrix and ROC,DET
print("Accuracy of HMM(Telugu characters) = ",Accuracy,"%")
print("The class predictions by the HMM for Telugu characters is ",predicted_classes)
Confusion_matrix(predicted_classes,"HMM on Handwriting Data")
ROC_DET(scores,ground_truth,"HMM on Handwriting Data")