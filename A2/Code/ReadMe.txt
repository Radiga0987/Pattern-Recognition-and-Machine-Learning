EE19B135 Rishabh Adiga
EE19B134 Amogh Patil

The "Code" folder contains 2 .py files.The regression.py file is for Part A regression and the bayesian.py is for Part B Bayesian classifier.


regression.py
The code can be directly run assuming the data is stored in the following format.
|---"Code" Folder
   |--- regression.py + "Data" folder
				  |----- 1d_team_10_train.txt 
					   +1d_team_10_dev.txt
					   +2d_team_10_train.txt
					   +2d_team_10_dev.txt

If datasets are stored in this relative format to the regression.py file, the code will work.
Note:
At the end of the code, under if __name__ == "__main__":,
several functions have been called for different plots.These can be commented out if any specific plots need to be checked.




bayesian.py
The code can be directly run assuming the data is stored in the following format.
|---"Code" Folder
     |--- bayesian.py + "Data" folder
				  |----- "LinearlySeperable" folder
					    |----- dev.txt+trian.txt
					
				  |----- "NonLinearlySeperable" folder
				  	    |----- dev.txt+trian.txt

				  |----- "RealData" folder
				  	    |----- dev.txt+trian.txt

At the end of the code, under if __name__ == "__main__":,
several functions have been called for different plots.These can be commented out if any specific plots need to be checked.
