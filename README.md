######################################################################################################
						PROJECT 2: COMP 551 - LANGUAGE CLASSIFICATION
						author: AMAR KUMAR[amar.kumar@mail.mcgill.ca]
								Kamila Mustafina[kamila.mustafina@mail.mcgill.ca]
								Tyler Kolody [tyler.kolody@mail.mcgill.ca]
						team name : Attack on TitanX
######################################################################################################
File descriptions:
---------------------------------------------
*** LR.ipynb
input --> 	train_x.csv
			train_y.csv
			test_x.csv
output -->	predict_LR.csv

details -->	The file is written in Jupyter notebook and contains all the models that have been mentioned in the report. The input files are same as
			the ones provided on the Kaggle and need to placed in a folder called input in the same path where this code is present. Default values of Logistic regression is used. The *.py file of this version is 
			also given by the filename 'LR.py'
---------------------------------------------
*** createMaxEnsamble.py
input/variables to be modified --> 	path = path to the reference file whose scores are known
									path2 = path of the folder where all the combination of ensembles created can be stored
									
output -->	<Ensembles>.csv - a collection of lot of ensemble files in the folder specified in path2

details -->	The file is written in Python 3.6 and is used to generate the ensembles from a given set of reference files
---------------------------------------------
*** predictScore.py
input/variables to be modified --> 	path = path to the reference file whose scores are known
									file_scores = the location of file which will contain scores of the reference files
									path2 = path of the folder where all the combination of ensembles created was stored. [This should be same as path2 of the file createMaxEnsamble.py]
									scoreFile = the path of the main file which will store the information about the scores of all the ensembles created
output -->	scoreFile.csv

details -->	The file is written in Python 3.6 and is used to generate the scoreFile once ensembles are generated
---------------------------------------------
*** plot_CM_arch3.m
input --> 	cm_arch3.mat
output -->	figure showing the detail analysis of the confusion matrix 

details -->	This is a matlab file which gives a detail analysis of the confusion matrix given to it. Infact in this case the confusion matrix from Arch3 is given.
---------------------------------------------
*** cm_arch3.mat - This is the confusion matrix for the file 'plot_CM_arch3.m'
---------------------------------------------
*** Final.ipynb
input --> 	train_x.csv
			train_y.csv
			test_x.csv
output -->	<filename>.csv 
variables to be modified --> 	slnum = serial number of the output files
								model = which model to select from the 4 given models
								user is free to choose the normalization functions either normalize1 or normalize2

details -->	The file is written in Jupyter notebook and contains all the models that have been mentioned in the report. The input files are same as
			the ones provided on the Kaggle and need to placed in a folder called input in the same path where this code is present. 
			The file contains all the model of the architectures described in the report. The *.py file of this version is 
			also given by the filename 'Final.py'
---------------------------------------------
*** Final-one_hot.ipynb
input --> 	train_x.csv
			train_y.csv
			test_x.csv
output -->	<filename>.csv 
variables to be modified --> 	slnum = serial number of the output files
								model = which model to select from the 4 given models
								user is free to choose the normalization functions either normalize1 or normalize2

details -->	The file is written in Jupyter notebook and contains all the models that have been mentioned in the report. The input files are same as
			the ones provided on the Kaggle and need to placed in a folder called input in the same path where this code is present. 
			This is same as the file 'Final.ipynb' except that a different method of implementing one hot vector as explained in the report is used. The *.py file of this version is 
			also given by the filename 'Final-one_hot.py'

---------------------------------------------
*** FeedForwardNetwork.py
input/variables to be modified --> 	mnist_images.pkl, mnist_labels.pkl, any other pkl file pairs
					If you wish to try to classify the project dataset, comment out the lower definition of 'classes'

output -->	"mnistResults.csv". 
		Will also print the first 100 predicted and actual weights, as well as the loss and accuracy every 100 epocs

details -->	The file is written in Python 3.6
---------------------------------------------
*** FeedForwardNetwork.py
input/variables to be modified --> 	mnist_images.pkl, mnist_labels.pkl, any other pkl file pairs
					If you wish to try to classify the project dataset, comment out the lower definition of 'classes'

output -->	"mnistOldBiasResults.csv". Will print results using the original bias, which does not update. 
		Will also print the first 100 predicted and actual weights, as well as the loss and accuracy every 10 epocs

details -->	The file is written in Python 3.6
---------------------------------------------
*** imageProcessing.py
input/variables to be modified --> train_x.csv, train_y.csv, test_x.csv, train_digits.csv, train_letters.csv
		To get the histogram, segmented or generated images, uncomment the appropriate lines in the body of the code. 
		To see segmented images, set showPlot to True. To show gnerated images, uncomment lines 179 and 189
	
output -->	GeneratedData.csv, segmentedImages.csv, training data histogram

details -->	The file is written in Python 3.6
---------------------------------------------
---------------------------------------------

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''{All the python codes are compatible with Python 2.7/3.6 unless specified.}''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
