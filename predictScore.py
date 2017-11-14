# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:28:44 2017

@author: akumar47
"""
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVR
import time
#import shutil
start_time = time.time()

def get_score(filename):
    path = "C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\ref_files\\"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    file_scores = open("C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\Scores.csv",'w')
    file_scores.write('Name,Score\n')
    scores = [x.split('_')[-1].replace('.csv','') for x in onlyfiles]
    
    for i in range(len(onlyfiles)):
        file_scores.write('{0},{1}\n'.format(onlyfiles[i],scores[i]))
    file_scores.close()
    
    train_X = []
    for file in onlyfiles:
    #    print (file)
        df = pd.read_csv(path+file, sep=',', header= 0 , dtype = {'Id':int ,'Label':int}) 
        train_X.append(list(df['Label'].values))
    
    train_Y = pd.read_csv("Scores.csv", sep=',', header= 0 , dtype = {'Score':float}) 
    train_Y = list(train_Y['Score'].values)
    
   
    test_X = pd.read_csv(filename, sep=',', header= 0 , dtype = {'Id':int ,'Label':int}) 
    test_X = list(test_X['Label'].values)
#    print('Filename: '+onlyfiles2[0])
    
    svr_poly = SVR(kernel='poly', C=1e3, degree=5)
    svr_poly.fit(train_X,train_Y)
    predictions = svr_poly.predict(np.array(test_X).reshape(1, -1))
    return predictions[0]

#path2 = "C:\\Users\\akumar47\\Desktop\\Project3\\output\\GPU\\ensembleFiles\\"
path2 = 'C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\ensemble_ens\\'
onlyfiles2 = [f for f in listdir(path2) if isfile(join(path2, f)) and '.csv' in f]
scoreFile = open('C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\scoreFile.csv','w')
scoreFile.write('Name,Score\n')
maxScore = 0
maxFile = ''
for files in onlyfiles2:
    score = get_score(path2+files)
    if(score>maxScore):
        maxScore = score
        maxFile = path2+files
    scoreFile.write("{0},{1}\n".format(files,score))
scoreFile.close()
print("--- %s seconds ---" % (time.time() - start_time))


#scores = pd.read_csv("scoreFile.csv")
#sc = scores.sort_values('Score',ascending = False)
#sc['Score'].values[0:10]
#name = sc['Name'].values[0:10]
#for n in name:
#    copyfile("C:\\Users\\akumar47\\Desktop\\Project3\\output\\GPU\\ensembleFiles\\"+n,"C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\top_10\\"+n)

#    print ("{0}\t\t\t{1}".format(files,get_score(path2+files)))
#    print ("SVR",predictions[0])

#reg = linear_model.LassoLars(alpha=.1)
#reg.fit(train_X,train_Y)  
#predictions = reg.predict(np.array(test_X).reshape(1, -1))
#print ('Linear Regression',predictions)
