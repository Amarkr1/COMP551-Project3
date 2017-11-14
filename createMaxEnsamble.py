# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:03:13 2017

@author: akumar47
"""
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import Counter
#from predictScore import get_score
import itertools
from shutil import copyfile
import time
#import shutil
start_time = time.time()
#deleting all the files in a folder
#shutil.rmtree('C:\\Users\\akumar47\\Desktop\\Project3\\output\\GPU\\ensembleFiles\\')

path = "C:\\Users\\akumar47\\Desktop\\Project3\\ref_files\\"
path2 = "C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\ensemble_ens\\"
#path = "C:\\Users\\akumar47\\Desktop\\Project3\\predict submission\\top_10\\"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if '.csv' in f]
for i in range(1,len(onlyfiles)+1):
    filestoSearch = map(list, list(itertools.combinations(onlyfiles, i)))
    print('Ensamble predictions starts--> i: {0}'.format(i))
    if(i==1):
        for file in filestoSearch:
            copyfile(path+file[0],path2+file[0] )
    else:
#        print(filestoSearch)
        for j,files in enumerate(filestoSearch):
            predict_file = open(path2+"Ensamble{0}_{1}.csv".format(i,j+1),'w')
            predict_file.write('Id,Label\n')
            array = np.array([])
            w = []
            for file in files:
                p1 = pd.read_csv(path+file, sep=',', header= 0 , dtype = {'Id':int ,'Label':int}) 
                l1 = np.array(p1['Label'])
                if(len(array)==0):
                    array = l1
                else:
                    array = np.vstack([array,l1])
                   
                        
            for val in range(array.shape[1]):
                dec = list(array[:,val])
                most_common,num_most_common = Counter(dec).most_common(1)[0]
                predict_file.write('{0},{1}\n'.format(val+1,most_common))
                
            predict_file.close()
print("--- %s seconds ---" % (time.time() - start_time))    