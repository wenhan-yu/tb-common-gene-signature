#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import os,pickle,sys,re,glob
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from collections import Counter
from importlib import reload
#customized module
from fun import predictive_model, validation
from fun import utilities as ut

cwd = os.getcwd()
datapath='/efs/bioinformatics/projects/tb-gene-signature/working-folder'
outputpath=cwd+'/data/update'
valdata_dir = datapath+'/../validation-dataset'
#viral infection data directory
vinf_dir = datapath+'/../validation-dataset/viral-infection-dataset'
#final model directory
final_ml_dir = datapath+'/pickled-files/final-ML-model/opt_score_rocauc_20220224'
pickled_objects_dir = datapath+'/pickled-files'
#data to be shared in the publication
publication_dir=outputpath

#For coding, if necessary
random_state=1#for creating randomized model
cpu=60 #cpu=-1 use all processors
reload(predictive_model);reload(ut)
gpmod=predictive_model.predModelClass(cwd,datapath,outputpath,pickled_objects_dir,random_state,cpu)
gval=validation.valModelClass(cwd,datapath,outputpath,random_state,valdata_dir)

rescale=1


reload(ut)
files=['GSE17156_GPL571','GSE117827_GPL23126']

for f in files:
    print('process '+f)
    data = pd.read_csv(vinf_dir+'/'+f+'_array_Exp_EachGene.csv',sep=',',index_col=0)
    newGene=[]
    for gene in data.index:
        upd_gene = ut.get_official_gene_symbol(gene)
        if upd_gene=='Gene not found':
            newGene.append(gene)
        else:
            newGene.append(upd_gene)
    data.index=newGene
    data.to_csv(vinf_dir+'/update/'+f+'_array_Exp_EachGene.csv',index=True)

