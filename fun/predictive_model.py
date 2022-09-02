import os,pickle,sys,re,glob
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.stats import uniform, randint
from os import listdir
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set_theme(color_codes=True);sns.set_style("white")
from joblib import Parallel, delayed
import itertools

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import ElasticNet,LassoCV,LassoLarsCV,lasso_path,LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV
from sklearn.feature_selection import SelectKBest,SelectPercentile,mutual_info_classif,f_classif
from sklearn.metrics import r2_score, roc_curve, auc, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression,PLSCanonical
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import xgboost as xgb

from fun import utilities as ut

class predModelClass():
    def __init__(self, cwd, datapath, outputpath, pickled_objects_dir, random_state, cpu):
        self.cwd=cwd
        self.datapath=datapath
        self.outputpath=outputpath
        self.pickled_objects_dir=pickled_objects_dir
        self.random_state=random_state
        self.cpu=cpu
    
    def scaleGroupOutcome(self,y):
        if y == 0:
            y_ = 0
        elif y == 1:
            y_ = 1 - 0.8*(630./720.)
        elif y == 2:
            y_ = 1 - 0.8*(450./720.)
        elif y == 3:
            y_ = 1 - 0.8*(270./720.)
        elif y == 4:
            y_ = 1 - 0.8*(90./720.)
        elif y == 5:
            y_ = 1.
        return str(y_)
    
    def scaleStatusOutcome(self,y):
        if y == 'ATB':
            y_ = 1
        elif y == 'HC' or y == 'LTBI' or y == 'OD' or y == 'Tret':
            y_= 0
        return y_
    
    def scaleQuantOutcome(self,y):
        if y == 'ATB':
            y_ = 2
        elif y == 'LTBI' :
            y_= 1
        elif y== 'HC':
            y_=0
        elif y== 'OD':
            y_=3
        return y_
    
    def ovlapFeaSelect(self,listData,nor_exp_data,alpha=0.8):
        #Function: determine what genes go for model training
        fcout=pd.Series()
        for e in listData.index:
            data = pd.read_csv(nor_exp_data+'/'+listData.loc[e,'File'],sep=',',index_col=0)
            data.index=[ut.genealias(x) for x in data.index]
            new=[x for x in data.index if x not in fcout.index]
            fcout=fcout.append(pd.Series(np.zeros(len(new)),index=new))
            fcout[data.index]=fcout[data.index]+1
        #Genes presenting in less than 80% of data sets were removed
        return list(fcout[fcout>=round(listData.shape[0]*alpha)].index)

    def TBexpressiondata(self, nor_exp_data,plot=1,alpha=0.8,rescale=1):
        #Function : Calculate averages of AUCs for TB disease stage differentiation given the gene sets
        
        #load all study list
        datalist = pd.read_csv(self.cwd+'/mega-data-list-model-building.csv',sep=',',index_col=0)
        
        #Collect datasets to fetch and combine
        exist=[]
        filelist=[]
        for i in datalist.index:
            tmp=datalist.loc[i,'Compare'].split('_v_')
            File1=datalist.loc[i,'GSEID']+ '_' + datalist.loc[i,'Condition1'] + '_' + datalist.loc[i,'Type'] + '_Exp_EachGene.csv'
            File2=datalist.loc[i,'GSEID']+ '_' + datalist.loc[i,'Condition2'] + '_' + datalist.loc[i,'Type'] + '_Exp_EachGene.csv'
            if File1 not in exist:
                filelist.append([datalist.loc[i,'GSEID'],tmp[0],datalist.loc[i,'Condition1'],File1])
                exist.append(File1)
            if File2 not in exist:
                filelist.append([datalist.loc[i,'GSEID'],tmp[1],datalist.loc[i,'Condition2'],File2])
                exist.append(File2)
        filelist=pd.DataFrame(filelist,columns=['GSEID','TB_Status','Condition','File'])
        
        #Determine genes
        genes=self.ovlapFeaSelect(filelist,nor_exp_data,alpha=alpha)
        #Combine the datasets
        #X:expression data, Y:Status data
        X=pd.DataFrame([],index=genes);Y=pd.DataFrame([],columns=['GSEID','GSMID','Status','Condition'])
        for i in filelist.index:
            data = pd.read_csv(nor_exp_data+'/'+filelist.loc[i,'File'],sep=',',index_col=0)
            data.index=[ut.genealias(x) for x in data.index]
            X = X.join(data)
            Y1=pd.DataFrame([],columns=['GSEID','GSMID','Status','Condition'])
            Y1['GSEID']=[filelist.loc[i,'GSEID']]*data.shape[1]
            Y1['GSMID']=data.columns
            Y1['Status']=[filelist.loc[i,'TB_Status']]*data.shape[1]
            Y1['Condition']=[filelist.loc[i,'Condition']]*data.shape[1]
            Y=pd.concat([Y, Y1],axis=0)
        Y.index=Y['GSMID']
        
        if rescale==1:
            #Rescale before meta-analysis: Take Zscore for each dataset 
            #Importantly don't correct for study, gender and comorbidity. It is preferable to include those as covariates in the statistical models
            xre = StandardScaler().fit_transform(X)
            X_rescale=pd.DataFrame(xre,index=X.index,columns=X.columns)
        else:
            X_rescale=X
            
        if plot==1:
            #PCA 
            newX=X.T
            self.PCAplot(X.loc[newX.isnull().sum()==0,:].T,Y,'PCA: before rescale','GSEID')
            self.PCAplot(X_rescale.loc[newX.isnull().sum()==0,:].T,Y,'PCA: after rescale','GSEID')
            plt.show()
        return X_rescale,Y
    
    def PCAplot(self,data,info,title,show_by):#show_by:groups showing on the legend
        #samples in rows, features in columns
        pca = PCA(n_components=2)
        PCs = pca.fit_transform(data)
        PCdf = pd.DataFrame(data = PCs,index=data.index, columns = ['PC1', 'PC2'])
        PCdf = PCdf.join(info)
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title(title, fontsize = 20)
        ax.set_xlim([-350, 600])
        ax.set_ylim([-70, 150])
        sns.scatterplot(data=PCdf, x="PC1", y="PC2", hue=show_by,palette='flare')
        ax.legend(loc='upper right',bbox_to_anchor=(1.4, 1.01))
        return
    
    def feaCombo(self,X,pickled_objects_dir,approach=1):
        #Function: generate paired gene substraction feature (A-B and B-A)
        
        if approach==1:
            #Fetch gene up or down regulation (directionality) between ATB and other conditions
            mean_logFC_AH = pd.read_pickle(pickled_objects_dir + '/network-files/mean-logFC-network-nodes-series/ATB_v_HC.pkl')
            mean_logFC_AL = pd.read_pickle(pickled_objects_dir + '/network-files/mean-logFC-network-nodes-series/ATB_v_LTBI.pkl')
            mean_logFC_AT = pd.read_pickle(pickled_objects_dir + '/network-files/mean-logFC-network-nodes-series/ATB_v_Tret.pkl')
            mean_logFC_AO = pd.read_pickle(pickled_objects_dir + '/network-files/mean-logFC-network-nodes-series/ATB_v_OD.pkl')
            up=[]
            down=[]
            for e in X.columns:
                if e in mean_logFC_AH.index:
                    up.append(e) if mean_logFC_AH[e]>0 else down.append(e)
                elif e in mean_logFC_AL.index:
                    up.append(e) if mean_logFC_AH[e]>0 else down.append(e)
                elif e in mean_logFC_AT.index:
                    up.append(e) if mean_logFC_AT[e]>0 else down.append(e)
                elif e in mean_logFC_AO.index:
                    up.append(e) if mean_logFC_AO[e]>0 else down.append(e)
                else:
                    print('Cannot find directionality for '+e);return

            #Generate paired genes from up and down groups
            new=pd.DataFrame([],index=X.index)
            for u in up:
                for d in down:
                    new[u+'_'+d]=X[u]-X[d]
                    
        elif approach==2:
            #Consider any pair comparison
            new=pd.DataFrame([],index=X.index)
            for i in range(0,len(X.columns)):
                for j in range(i+1,len(X.columns)):
                    new[X.columns[i]+'_'+X.columns[j]]=X.iloc[:,i]-X.iloc[:,j]
                    
        elif approach==3:#Consider both individual and pair
            #Consider any pair comparison
            new=pd.DataFrame([],index=X.index)
            for i in range(0,len(X.columns)):
                for j in range(i+1,len(X.columns)):
                    new[X.columns[i]+'_'+X.columns[j]]=X.iloc[:,i]-X.iloc[:,j]
                    
            new = pd.concat([X, new], axis=1)
        return new
    
    
    
#############################################################################################################################################################
# Machine-learning functions
#
#
#
#############################################################################################################################################################
    def ML_CV_param_search_framework(self, X, y, inner_cv, outer_cv, ROCAUC_filename, ROCAUC_title, randomCV_n_itr=25, mix1=0, status='', rocauc=1):
        #Choose the model with the best predictive performance
        #Function: Explore and compare multiple feature selection methods and ML models along with searching optimal hyper-parameters specific to the ML models, 
        #based on a nested CV framework.
        import timeit
        
        results=[]
        Sfea=dict()
        #Two different L1-based feature selection methods
        for fs_i in [2]:
            #Four ML models are compared
            for ml_i in ['xgboost','support_vector_machine','random_forest','elastic_net','adaboost','PLSRegression','NN_MLPRegressor']:
            #for ml_i in ['random_forest']:
                test_y=np.array([]);predict_y=np.array([])
                ###### 4.1 Split dataset into training and testing datasets
                c=1; tprs=[]; fprs=[]; aucs=[]; itp_fpr = np.linspace(0, 1, 100)
                for o_train_i, o_test_i in outer_cv.split(X, y):
                    o_train_X, o_test_X = X.iloc[o_train_i,:], X.iloc[o_test_i,:]
                    o_train_Y, o_test_Y = y[o_train_i], y[o_test_i]
                    start = timeit.default_timer()
                    
                    if str(fs_i)+'_Split'+str(c) in Sfea:#Fea selection results should be idential from each loop
                        o_train_X=o_train_X.loc[:,Sfea[str(fs_i)+'_Split'+str(c)]]
                        o_test_X=o_test_X.loc[:,Sfea[str(fs_i)+'_Split'+str(c)]]
                    else:
                        ###### 4.2 Removing feature with low variance (not appliable here)
                        ###### 4.3 Univariate feature selection
                        #Mutual information is used to measure the dependencey between each feature vs the group
                        o_train_X = self.feaSel_univariate(o_train_X, o_train_Y, percent=0.9)
                        
                        ###### 4.4 Multivariate feature selection                 
                        if mix1==1:
                            o_train_status, o_test_status = status[o_train_i], status[o_test_i]#For mixed model
                            o_train_X_fsel=o_train_X.loc[(o_train_status=='HC')|(o_train_status=='LTBI')|(o_train_status=='ATB'),:]
                            o_train_Y_fsel=o_train_Y.loc[o_train_X_fsel.index]
                            #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
                            o_train_X=pd.DataFrame(StandardScaler().fit_transform(o_train_X),index=o_train_X.index,columns=o_train_X.columns)
                            o_train_X_fsel=pd.DataFrame(StandardScaler().fit_transform(o_train_X_fsel),index=o_train_X_fsel.index,columns=o_train_X_fsel.columns)
                            o_train_X_fsel = self.feaSel_Lasso(o_train_X_fsel, o_train_Y_fsel, inner_cv, approach=fs_i)
                            #update train_X, test_X
                            o_train_X=o_train_X.loc[:,o_train_X_fsel.columns]
                            o_test_X=o_test_X.loc[:,o_train_X_fsel.columns]
                        else:
                            #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
                            o_train_X=pd.DataFrame(StandardScaler().fit_transform(o_train_X),index=o_train_X.index,columns=o_train_X.columns)
                            o_train_X = self.feaSel_Lasso(o_train_X, o_train_Y, inner_cv, approach=fs_i)
                            #update test_X
                            o_test_X=o_test_X.loc[:,o_train_X.columns]
                        #Store feas
                        Sfea[str(fs_i)+'_Split'+str(c)]=o_test_X.columns.tolist()
                        
                    ###### 4.5 Build the predictive model given a sparse feature matrix (selected features)
                    final_model, inner_cv_score, scorer = self.MLconstruction(o_train_X, o_train_Y, inner_cv, search='random', approach=ml_i, randomCV_n_itr=randomCV_n_itr)
                    stop = timeit.default_timer()
                    print(ml_i+' time: ', stop - start)  
                    ##### 4.6 Test the final model on validated dataset (outer CV)
                    o_test_Y_pred = final_model.predict(o_test_X)
                    if ml_i in ['PLSRegression','PLSCanonical']:
                        o_test_Y_pred=[float(e) for e in o_test_Y_pred]
                    outer_cv_r2 = self.rsquared(o_test_Y, o_test_Y_pred)
                    outer_cv_mean_absolute_error=mean_absolute_error(o_test_Y, o_test_Y_pred)*-1
                    outer_cv_mean_squared_error=mean_squared_error(o_test_Y, o_test_Y_pred)*-1
                    if rocauc==1:
                        outer_cv_fpr, outer_cv_tpr, outer_cv_roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(o_test_Y, o_test_Y_pred)
                        #Collect CV ROC data results
                        tprs.append(outer_cv_tpr)
                        fprs.append(outer_cv_fpr)
                        aucs.append(outer_cv_roc)
                    else:
                        outer_cv_roc=np.nan
                    results.append([fs_i,ml_i,'Split_'+str(c), inner_cv_score, outer_cv_mean_squared_error, outer_cv_mean_absolute_error, outer_cv_r2, outer_cv_roc,o_test_X.shape[1],final_model])
                    c+=1

                    #Collect Y
                    test_y=np.concatenate([test_y,np.array(o_test_Y.tolist())])
                    predict_y=np.concatenate([predict_y,o_test_Y_pred])
                    
                ##### 4.7 Generate a ROCAUC report
                fname=self.pickled_objects_dir+'/ML-param-search/'+ROCAUC_filename+'_fs_'+str(fs_i)+'_'+ml_i+'_CV_ROC_COR.pdf'
                self.rocauc_cvplot(aucs,tprs,fprs,itp_fpr,test_y,predict_y,ROCAUC_title+' \n FS:'+str(fs_i)+'|ML:'+ml_i, fname, rocauc=rocauc)
        
        benchm_model=pd.DataFrame(results,columns=['Fea selection type','ML type', 'Split', 'inner_cv_'+scorer, 'outer_cv_neg_mean_squared_error', 'outer_cv_neg_mean_absolute_error', 'outer_cv_r2', 'outer_cv_roc_auc', '# features', 'parameters'])
    
        return Sfea, benchm_model
    
    def ML_CV_param_search_framework_minFea(self, ml, X, y, inner_cv, outer_cv, ROCAUC_filename, ROCAUC_title, randomCV_n_itr=25, mix1=0,status='', rocauc=1):
        #Choose the model with minimized features while maintaining the predictive performance compared to the best model
        #Function: Given a specific ML, determine the model with minimized features while maintaining a good predictive performance.
        import timeit
        
        results=[]
        Sfea=dict()
        if ml in ['PLSRegression','PLSCanonical']:
            topgs=range(30,2,-2)[::-1]#fea select cutoff
        else:
            topgs=range(30,0,-2)[::-1]#fea select cutoff
        rocaucs=dict();MSE=dict()
        #Split dataset into training and testing datasets
        c=1; itp_fpr = np.linspace(0, 1, 100)
        for o_train_i, o_test_i in outer_cv.split(X, y):
            o_train_X, o_test_X = X.iloc[o_train_i,:], X.iloc[o_test_i,:]
            o_train_Y, o_test_Y = y[o_train_i], y[o_test_i]
                   
            start = timeit.default_timer()
            ###### 4.3 Univariate feature selection
            #Mutual information is used to measure the dependencey between each feature vs the group
            o_train_X = self.feaSel_univariate(o_train_X, o_train_Y, percent=0.9)
            
            ###### 4.4 Multivariate feature selection
            if mix1==1:
                o_train_status, o_test_status = status[o_train_i], status[o_test_i]#For mixed model
                o_train_X_fsel=o_train_X.loc[(o_train_status=='HC')|(o_train_status=='LTBI')|(o_train_status=='ATB'),:]
                o_train_Y_fsel=o_train_Y.loc[o_train_X_fsel.index]
                #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
                o_train_X_fsel=pd.DataFrame(StandardScaler().fit_transform(o_train_X_fsel),index=o_train_X_fsel.index,columns=o_train_X_fsel.columns)
                o_train_X=pd.DataFrame(StandardScaler().fit_transform(o_train_X),index=o_train_X.index,columns=o_train_X.columns)
                sel_feas,fea_rank = self.feaSel_Lasso(o_train_X_fsel, o_train_Y_fsel, inner_cv, approach=3, boostrap_top_fea=topgs)
            else:
                #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
                o_train_X=pd.DataFrame(StandardScaler().fit_transform(o_train_X),index=o_train_X.index,columns=o_train_X.columns)
                sel_feas,fea_rank = self.feaSel_Lasso(o_train_X, o_train_Y, inner_cv, approach=3, boostrap_top_fea=topgs)
            
            for e,feas in sel_feas.items():
                if e not in rocaucs:#Set the structure to store tpr,fpr,roc
                    rocaucs[e]=dict()
                    rocaucs[e]['tprs']=[]
                    rocaucs[e]['fprs']=[]
                    rocaucs[e]['aucs']=[]
                    MSE[e]=[]
                #Store feas
                Sfea['3_top'+str(e)+'_Split'+str(c)]=feas
                o_test_X_sub=o_test_X.loc[:,feas]#update test_X
                o_train_X_sub=o_train_X.loc[:,feas]#update train_X
                ###### 4.5 Build the predictive model given a sparse feature matrix (selected features)
                final_model, inner_cv_score, scorer = self.MLconstruction(o_train_X_sub, o_train_Y, inner_cv, search='random', approach=ml, randomCV_n_itr=randomCV_n_itr)
                
                ##### 4.6 Test the final model on validated dataset (outer CV)
                o_test_Y_pred = final_model.predict(o_test_X_sub)
                if ml in ['PLSRegression','PLSCanonical']:
                    outer_cv_r2 = self.rsquared(o_test_Y, [float(e) for e in o_test_Y_pred])
                else:
                    outer_cv_r2 = self.rsquared(o_test_Y, o_test_Y_pred)
                outer_cv_mean_absolute_error=mean_absolute_error(o_test_Y, o_test_Y_pred)*-1
                outer_cv_mean_squared_error=mean_squared_error(o_test_Y, o_test_Y_pred)*-1
                if rocauc==1:
                    outer_cv_fpr, outer_cv_tpr, outer_cv_roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(o_test_Y, o_test_Y_pred)
                    #Collect CV ROC data results
                    rocaucs[e]['tprs'].append(outer_cv_tpr)
                    rocaucs[e]['fprs'].append(outer_cv_fpr)
                    rocaucs[e]['aucs'].append(outer_cv_roc)
                else:
                        outer_cv_roc=np.nan
                results.append(['3_top'+str(e),ml,'Split_'+str(c), inner_cv_score, outer_cv_mean_squared_error, outer_cv_mean_absolute_error, outer_cv_r2, outer_cv_roc,len(feas),final_model])
                MSE[e].append(outer_cv_mean_squared_error)
            stop = timeit.default_timer()
            print(ml+'|split'+str(c)+' time: ', stop - start)  
            c+=1
        fname=self.pickled_objects_dir+'/ML-param-search/'+ROCAUC_filename+'_fs_3_'+ml+'_CV_ROC_feasel.pdf'
        self.rocaucs_feaselection(topgs, rocaucs, MSE, c-1, fname, ROCAUC_title, rocauc)
        #for e,feas in sel_feas.items():
        #    ##### 4.7 Generate a ROCAUC report
        #    fname=self.pickled_objects_dir+'/ML-param-search/'+ROCAUC_filename+'_fs_3_top'+str(e)+'_'+ml+'_CV_ROC.pdf'
        #    self.rocauc_cvplot(rocaucs[e]['aucs'],rocaucs[e]['tprs'],rocaucs[e]['fprs'],itp_fpr,ROCAUC_title+' \n FS:3_top'+str(e)+'|ML:'+ml, fname)
        
        benchm_model=pd.DataFrame(results,columns=['Fea selection type','ML type', 'Split', 'inner_cv_'+scorer, 'outer_cv_neg_mean_squared_error', 'outer_cv_neg_mean_absolute_error', 'outer_cv_r2', 'outer_cv_roc_auc', '# features', 'parameters'])
    
        return Sfea, benchm_model
    
    def feaSel_univariate(self, X, y, percent=0.9):
        #Function-feature selection: keep top percent of the features who are correlated to the outcome
        Mi = mutual_info_classif(X,y)
        rank=np.argsort(-1*Mi)#Descending the rank
        feas=X.columns[rank][0:int(0.9*len(Mi))]
        X=X.loc[:,feas]
        return X
    
    def feaSel_Lasso(self, X, y, cv, approach=2, n_bootstraps=1000, boostrap_top_fea=[10], plot=0):
        #Function-feature selection: A L1-based feature selection, so l1_ratio is set up to be 1
        
        if approach==1:#Regular Lasso
            #use LassoCV (same idea but much faster)
            reg = LassoCV(cv=cv, n_alphas=500, max_iter=int(1e6), tol=0.002, n_jobs=self.cpu).fit(X, y)
            #print(reg.alpha_);#print(reg.alphas_)
            newX=X.loc[:,reg.coef_!=0]
            #print(newX.shape)
            return newX
        elif approach==2:#Lasso with LARS
            #Lasso with LARS algorithm 
            regLars = LassoLarsCV(cv=cv, max_iter=int(1e6), n_jobs=self.cpu).fit(X, y)
            #print(reg.alpha_);
            newX=X.loc[:,regLars.coef_!=0]
            #print(newX.shape)
            return newX
        elif approach==3:#Bootstrap Lasso
            # resampling the train data and computing a Lasso on each resampling. In short, the features 
            # selected more often are robust features.
            n_samples_in_bootstrap = int(X.shape[0]/2)
            alphas=np.logspace(-4, 1, num=100)
            paths = Parallel(n_jobs=self.cpu)(
                delayed(self.fit_bootstrap_sample)(X, y, alphas, n_samples_in_bootstrap) for _ in range(n_bootstraps))
            
            #Calculate stability score
            new_alphas=np.logspace(-4,1,num=500)
            stab_scores=np.zeros((X.shape[1],len(new_alphas)))
            for each in paths:
                stab_scores+=(each(new_alphas)!=0)
            #Calculate the probability of feature selection from iterative bootstrappings 
            stab_scores /= n_bootstraps
            #Calculate the final scores for each feature
            overall_scores=np.amax(stab_scores, axis=1)
            
            s_fea=dict()
            for e in boostrap_top_fea:
                #Rank the score and choose the top N gene with the highest score
                select_ind=np.argsort(overall_scores)[::-1][:e]
                s_fea[e]=X.columns[select_ind].tolist()
                
            a_fea=dict()
            #Rank the score and choose the top N gene with the highest score
            select_ind=np.argsort(overall_scores)[::-1]
            a_fea=zip(X.columns[select_ind].tolist(),overall_scores[select_ind])
            if plot==1:
                plt.figure()
                for i in range(0,stab_scores.shape[0]):
                    if i in select_ind :
                        alpha=1;linestyle='-';color='blue'
                    else:
                        alpha=.1;linestyle='--';color='grey'
                    plt.plot(np.log10(new_alphas), stab_scores[i,:], alpha=alpha, linestyle=linestyle, color=color, lw=1)
                    plt.xlabel('$log_{10}(\lambda)$')
                    plt.ylabel('Stability score (select probability)')
                #plt.legend(loc="lower right")
                plt.show()
            return s_fea,a_fea
    
    def fit_bootstrap_sample(self, X, y, alphas, n_samples_in_bootstrap):
        #Function : generate lasso paths based on bootstraping  
        X_, y_ = resample(X, y, n_samples=n_samples_in_bootstrap,replace=False,stratify=y)
        #alphas_, coefs_, _ = lasso_path(X_, y_, fit_intercept=False,alphas=alphas)
        alphas_, coefs_, _ = lasso_path(X_, y_,alphas=alphas)
        interpolator = interp1d(alphas_[::-1], coefs_[:, ::-1],
                                bounds_error=False, fill_value=0.)
        return interpolator
    
   
    def MLconstruction(self, X, y, cv, search='grid', approach='', scorer='neg_mean_squared_error', randomCV_n_itr=25):
        #Function : Create and optimize ML model(s) and select the best model if multiple models are tested 
        #Ml settings
        #scoring parameter : r2, roc_auc, neg_mean_squared_error, neg_mean_absolute_error
        models=self.MLparameters(self.random_state, X, scorer)
        
        #Specify which approach 
        m=models[approach]['model']
        param_grid = {k : v for k, v in models[approach]['params'].items()}
        
        #Grid or randomized search
        if search == 'grid':
            search_model = GridSearchCV(
                    estimator = m, n_jobs=self.cpu, refit=True,
                    param_grid = param_grid, cv=cv,
                    scoring = models[approach]['scorer'])
        else:
            search_model = RandomizedSearchCV(
                    estimator = m, n_jobs=self.cpu, refit=True, n_iter=randomCV_n_itr,
                    param_distributions = param_grid, cv=cv,
                    scoring = models[approach]['scorer'])
            #search_model = HalvingRandomSearchCV(
            #        estimator = m, n_jobs=self.cpu, refit=True, random_state=self.random_state,
            #        param_distributions = param_grid, cv=cv,
            #        scoring = models[approach]['scorer'])
        
        search_model.fit(X, y)
        #print(search_model.cv_results_)
        best_score = search_model.best_score_
        final_model=search_model.best_estimator_
        
        #Update the model to avoid any missing parameters
        update=search_model.cv_results_['params'][np.argwhere(search_model.cv_results_['mean_test_score']==search_model.best_score_)[0][0]]
        final_model.set_params(**update)
        final_model.fit(X,y)
        
        return final_model, best_score, scorer
    
    def MLparameters(self,random_state, X, scorer):
        #Function: Define a set of regressor models, their hyperparameter search spaces, and the evaluation metric
        #scoring parameter : r2, roc_auc, neg_mean_squared_error, neg_mean_absolute_error (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        
        regressors = {
            'random_forest': {
                'model': RandomForestRegressor(n_jobs=self.cpu, random_state=random_state),
                'params': {
                    #number of trees in the foreset
                    'n_estimators': [int(x) for x in np.linspace(100, 2000, num = 20)],
                    #max number of features considered for splitting a node
                    'max_features': ['auto', 'sqrt', 'log2'],
                    #max number of levels in each decision tree
                    'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
                    #min number of data points placed in a node before the node is split
                    'min_samples_leaf': [1,2,4],
                    #min number of data points allowed in a leaf node
                    'min_samples_split': [2,5,10],
                    #method for sampling data points (with or without replacement)
                    'bootstrap': [True, False]
                },
                'scorer': scorer
            },

            'elastic_net': {
                'model': ElasticNet(max_iter=int(1e6), tol=0.01, random_state=random_state),
                'params': {
                    'alpha': np.logspace(-4, 0, num=100), 
                    'l1_ratio': np.arange(0.1, 1.1, 0.1)
                },
                'scorer': scorer
            },

            'linear_regression': {
                'model': LinearRegression(n_jobs=self.cpu),
                'params': {},
                'scorer': scorer
            },

            'support_vector_machine': {
                'model': SVR(max_iter=int(1e6),tol=0.1,cache_size=1000),
                'params': {
                    'kernel': ['linear', 'poly', 'rbf'],
                    'gamma':  np.logspace(-5, 1, num=7),
                    'C': np.logspace(-5, 3, num=9),
                    #'epsilon':[0.1,0.25,0.5]
                },
                'scorer': scorer
            },

            'adaboost': {
                'model': AdaBoostRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [int(x) for x in np.linspace(10, 1000, num = 10)],
                    'learning_rate':np.logspace(-2, 1, num=5),
                    'loss': ['linear', 'square', 'exponential']
                },
                'scorer': scorer
            },
            
            'PLSRegression': {
                'model': PLSRegression(max_iter=int(1e6), tol=0.002),
                'params': {
                    'n_components': [int(x) for x in range(2,min([16, X.shape[0],X.shape[1]]))]
                },
                'scorer': scorer
            },
            
            'PLSCanonical': {
                'model': PLSCanonical(max_iter=int(1e6), tol=0.002),
                'params': {
                    #'n_components': [int(x) for x in range(2,min([20, X.shape[0],X.shape[1]]))]
                    'n_components': [1,2], #min(n_samples, n_features, n_targets)
                    'algorithm': ['nipals', 'svd']
                },
                'scorer': scorer
            },
            
            'NN_MLPRegressor': {
                'model': MLPRegressor(random_state=random_state, max_iter=int(1e6)),
                'params': {
                    'hidden_layer_sizes': [x for x in itertools.product((1,10,25,50,100,200),repeat=3)] + 
                        [x for x in itertools.product((1,10,25,50,100,200),repeat=4)],
                    'activation': ['relu','tanh','logistic','identity'],
                    'alpha': np.logspace(-1, 1, num=20),
                    'learning_rate': ['constant','adaptive','invscaling'],
                    'solver': ['sgd', 'adam']
                },
                'scorer': scorer
            },
            
            'xgboost':{
                'model': xgb.XGBRegressor(objective="reg:linear", n_jobs=self.cpu, random_state=random_state),
                'params': {
                    'colsample_bytree': uniform(0.7, 0.3),
                    'gamma': uniform(0, 0.5),
                    'learning_rate': uniform(0.03, 0.3), # default 0.1 
                    'max_depth': randint(2, 6), # default 3
                    'n_estimators': randint(100, 150), # default 100
                    'subsample': uniform(0.6, 0.4)
                },
                'scorer': scorer
            }
        }
        
        return regressors
    
    def rsquared(self, x, y):
        #Function: calculate R^2 between two variables
        """ Return R^2 where x and y are array-like."""
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        return r_value**2
    
    def rocauc_cvplot(self, aucs, tprs, fprs, mean_fpr, y, predict_y, title, save, rocauc=1):
        #Function: generate a roc plot based on the result of cross-validation 
        if rocauc==1:
            fig, axs = plt.subplots(1,2,figsize=(15,8))
            ax=axs[0]
            ipt_tprs=[]
            for i in range(0,len(aucs)):
                interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
                interp_tpr[0] = 0.0         
                ipt_tprs.append(interp_tpr)
                ax.plot(fprs[i], tprs[i], color='grey',
                    label=r'ROC fold %2d (AUC = %0.2f)' % (i+1, aucs[i]),  linewidth=0.7, alpha=.5)

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
            mean_tpr = np.mean(ipt_tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(ipt_tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
            ax.legend(loc="lower right")
            
            ax=axs[1]
        else:
            fig, ax = plt.subplots(1,1,figsize=(8,8))
        
        pr,rpval = scipy.stats.spearmanr(y,predict_y)
        sns.regplot(x=y, y=predict_y, color='g',ci=95,truncate=False,ax=ax)
        sns.scatterplot(x=y,y=predict_y, legend=False, s=100,ax=ax)
        ax.annotate("Spearman $\itr$ = {:.2f}".format(pr) + "\n$\itp$-value = {:.4f}".format(rpval),xy=(.05, .78), xycoords=ax.transAxes, fontsize=15)
        ax.set_xlabel('Actual', fontsize=10, weight='bold')
        ax.set_ylabel('Predicted', fontsize=10, weight='bold')
        ax.set_title('Actual vs predicted')
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save)
        plt.show()
        
        return
    
    def rocaucs_feaselection(self, topgs, aucs, mses, split, fname, title, rocauc):
        #Function: generate a rocaucs based on the result of cross-validation along with different number of top selected features 
        if rocauc==1:
            #Create a table
            results=[]
            for e in topgs:
                for i in range(0,split):
                    results.append([e,'split_'+str(i+1),aucs[e]['aucs'][i], mses[e][i]])
            T=pd.DataFrame(results,columns=['topFea','split','auc','mse'])
            fig, axs =  plt.subplots(1, 2,figsize=(15,8));
            ax=axs[0]
            sns.lineplot(data=T,x="topFea", y="auc", ci=95, err_style='band', dashes=False, ax=ax)
            plt.xlabel('# top features')
            plt.ylabel('roc auc')
            plt.title(title+' CV ROCAUCs')
            ax=axs[1]
            sns.lineplot(data=T,x="topFea", y="mse", ci=95, err_style='band', dashes=False, ax=ax)
            plt.xlabel('# top features')
            plt.ylabel('MSE')
            plt.title(title+' CV MSEs')
        else:
            #Create a table
            results=[]
            for e in topgs:
                for i in range(0,split):
                    results.append([e,'split_'+str(i+1), mses[e][i]])
            T=pd.DataFrame(results,columns=['topFea','split','mse'])
            fig, ax =  plt.subplots(figsize=(10,7));
            sns.lineplot(data=T,x="topFea", y="mse", ci=95, err_style='band', dashes=False, ax=ax)
            plt.xlabel('# top features')
            plt.ylabel('MSE')
            plt.title(title+' CV MSEs')
            
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(fname)
        plt.show()
        return
    
    def build_finalmodel(self, X, y, ml, fs, cv, topN=10, mix1=0,status=''):
        #Function: Build the final using all training datasets, the model will be validated on the independent datasets
        fea_rank=[]
        #######Mutual information is used to measure the dependencey between each feature vs the group
        X = self.feaSel_univariate(X, y, percent=0.9)
        
        ###### Multivariate feature selection
        if mix1==1:
            X_fsel=X.loc[(status=='HC')|(status=='LTBI')|(status=='ATB'),:]
            y_fsel=y.loc[X_fsel.index]       
            #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
            X_fsel=pd.DataFrame(StandardScaler().fit_transform(X_fsel),index=X_fsel.index,columns=X_fsel.columns)
            X=pd.DataFrame(StandardScaler().fit_transform(X),index=X.index,columns=X.columns)
            if fs==2:
                X_fsel = self.feaSel_Lasso(X_fsel, y_fsel, cv, approach=fs)
                #Store feas
                select_fea=X_fsel.columns.tolist()
            elif fs==3:
                fe,fea_rank = self.feaSel_Lasso(X_fsel, y_fsel, cv, approach=fs,boostrap_top_fea=[topN],plot=1)
                select_fea=fe[topN]
            X=X.loc[:,select_fea]
        else:
            #Standardize the variance of the features before feature selection. Avoid to likely choose the features with larger variance
            X=pd.DataFrame(StandardScaler().fit_transform(X),index=X.index,columns=X.columns)
            if fs==2:
                X = self.feaSel_Lasso(X, y, cv, approach=fs)
                #Store feas
                select_fea=X.columns.tolist()
            elif fs==3:
                fe,fea_rank = self.feaSel_Lasso(X, y, cv, approach=fs,boostrap_top_fea=[topN],plot=1)
                select_fea=fe[topN]
                X=X.loc[:,select_fea]
            
        print('Number of selected features:'+str(len(select_fea)))
        ###### Build the final model with a grid search
        final_model, inner_cv_score, scorer = self.MLconstruction(X, y, cv, search='random', approach=ml, randomCV_n_itr=1000)
        
        return select_fea, final_model, inner_cv_score, scorer, fea_rank