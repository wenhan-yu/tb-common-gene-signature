import os,pickle,sys,re,glob
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import interp1d
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
from os import listdir
from collections import Counter
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set_theme(color_codes=True);sns.set_style("white")
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet,LassoCV,LassoLarsCV,lasso_path,LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectKBest,SelectPercentile,mutual_info_classif,f_classif
from sklearn.metrics import r2_score, roc_curve, auc, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression,PLSCanonical
from sklearn.svm import SVR
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother

from fun import utilities as ut

class valModelClass():
    def __init__(self,cwd,datapath,outputpath,random_state,valdata_dir):
        self.cwd=cwd
        self.datapath=datapath
        self.outputpath=outputpath
        self.random_state=random_state
        self.valdata_dir=valdata_dir
    
    ######General functions1 - prediction performance matrics
    #########################################################################################################################################################################
    ### Model performance evaluation metrics
    #https://www.ncbi.nlm.nih.gov/books/NBK430867/
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7500131/
    def roc_auc_ci(self, AUC, y_true, positive=1):
        #Function: calculate ROCAUC 95% confidence interval
        n1 = sum(np.array(y_true) == positive) #disease cases
        n0 = sum(np.array(y_true) != positive) #no disease cases
        Q1 = AUC / (2 - AUC)
        Q2 = 2*AUC**2 / (1 + AUC)
        SE_AUC = sqrt((AUC*(1 - AUC) + (n1 - 1)*(Q1 - AUC**2) + (n0 - 1)*(Q2 - AUC**2)) / (n1*n0))
        lower = AUC - 1.96*SE_AUC
        upper = AUC + 1.96*SE_AUC
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return lower, upper
    
    def sensitivity(self, ys, cutoff):
        #sensitivity= TP/(TP+FN)
        #95% CIs of Se = Se +/- 1.96*squrt(Se*(1-Se)/n1)  [n1:disease cases]
        disease=ys.loc[ys['group']==1,:]
        control=ys.loc[ys['group']==0,:]
        n1=disease.shape[0];n0=control.shape[0]
        TP=disease.loc[disease['Y_pred']>=cutoff,:].shape[0]
        FN=disease.loc[disease['Y_pred']<cutoff,:].shape[0]
        Se=TP/(TP+FN)
        SE_Se= sqrt(Se*(1-Se)/n1)
        lower = Se - 1.96*SE_Se
        upper = Se + 1.96*SE_Se
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return Se, lower, upper
    
    def specificity(self, ys, cutoff):
        #specificity= TN/(TN+FP)
        #95% CIs of Sp = Sp +/- 1.96*squrt(Sp*(1-Sp)/n0)  [n0:no disease cases]
        disease=ys.loc[ys['group']==1,:]
        control=ys.loc[ys['group']==0,:]
        n1=disease.shape[0];n0=control.shape[0]
        TN=control.loc[control['Y_pred']<cutoff,:].shape[0]
        FP=control.loc[control['Y_pred']>=cutoff,:].shape[0]
        Sp=TN/(TN+FP)
        SE_Sp= sqrt(Sp*(1-Sp)/n0)
        lower = Sp - 1.96*SE_Sp
        upper = Sp + 1.96*SE_Sp
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return Sp, lower, upper
    
    def PPV(self, ys, prev, cutoff):
        #PPV and NPV are dependent on both underlying prevalence of disease in the population and the intrinsic accuracy of the model (sensitivity and specificity)
        #prevalence could be considered similar to the pre-test probability.  
        #PPV:the probability that a positive test correctly predicts the prescence of disease
        #PPV = (Se*P) / [(Se*P)+(1-Sp)*(1-P)]
        #95% CIs of PPV = PPV +/- 1.96*squrt(Var(PPV)) 
        #Var(PPV)=[P*(1-Sp)*(1-P)]^2*(Se*(1-Se)/n1)+[P*Se*(1-P)]^2*(Sp*(1-Sp)/n0) / [(Se*P)+(1-Sp)*(1-P)]^4
        disease=ys.loc[ys['group']==1,:]
        control=ys.loc[ys['group']==0,:]
        n1=disease.shape[0];n0=control.shape[0]
        Se,_,_=self.sensitivity(ys,cutoff)
        Sp,_,_=self.specificity(ys,cutoff)
        if Se==0:
            PPV=0;var_ppv=0
        else:
            PPV=(Se*prev) / ((Se*prev) + ((1-Sp)*(1-prev)))
            var_ppv=(prev*(1-Sp)*(1-prev))**2 * (Se*(1-Se)/n1)+(prev*Se*(1-prev))**2 * (Sp*(1-Sp)/n0) / ((Se*prev)+((1-Sp)*(1-prev)))**4
        lower = PPV - 1.96*sqrt(var_ppv)
        upper = PPV + 1.96*sqrt(var_ppv)
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return PPV, lower, upper
    
    def NPV(self, ys, prev, cutoff):
        #NPV: the probability that a negative test correctly predicts the absense of disease
        #NPV = Sp*(1-P) / [Sp*(1-P)+(1-Se)*P]
        #95% CIs of NPV = NPV +/- 1.96*squrt(Var(NPV)) 
        #Var(NPV)=[Sp*P*(1-P)]^2*(Se*(1-Se)/n1)+[(1-Se)*P*(1-P)]^2*(Sp*(1-Sp)/n0) / [Sp*(1-P)+(1-Se)*P]^4
        disease=ys.loc[ys['group']==1,:]
        control=ys.loc[ys['group']==0,:]
        n1=disease.shape[0];n0=control.shape[0]
        Se,_,_=self.sensitivity(ys,cutoff)
        Sp,_,_=self.specificity(ys,cutoff)
        NPV=(Sp*(1-prev)) / ((Sp*(1-prev)) + ((1-Se)*prev))
        var_npv=(Sp*prev*(1-prev))**2 * (Se*(1-Se)/n1)+((1-Se)*prev*(1-prev))**2 * (Sp*(1-Sp)/n0) / ((Sp*(1-prev))+((1-Se)*prev))**4
        lower = NPV - 1.96*sqrt(var_npv)
        upper = NPV + 1.96*sqrt(var_npv)
        if lower < 0:
            lower = 0
        if upper > 1:
            upper = 1
        return NPV, lower, upper
        
    def prognosis_metrics(self,Name,pred,Youden_cutoffs, predefine_range=1,t_interval=2):
        #Function: calculate sensitivity, specificity, PPV and NPV
        #Two different pre-specified cutoffs are used
        #1) Maximal Youden Index
        #2) 97.5th percentile of the IGRA-negative control population
        #t_interval=2: overlapped time interval, t_interval=1: exclusive time interval
        
        noprogress=pred.loc[pred.group==0,:]
        progress=pred.loc[pred.group!=0,:]
        progress['Time_to_TB']=[int(e) for e in progress['Time_to_TB']]
        
        #For 95th percentile of the IGRA-negative control population
        #Identify the distribution best fitted to the data and find the 95th percentile of the distribution
        #pdf, cdf, best_dist=self.fit_score_dist(noprogress['Y_pred'], '', bins=100, plot=0, predefine_range=predefine_range)
        #tmp=cdf.index[cdf>=0.975].tolist()
        #cutoff2=tmp[0]
        nop_mean=np.mean(np.array(list(noprogress['Y_pred'])))
        nop_std=np.std(np.array(list(noprogress['Y_pred'])))
        cutoff2=nop_mean+2*nop_std
        
        if t_interval==1:
            groupN, intval=self.disease_intervals(1)
        else: 
            groupN, intval=self.disease_intervals(2)
        output_youden=[]#based on Youden Index
        output_colpopu=[]#based on control population
        for i in range(0,len(intval)):
            subset=progress.loc[(progress.Time_to_TB<=intval[i][0]) & (progress.Time_to_TB>intval[i][1]),:]
            subset['group']=1
            subset=pd.concat([subset,noprogress])
            
            #Assume 2% pre-test probability(prevalence)
            Se,Se_l,Se_u=self.sensitivity(subset.loc[:,['group','Y_pred']],Youden_cutoffs[groupN[i]])
            Sp,Sp_l,Sp_u=self.specificity(subset.loc[:,['group','Y_pred']],Youden_cutoffs[groupN[i]])
            PPV,PPV_l,PPV_u=self.PPV(subset.loc[:,['group','Y_pred']],0.02,Youden_cutoffs[groupN[i]])
            NPV,NPV_l,NPV_u=self.NPV(subset.loc[:,['group','Y_pred']],0.02,Youden_cutoffs[groupN[i]])
            output_youden.append([r'%0.3f (%0.3f - %0.3f)' % (Se, Se_l, Se_u), r'%0.3f (%0.3f - %0.3f)' % (Sp, Sp_l, Sp_u), 
                           r'%0.3f (%0.3f - %0.3f)' % (PPV,PPV_l,PPV_u), r'%0.3f (%0.3f - %0.3f)' % (NPV, NPV_l, NPV_u)])
            
            Se,Se_l,Se_u=self.sensitivity(subset.loc[:,['group','Y_pred']],cutoff2)
            Sp,Sp_l,Sp_u=self.specificity(subset.loc[:,['group','Y_pred']],cutoff2)
            PPV,PPV_l,PPV_u=self.PPV(subset.loc[:,['group','Y_pred']],0.02,cutoff2)
            NPV,NPV_l,NPV_u=self.NPV(subset.loc[:,['group','Y_pred']],0.02,cutoff2)
            output_colpopu.append([r'%0.3f (%0.3f - %0.3f)' % (Se, Se_l, Se_u), r'%0.3f (%0.3f - %0.3f)' % (Sp, Sp_l, Sp_u), 
                           r'%0.3f (%0.3f - %0.3f)' % (PPV,PPV_l,PPV_u), r'%0.3f (%0.3f - %0.3f)' % (NPV, NPV_l, NPV_u)])
        
        output_youdenT=pd.DataFrame(output_youden,index=[Name+'_'+e for e in groupN],columns=['Sensitivity', 'Specificity', 'PPV', 'NPV'])
        output_colpopuT=pd.DataFrame(output_colpopu,index=[Name+'_'+e for e in groupN],columns=['Sensitivity', 'Specificity', 'PPV', 'NPV'])
        return output_youdenT, output_colpopuT
            
    ######General functions2 - distribution fiting
    #########################################################################################################################################################################
    ### Identify the distribution model that fits data
    def fit_score_dist(self, test, title, bins=25, plot=0, rcdf=0, predefine_range=[], dist=1):
        if plot==1:
            fig, axs = plt.subplots(1, 2,figsize=(15,8));
            ax=axs[0]
            ax.hist(test, bins = bins, density=True, color=list(mpl.rcParams['axes.prop_cycle'])[1]['color'],alpha=0.5)
            # Save plot limits
            dataYLim = ax.get_ylim()
        else:
            ax=None
        # Find best fit distribution
        best_distibutions = self.best_fit_distribution(test, bins=bins, ax=ax, dist=dist)
        best_dist = best_distibutions[0]
        if plot==1:
            ax.set_ylim(dataYLim)
            ax.set_title(title+u' score fitting')
            ax.set_xlabel(u'Scores')
            ax.set_ylabel('Probability density')

        # Make PDF with best params 
        pdf,cdf = self.make_pdf(best_dist[0], best_dist[1],rcdf=rcdf,predefine_range=predefine_range)
        #param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
        #param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
        #dist_str = '{}({})'.format(best_dist[0].name, param_str)
        if plot==1:
            # Display
            ax=axs[1]
            pdf.plot(lw=2, label='PDF', legend=True, ax=ax)
            test.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
            #ax.set_xlim([0,1])
            ax.set_title(u'Score distribution with best fit:'+best_dist[0].name)
            ax.set_xlabel(u'Scores')
            ax.set_ylabel('Probability density') 
            plt.show()
        return pdf, cdf, best_dist
    
    def best_fit_distribution(self, data, bins=100, ax=None,dist=1):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        # Best holders
        best_distributions = []
        # Estimate distribution parameters from data
        if dist==1:
            _distn_names=['alpha','beta','chi','chi2','expon','exponnorm','genlogistic','genexpon','genextreme','gengamma',
                          'gamma','logistic','loggamma','lognorm','norm','gennorm','norminvgauss','powerlaw','powerlognorm','powernorm','skewnorm']
        else:
            _distn_names=['alpha','expon','exponnorm','genexpon','genextreme', 'gamma','logistic','exponnorm','lognorm',
                          'norm','norminvgauss','powerlognorm','powernorm','skewnorm']
        for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):
            distribution = getattr(st, distribution)
            # Try to fit the distribution
            try:
                # fit dist to data
                params = distribution.fit(data)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass
                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
            except Exception:
                pass
        return sorted(best_distributions, key=lambda x:x[2])

    def make_pdf(self, dist, params, size=10000, rcdf=0,predefine_range=[]):
        """Generate distributions's Probability Distribution Function """
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        # Get start and end points of distribution
        if len(predefine_range)==2: #for the predicted score between 0 and 1
            start=predefine_range[0];end=predefine_range[1]
        else:
            start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
            end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)            
        y = dist.cdf(x, loc=loc, scale=scale, *arg)
        if rcdf==1: 
            #Original CDF: the probability that X will take a value less than or equal to x.
            #Reverse CDF: the probability that X will take a value larger than or equal to x.
            y=1-y
        cdf = pd.Series(y, x)
        return pdf, cdf
    
    def cdf_conf_set(self, F, nobs, alpha=0.05):#alpha=0.05 :: 95% confidence interval
        #The Dvoretzky–Kiefer–Wolfowitz inequality is one method for generating CDF-based confidence.
        #https://en.wikipedia.org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality
        epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * nobs))
        lower = np.clip(F - epsilon, 0, 1)
        upper = np.clip(F + epsilon, 0, 1)
        return lower, upper

    ######General functions3 - model score calculation 
    #########################################################################################################################################################################
    def modelscore(self, info, gset, pubmodel=0, model='',inrange=0):
        gset=gset.loc[info.index,:]
        #Predict outcome given the data
        if pubmodel==0: #in-house model
            validate_Y_pred = model.predict(gset)
            
            #Ensure score between 0-1
            if inrange==1:
                validate_Y_pred[validate_Y_pred>1]=1
                validate_Y_pred[validate_Y_pred<0]=0
        else:
            if model=='Sweeney 3':
                #score = (GBP5 + DUSP3)/2 − KLF2
                validate_Y_pred=(gset['GBP5']+gset['DUSP3'])/2 - gset['KLF2']
                
            elif model=='RISK 6':
                #score = geometricMean(GBP2,FCGR1B, SERPING1) - geometricMean(TUBGCP6,TRMT2A,SDR39U1)
                #Adjust gene expression by rows (samples) if genes in the row (sample) whose values are negative
                for e in gset.index[gset[gset<0].sum(axis=1)<0]:
                    gset.loc[e,:]= gset.loc[e,:]- gset.loc[e,:].min()+1
                    
                validate_Y_pred=scipy.stats.mstats.gmean(np.array(gset.loc[:,['GBP2','FCGR1B','SERPING1']]),axis=1) - scipy.stats.mstats.gmean(np.array(gset.loc[:,['TUBGCP6','TRMT2A','SDR39U1']]),axis=1)
                
            elif model=='BATF2':
                validate_Y_pred=gset['BATF2']
                
            elif model=='Suliman 2':
                #score=ANKRD22-OSBPL10
                validate_Y_pred=gset['ANKRD22']-gset['OSBPL10']
                
            elif model=='Suliman 4':
                #score=(GAS6+SEPT4)-(CD1C+BLK) #(SEPT4 is one of aliases for SEPTIN4)
                validate_Y_pred=(gset['GAS6']+gset['SEPTIN4'])-(gset['CD1C']+gset['BLK'])

        infoY=info.copy()
        infoY['Y_pred']=[float(e) for e in validate_Y_pred]
        return infoY
    
    
    ######General functions4 - data processing and visualization
    #########################################################################################################################################################################
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
      
    def load_valdata(self, valdata_dir, datasets, name, nameupdate=1):
        if datasets.loc[name,'Platform'] == 'microarray':
            dtype='_array'
        else:
            dtype='_seq'
        
        if name == 'GSE89403':
            gset = pd.read_csv(valdata_dir+'/'+name+dtype+'_Exp_EachGene.csv',sep=',',index_col=0)
        else:
            gset = pd.read_csv(valdata_dir+'/'+datasets.loc[name,'Dataset_ID']+dtype+'_Exp_EachGene.csv',sep=',',index_col=0)
        
        info= pd.read_csv(valdata_dir+'/'+datasets.loc[name,'Dataset_ID']+dtype+'_Exp_Info.csv',sep=',',index_col=0)
        gset=gset.loc[:,info.index]
        #update gene name
        if nameupdate==1:
            gset.index=[ut.genealias(e) for e in gset.index]

        return info, gset
    
    def load_ml_model(self, name, path):
        if ' vs ' in name:
            name=re.sub(' vs ','_',name)
        #open model
        with open(path+'/'+name+'.pickle' , 'rb') as handle:     
            select_fea,final_model,select_ml = pickle.load(handle)
            
        return select_fea,final_model,select_ml
    
    def valdata_process(self, X, select_fea, rescale=1, pair=1,recover=1):
        #Function: process the data to feed ML
        
        if rescale==1:
            #Convert expression to Z-score 
            xre = StandardScaler().fit_transform(X)
            X_new=pd.DataFrame(xre,index=X.index,columns=X.columns)
        else:
            X_new=X
        
        #Keep the selected genes only
        genes=[]
        if pair==1:#For paired-gene feature
            for each in select_fea:
                for e in each.split('_'):
                    genes.append(e)
            genes=list(set(genes))
        else:
            genes=select_fea
        #print(genes)
        #Confirm gene names used in X
        gene_confirm=[]
        for e in genes:
            if e not in X_new.index:
                if ut.genealias(e) not in X_new.index:
                    #print(e+'||'+ut.genealias(e)+' not available in this cohort')
                    return pd.DataFrame([])
                else:
                    gene_confirm.append(ut.genealias(e))
            else:
                gene_confirm.append(e)
        #print(gene_confirm)
        #Update gene name in the X_new   
        genes_newname=[ut.genealias(e) for e in gene_confirm]
        X_new=X_new.loc[gene_confirm,:]
        X_new.index=genes_newname
        #Recover the missing value if any
        if recover==1:
            X_new=ut.MissingValueRecovery(X_new)
        else:#remove samples whose gene expression is nan
            X_new=X_new.loc[:,X_new.isnull().sum()==0]
            
        #Matrix transpose 
        X_new=X_new.T
        
        if pair==1:
            #Generate paired data
            X_new_p=pd.DataFrame([],index=X_new.index,columns=select_fea)
            for pair in select_fea:
                if '_' in pair:
                    g1,g2=pair.split('_')
                    g1 = g1 if g1 in X_new.columns else ut.genealias(g1)
                    g2 = g2 if g2 in X_new.columns else ut.genealias(g2)
                    X_new_p[pair]=X_new.loc[:,g1]-X_new.loc[:,g2]
                else:
                    X_new_p[pair]=X_new.loc[:,pair]
            return X_new_p
        else:
            return X_new
    
    def disease_intervals(self, types):
        if types==1: #exclusive time intervals
            newgroupN=['0-3m to disease','3-6m to disease','6-12m to disease',
                   '12-18m to disease','18-24m to disease','24-30m to disease']
            intval=[[0,-90],[-90,-180],[-180,-360],[-360,-540],[-540,-720],[-720,-900]]
            
        elif types==6: #inclusive time intervals
            newgroupN=['0-3m to disease','3-6m to disease','6-12m to disease',
                   '12-18m to disease','18-24m to disease','24-30m to disease','>30m to disease']
            intval=[[0,-90],[-90,-180],[-180,-360],[-360,-540],[-540,-720],[-720,-900],[-900,-2000]]
        
        elif types==2: #inclusive time intervals
            newgroupN=['< 3m to disease','< 6m to disease','< 12m to disease',
                   '< 18m to disease','< 24m to disease','< 30m to disease']
            intval=[[0,-90],[0,-180],[0,-360],[0,-540],[0,-720],[0,-900]]
        
        elif types==3: #exclusive time intervals
            #newgroupN=['< 3m to disease','3-12m to disease','12-24m to disease']
            #intval=[[0,-90],[-90,-360],[-360,-720]]
            newgroupN=['< 3m to disease','<12m to disease','<24m to disease']
            intval=[[0,-90],[0,-360],[0,-720]]
        
        elif types==4: #treatment time intervals
            newgroupN=['Baseline','Week 1','Week 2','Week 3-4','Month 2-3','Month 4-6','Month 7-12','> 1 year']
            intval=[[-7,0],[1,7],[8,14],[15,30],[31,90],[91,180],[181,365],[366,2000]]
        elif types==5: #treatment time intervals
            newgroupN=['Baseline','Week 1','Week 2','Week 3','Week 4','Month 2','Month 3','Month 4','Month 5','Month 6', 'Month 7-12','> 1 year']
            intval=[[-7,0],[1,7],[8,14],[15,21],[22,30],[31,60],[61,90],[91,120],[91,150],[151,181],[182,365],[366,2000]]
            
        return newgroupN, intval
        
    def valmodel_assess_rocauc(self, info, title, save):
        #Function: show model perfomance in rocauc curves
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        noprogress=info.loc[info.group==0,:]
        progress=info.loc[info.group!=0,:]
        progress['Time_to_TB']=[int(e) for e in progress['Time_to_TB']]
        output=[]
        
        #group timepoints every 6 months
        ax=axs[0]
        cutoffs1=dict()
        newgroupN, intval=self.disease_intervals(1)
        c=0
        for i in range(0,len(intval)):
            subset=progress.loc[(progress.Time_to_TB<=intval[i][0]) & (progress.Time_to_TB>intval[i][1]),:]
            if subset.shape[0]>0:
                subset['group']=1
                subset=pd.concat([subset,noprogress])
                fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['group'].tolist(), subset['Y_pred'].tolist())
                lower, upper=self.roc_auc_ci(roc, subset['group'].tolist())
                cutoffs1[newgroupN[c]]=cutoff
                output.append([title, newgroupN[c], '{0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper), cutoff, tpr_cf, fpr_cf])
                ax.plot(fpr, tpr, label=newgroupN[c]+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
                ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
            c=c+1
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC 1')
        ax.legend(loc="lower right")
        
        #group timepoints from sample collection to disease (0-3, 0-6, 0-12, 0-18, 0-24, 0-30 months)
        ax=axs[1]
        newgroupN, intval=self.disease_intervals(2)
        cutoffs2=dict()
        for i in range(0,len(intval)):
            subset=progress.loc[(progress.Time_to_TB<=intval[i][0]) & (progress.Time_to_TB>intval[i][1]),:]
            subset['group']=1
            subset=pd.concat([subset,noprogress])
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['group'].tolist(), subset['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, subset['group'].tolist())
            cutoffs2[newgroupN[i]]=cutoff
            output.append([title, newgroupN[i], '{0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper), cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label=newgroupN[i]+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC 2')
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_ROCAUCs.pdf')
        plt.show()
        return cutoffs1,cutoffs2,output
    
    def timecourse_assess(self, plotT, progress_plotT,title,save):
        #Function: generate the boxplot over different defined status of TB progression and cthe orrelation scatterplot
        #Restrict within 900 day to disease
        progress_plotT=progress_plotT.loc[progress_plotT['Time_to_TB']>=-900,:]
        
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        
        ax=axs[0]
        newgroupN, intval=self.disease_intervals(1)
        c=1
        for i in range(0,len(intval))[::-1]:
            inds=progress_plotT.index[(progress_plotT.Time_to_TB<=intval[i][0]) & (progress_plotT.Time_to_TB>intval[i][1])]
            if len(inds)>0:
                plotT.loc[inds,'group']=c
                plotT.loc[inds,'groupName']=newgroupN[i]
                c=c+1
        groups=plotT.loc[~plotT.group.duplicated(),['group','groupName']]
        
        sns.set_theme(style="ticks")
        ax=sns.violinplot(x="group", y="Y_pred", data=plotT, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=plotT, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(groups.sort_values(by=['group'])['groupName'].values.tolist(),rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title(title)
        
        ax=axs[1]
        pr,rpval = scipy.stats.spearmanr(progress_plotT['Time_to_TB'],progress_plotT['Y_pred'])
        sns.regplot(x='Time_to_TB', y='Y_pred', data=progress_plotT, color='g',ci=95,truncate=False,ax=ax)
        sns.scatterplot(x='Time_to_TB', y='Y_pred', data=progress_plotT, legend=False, s=100,ax=ax)
        ax.annotate("Spearman $\itr$ = {:.2f}".format(pr) + "\n$\itp$-value = {:.4f}".format(rpval),xy=(.05, .78), xycoords=ax.transAxes, fontsize=15)
        ax.set_xlabel('Time to TB (days)', fontsize=10, weight='bold')
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold')
        ax.set_title(title)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_predScores_timecourse.pdf')
        plt.show()
        
        return
    
    def longitudinal_dynamics(self, progress_plotT, nonprogress_plotT, title, cutoff):
        #Function: generate the boxplot over different defined status of TB progression and cthe orrelation scatterplot
        #The cutoff (to decide prediction classification) is determined by mean of the cutoffs from all rocauc curves
        
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        all=pd.concat([progress_plotT['Y_pred'],nonprogress_plotT['Y_pred']])
        ylim=[all.min()-0.1,all.max()+0.1]
        
        ax=axs[0]
        sns.lineplot(x='sampleTime', y='Y_pred', data=nonprogress_plotT, 
                     hue="ID", style="ID",markers=True, dashes=False, legend=False, ax=ax)
        ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_xlabel('Sampling timepoints (days)', fontsize=10, weight='bold')
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold')
        ax.set_ylim(ylim)
        ax.set_title('No progression dynamics')
        
        ax=axs[1]
        sns.lineplot(x='Time_to_TB', y='Y_pred', data=progress_plotT, 
                     hue="ID", style="ID",markers=True, dashes=False, legend=False, ax=ax)
        ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_xlabel('Time to TB (days)', fontsize=10, weight='bold')
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold')
        ax.set_ylim(ylim)
        ax.set_title('Progression dynamics')
        plt.tight_layout()
        plt.show()
        
        return
    
    def valmodel_treatment_rocauc(self, info, timecolumn, title, save):
        #Function: show model perfomance in rocauc curves
        control=info.loc[info.group==0,:]
        treat=info.loc[info.group!=0,:]
        treat[timecolumn] = pd.to_numeric(treat[timecolumn])
        output=[]
        
        sns.set_theme(style="ticks")
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        ax=axs[0]
        newgroupN, intval=self.disease_intervals(4)
        for i in range(0,len(intval)):
            subset=treat.loc[(treat[timecolumn]>=intval[i][0]) & (treat[timecolumn]<=intval[i][1]),:]
            if subset.shape[0]>0:
                subset['group']=1
                subset=pd.concat([subset,control])
                fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['group'].tolist(), subset['Y_pred'].tolist())
                lower, upper=self.roc_auc_ci(roc, subset['group'].tolist())
                output.append([title, newgroupN[i], roc, tpr_cf, fpr_cf])
                ax.plot(fpr, tpr, label=newgroupN[i]+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
                ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC')
        ax.legend(loc="lower right")
        
        ax=axs[1]
        c=1
        for i in range(0,len(intval)):
            inds=info.index[(info[timecolumn]>=intval[i][0]) & (info[timecolumn]<=intval[i][1])]
            if len(inds)>0:
                info.loc[inds,'group']=c
                info.loc[inds,'groupName']=newgroupN[i]
                c=c+1
        info.loc[info['group']==0,'group']=c #Assign the control group to be plotted in the end of baxplot
        groups=info.loc[~info.group.duplicated(),['group','groupName']]
        ax=ut.violinplot_compare(info, 'groupName', 'Y_pred', groups.sort_values(by=['group'])['groupName'].values.tolist(), title, 
                                  xlabel='', ylabel='Predicted scores', save='', fig=0, stats=0, ax=ax)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_rocauc_groupDisp.pdf')
        
        return output, info
    ######General functions6 - deal with multiple cohorts
    #########################################################################################################################################################################
    def process_progression_cohorts(self, valdata_dir, datasets, select_fea, rescale, pair):
        
        info_col=['Cohort','ID','sampleTime','Time_to_TB','group','groupName']
        progress_info=pd.DataFrame([])
        progress_gset=pd.DataFrame([])
        
        #Progress_ACS 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_ACS')
        info=info.loc[info['group']!=5,:]#Remove post diagonsis
        info=info.rename(columns={'PP_time_to_diagnosis':'Time_to_TB'})
        info['Cohort']='ACS'
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])

        #Progress_GC6
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_GC6')
        info=info.rename(columns={'Sampletime_to_TB':'Time_to_TB','subjectid':'ID','time.from.exposure.months':'sampleTime'})
        info['sampleTime']=[int(e)*30 for e in info['sampleTime']]
        info['Cohort']='GC6'
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])

        #Progress_Brizal
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_Brizal')
        info=info.loc[info['group']!=6,:]#Remove post diagonsis
        info=info.rename(columns={'days_to_tb':'Time_to_TB'})
        info['Cohort']='Brizal';info['sampleTime']=np.nan;info['ID']=np.nan
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])

        #Progress_Leicester
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_Leicester')
        HCs=info.index[info['class']=='Control']
        info=info.loc[info['group']!=5,:]#Remove post diagonsis
        info=info.rename(columns={'time_to_TB_diagnosed':'Time_to_TB','patient_id':'ID'})
        info['Cohort']='Leicester'
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])
    
        #Progress_London
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_London')
        info=info.rename(columns={'days_to_tb_diagnosis':'Time_to_TB'})
        info['Cohort']='Landon';info['sampleTime']=np.nan;info['ID']=np.nan
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])
        
        #Progress_Leicester2
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_Leicester2')
        info=info.rename(columns={'days_from_att':'Time_to_TB','patient id':'ID','group':'category'})
        info['group']=1;info['groupName']='';info['sampleTime']=np.nan;
        info.loc[info['category']=='Control','group']=0
        info.loc[info['category']=='Control','groupName']='Healthy control'
        info=info.loc[(info['category']=='Control')|(info['Time_to_TB']<1),:]
        info['Cohort']='Leicester2'
        HCs2=info.index[info['category']=='Control']
        info=info.loc[:,info_col]
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair)
        progress_info=pd.concat([progress_info,info])
        progress_gset=pd.concat([progress_gset,gset_m])
        
        #Add new column
        progress_info['HealthyControl']=0
        progress_info.loc[HCs.tolist()+HCs2.tolist(),'HealthyControl']=1
        progress_info.loc[progress_info['HealthyControl']==1,'groupName']='Healthy control'
        
        return progress_info, progress_gset
    
    def progress_assess(self, modelN, final_model, info, gset_m, title, save, pubmodel=0, predefine_range=[]):
        info=info.loc[info['HealthyControl']==0,:]
        gset_m=gset_m.loc[info.index,:]
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title, save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        rocauc_outputT=pd.DataFrame([vals],index=[modelN],columns=columns)
        
        plotT.loc[plotT['Time_to_TB']=='---','Time_to_TB']=np.nan
        progress_plotT=plotT.loc[~pd.isna(plotT['Time_to_TB']),:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        nonprogress_plotT=plotT.loc[pd.isna(plotT['Time_to_TB']),:]
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        #Calculate sensitivity, specificity, PPV and NPV based on a pre-defined thresshold 
        prog_output_youdenT_exclu,prog_output_colpopuT_exclu=self.prognosis_metrics(modelN,plotT.loc[plotT['HealthyControl']==0,['Time_to_TB','group','Y_pred']],cutoffs1,predefine_range=predefine_range,t_interval=1)
        prog_output_youdenT_olap,prog_output_colpopuT_olap=self.prognosis_metrics(modelN,plotT.loc[plotT['HealthyControl']==0,['Time_to_TB','group','Y_pred']],cutoffs2,predefine_range=predefine_range,t_interval=2)
        
        return plotT,rocauc_outputT,prog_output_youdenT_olap,prog_output_colpopuT_olap,prog_output_youdenT_exclu,prog_output_colpopuT_exclu
    
    def TB_diagonsis(self,preds,title,save):
        preds=preds.loc[(preds['Status']=='Active TB')|(preds['Status']=='Healthy control')|(preds['Status']=='LTBI & No progression')|(preds['Status']=='Other disease')|(preds['Status']=='Viral infection'),:]
        preds['class']=0
        preds.loc[preds['Status']=='Active TB','class']=1
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        ax=axs[0]
        #HC vs ATB
        subset=preds.loc[(preds.Status=='Healthy control') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='HC vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #LTBI vs ATB
        subset=preds.loc[(preds.Status=='LTBI & No progression') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='LTBI vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #other disease vs ATB
        subset=preds.loc[(preds.Status=='Other disease') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='OLD vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #viral infection vs ATB
        subset=preds.loc[(preds.Status=='Viral infection') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='VI vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #LTBI+control vs ATB
        subset=preds.loc[(preds.Status=='LTBI & No progression') | (preds.Status=='Healthy control') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='HC+LTBI vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #non ATB vs ATB
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(preds['class'].tolist(), preds['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, preds['class'].tolist())
        ax.plot(fpr, tpr, label='Non ATB vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC')
        ax.legend(loc="lower right")
        
        ax=axs[1]
        sns.set_theme(style="ticks")
        ax=ut.violinplot_compare(preds, 'Status', 'Y_pred', ['Healthy control', 'LTBI & No progression','Other disease','Viral infection','Active TB'], title+' | score distribution', ylabel='TB scores',fig=0,ax=ax)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_dist.pdf')
        plt.show()
        return
    
    def TBprogress_data_integrate(self, preds_t, preds_v):
        #Function : merge two predicted matries 
        #Process preds_t
        preds_t=preds_t.loc[:,['GSEID','Status','group','Y_pred']]
        preds_t.loc[preds_t['Status']=='HC','group']=0
        preds_t.loc[preds_t['Status']=='LTBI','group']=1
        preds_t.loc[preds_t['Status']=='LTBI','Status']='LTBI & No progression'
        preds_t=preds_t.rename(columns={"GSEID": "Cohort"})
        #Process preds_v
        preds_v.loc[preds_v['groupName']=='No progression','groupName']='LTBI & No progression'
        preds_v.loc[preds_v['HealthyControl']==1,'groupName']='HC'
        preds_v=preds_v.loc[:,['Cohort','Time_to_TB','group','groupName','Y_pred']]
        preds_v=preds_v.rename(columns={"groupName": "Status"})
        preds_v.loc[preds_v['Status']=='HC','group']=0
        preds_v.loc[preds_v['Status']=='LTBI & No progression','group']=1
        #preds_v.loc[preds_v['Status']=='No progression or Healthy Control','Status']='LTBI & No progression'
        #Move HC and LTBI samples from preds_v to preds_t first
        preds_t=pd.concat([preds_t,preds_v.loc[(preds_v['Status']=='HC')|(preds_v['Status']=='LTBI & No progression'),['Cohort','Status','group','Y_pred']]])
        preds_t['Time_to_TB']=np.nan
        #Categorize the samplepoints into groups (exclusive time intervals)
        preds_v=preds_v.loc[(preds_v['Status']!='HC') & (preds_v['Status']!='LTBI & No progression'),:]
        preds_v['Time_to_TB']=[int(e) for e in preds_v['Time_to_TB']]
        gN1, int1=self.disease_intervals(1)
        c=2
        for i in range(0,len(int1))[::-1]:
            inds=preds_v.index[(preds_v.Time_to_TB<=int1[i][0]) & (preds_v.Time_to_TB>int1[i][1])]
            if len(inds)>0:
                preds_v.loc[inds,'group']=c
                preds_v.loc[inds,'Status']=gN1[i]
            c=c+1
        preds=pd.concat([preds_t.loc[:,preds_v.columns],preds_v])
        preds.loc[preds['Status']=='ATB','group']=c
        preds.loc[preds['Status']=='OD','group']=c+1
        preds.loc[preds['Status']=='HC','Status']='Healthy control'
        preds.loc[preds['Status']=='ATB','Status']='Active TB'
        preds.loc[preds['Status']=='OD','Status']='Other disease'
        #Use a alternative time interval definition for modeling 
        preds_alt=preds.loc[(preds['Status']=='Healthy control')|(preds['Status']=='LTBI & No progression') |(preds['Status']=='Active TB'),:]
        gN3, int3=self.disease_intervals(3)
        c=2
        for i in range(0,len(int3))[::-1]:
            subset=preds.loc[(preds.Time_to_TB<=int3[i][0]) & (preds.Time_to_TB>int3[i][1]),:]
            subset['group']=c
            subset['Status']=gN3[i]
            preds_alt=pd.concat([preds_alt,subset])
            c=c+1
        preds_alt.loc[preds_alt['Status']=='Active TB','group']=c
        return preds, preds_alt
    
    def TBprog_model(self,preds,preds_alt,title,save, predefine_range=[],dist=1,ylim=0):
        #Function: Fit the predicted scores from different time intervals into specific distributions and generate a TB progression risk probablistic model.
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        ax=axs[0]
        preds=preds.loc[(preds['Status']!='Viral infection')&(preds['Status']!='Other disease'),:]
        groups=preds.loc[~preds.group.duplicated(),['group','Status']]
        sns.set_theme(style="ticks")
        ax=ut.violinplot_compare(preds, 'Status', 'Y_pred', groups.sort_values(by=['group'])['Status'].values.tolist(), title+' | score distribution over time intervals', ylabel='predicted scores',fig=0,ax=ax)
        
        #ax=sns.violinplot(x="group", y="Y_pred", data=preds, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        #ax=sns.stripplot(x="group", y="Y_pred", data=preds, size=4, color=".2", linewidth=0, ax=ax)
        #ax.set_xticklabels(groups.sort_values(by=['group'])['Status'].values.tolist(),rotation=45)
        #ax.set_xlabel('')
        #ax.set_ylabel('predicted scores')
        #ax.set_title(title+' | Score distribution over time intervals')
        if ylim==1:
            ax.set_ylim([-0.1,1.1])
        
        pdfy=[];cdfy=[];cdfy_upper=[];cdfy_lower=[]
        for i in list(set(preds_alt.group)):
            rcdf = 1 if i < 2 else 0 #reverse CDF
            test=preds_alt.loc[preds_alt.group==i,'Y_pred']
            name=preds_alt.loc[preds_alt.group==i,'Status'][0]
            pdf, cdf, best_dist=self.fit_score_dist(test,name,plot=0,rcdf=rcdf,predefine_range=predefine_range,dist=dist)
            pdfy.append(pdf.tolist())
            cdfy.append(cdf.tolist())
            #95% confidence 
            lower, upper=self.cdf_conf_set(cdf, len(test), alpha=0.05)
            cdfy_lower.append(lower)
            cdfy_upper.append(upper)
        
        pdfT=pd.DataFrame(pdfy,columns=pdf.index)
        pdfT=pdfT.T
        pdfT.columns=['Healthy control','No progression','< 24m to disease','< 12m to disease','< 3m to disease','Active TB']
        cdfT=pd.DataFrame(cdfy,columns=cdf.index)
        cdfT=cdfT.T
        cdfT.columns=pdfT.columns
        cdfT_lower=pd.DataFrame(cdfy_lower)
        cdfT_lower=cdfT_lower.T
        cdfT_lower.columns=pdfT.columns
        cdfT_upper=pd.DataFrame(cdfy_upper)
        cdfT_upper=cdfT_upper.T
        cdfT_upper.columns=pdfT.columns
        
        ax=axs[1]
        for e in cdfT.columns:
            ax.plot(cdfT.index,cdfT.loc[:,e],label=e)
            ax.fill_between(cdfT.index, cdfT_lower.loc[:,e], cdfT_upper.loc[:,e], color='grey', alpha=.1)
        if predefine_range==1:
            ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('Predicted score')
        ax.set_ylabel('Cumulative distribution function (probability)')
        ax.set_title(title+' | disease risk estimation')
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_probmodel.pdf')
        plt.show()
        
        return pdfT,cdfT
    
    
    def process_treatment_cohorts(self, valdata_dir, datasets, select_fea, rescale, pair):
        
        info_col=['Cohort','ID','time','group','outcome','timetonegativity']
        treat_info=pd.DataFrame([],columns=info_col)
        treat_gset=pd.DataFrame([])
        
        #South Africa Catalysis (Cortis) cohort 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Treat_Cortis')
        info=info.rename(columns={'disease state':'group','subject':'ID','treatmentresult':'outcome'})
        info.loc[(info['outcome']=='Possible Cure')|(info['outcome']=='Probable Cure')|(info['outcome']=='Definite Cure'),'outcome']='Cure'
        info.loc[info['outcome']=='Not Cured','outcome']='Fail'
        #Label replase subjects
        cdata=pd.read_excel(valdata_dir+'/GSE89403_clinical_data.xlsx', sheet_name='clinical_data_clean', index_col=0) 
        cdata=cdata.iloc[0:100,:]
        info.loc[[info.index[ind] for ind, f in enumerate(info.ID) if f in cdata[cdata.Outcome=='Recur'].index],'outcome']='Recur'
        info['Cohort']='Treat_Cortis'
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Treat_Cortis:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m])
        
        #South Africa 2015 cohort 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Treat_SA2015')
        info=info.rename(columns={'patient':'ID'})
        info['Cohort']='Treat_SA2015';info['timetonegativity']=np.nan;info['group']='TB Subjects'
        info.loc[info['outcome']=='Cured','outcome']='Cure'
        info.loc[info['outcome']=='Relapse','outcome']='Recur'
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Treat_SA2015:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m])
       
        #Leicester (new) cohort -- treatment 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Progress_Leicester2')
        #Keep only treatment timepoint
        info=info.loc[(info['group']=='Control') | (info['days_from_att']>=-7),:]
        info=info.rename(columns={'days_from_att':'time','patient id':'ID'})
        info['Cohort']='Progress_Leicester2';info['timetonegativity']=np.nan;info['outcome']='Cure'
        info.loc[info['group']!='Control','group']='TB Subjects'
        info.loc[info['group']=='Control','group']='Healthy Controls'
        info.loc[(info['group']!='Healthy Controls')&(~info['subgroup_att'].isnull()),'group'] = info.loc[(info['group']!='Healthy Controls')&(~info['subgroup_att'].isnull()),'group']+' '+info.loc[(info['group']!='Healthy Controls')&(~info['subgroup_att'].isnull()),'subgroup_att']
        info.loc[info['group']=='Healthy Controls','outcome']=np.nan
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        gset_m=gset_m.loc[info.index,:]
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Progress_Leicester2:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m.loc[info.index,:]])
        
        #South Africa 2011 cohort 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Treat_SFUK')
        info=info.loc[:,['time','disease','ID']]
        info=info.rename(columns={'disease':'group'})
        info['Cohort']='Treat_SFUK';info['timetonegativity']=np.nan;info['outcome']=np.nan
        info.loc[info['group']=='PTB','group']='TB Subjects'
        info.loc[info['group']=='LTB','group']='LTBI'
        info.loc[info['group']=='TB Subjects','outcome']='Cure'
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Treat_SFUK:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m])
        """
        #South Africa 2013 cohort 
        info,gset = self.load_valdata(valdata_dir, datasets, 'Treat_SA')
        info=info.loc[:,['patient_id','time']]
        info=info.rename(columns={'patient_id':'ID'})
        info['Cohort']='Treat_SA';info['timetonegativity']=np.nan;info['outcome']='Cure';info['group']='TB Subjects'
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Treat_SA:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m])
        
        #Indonesia Jakarta cohort
        info,gset = self.load_valdata(valdata_dir, datasets, 'Treat_Jakarta')
        info=info.loc[:,['condition','time']]
        info['Cohort']='Treat_Jakarta';info['timetonegativity']=np.nan;info['ID']=np.nan;info['outcome']='Cure';info['group']='TB Subjects'
        info.loc[info['condition']=='Control','group']='Healthy Controls'
        info.loc[info['group']=='Healthy Controls','outcome']=np.nan;
        gset_m = self.valdata_process(gset,select_fea,rescale=rescale,pair=pair,recover=0)
        info=info.loc[ut.intersection(info.index,gset_m.index),:]
        gset_m=gset_m.loc[ut.intersection(info.index,gset_m.index),:]
        #print('Treat_Jakarta:'+str(info.shape[0])+':'+str(gset_m.shape[0]))
        if ~gset_m.empty:
            treat_info=pd.concat([treat_info,info.loc[:,info_col]])
            treat_gset=pd.concat([treat_gset,gset_m])
        """
        return treat_info, treat_gset
    
    def treatment_assess(self, modelN, final_model, info, gset_m, title, save, pubmodel=0, inclusive=1):
        #inclusive=1: include known MDR or treatment-failed subjects 
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
         
        plotT['bin']=np.nan;plotT['binName']=np.nan
        newgroupN, intval=self.disease_intervals(5)
        c=1
        for i in range(0,len(intval)):
            inds=plotT.index[(plotT['time']>=intval[i][0]) & (plotT['time']<=intval[i][1])]
            if len(inds)>0:
                plotT.loc[inds,'bin']=c
                plotT.loc[inds,'binName']=newgroupN[i]
                c=c+1
        plotT.loc[(plotT['group']=='LTBI')|(plotT['group']=='Healthy Controls'),'bin']=c #Assign the control group to be plotted in the end of baxplot
        plotT.loc[plotT['bin']==c,'binName']='Healthy Controls'
        subplotT=plotT.copy()
        subplotT=subplotT.loc[subplotT['group']!='LTBI',:]
        if inclusive==0:#exclude known MDR or treatment-failed subjects 
            subplotT=subplotT.loc[(subplotT['outcome']!='Fail')&(subplotT['group']!='TB Subjects MDR')&(subplotT['group']!='TB Subjects TB Drug Resistance')&(subplotT['group']!='TB Subjects Outbreak TB strain'),:]
        groups=subplotT.loc[~subplotT.bin.duplicated(),['bin','binName']]
        
        fig, ax = plt.subplots(1, 1,figsize=(8,8))
        sns.set_theme(style="ticks")
        ax=sns.violinplot(x="bin", y="Y_pred", data=subplotT, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="bin", y="Y_pred", data=subplotT, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(groups.sort_values(by=['bin'])['binName'].values.tolist(),rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_groupDisp.pdf')
        plt.show()
        return plotT
    
    def plot_dis_alldata(self,modelN,final_ml_dir,save,inclusive=0):
        
        preds_progress=pd.read_csv(final_ml_dir+'/Com_progress_pred_'+modelN+'.csv',index_col=0) 
        #Remove other disease
        #preds_progress=preds_progress.loc[preds_progress['Status']!='Other disease',:]q
        preds_progress=preds_progress.rename(columns={'Time_to_TB':'time','group':'bin','groupName':'binName'})
        preds_progress=preds_progress.loc[preds_progress['HealthyControl']!=1,:]
        preds_progress.loc[(preds_progress['HealthyControl']==0)&(preds_progress['binName']=='No progression or Healthy Control'),'bin']=0
        preds_progress.loc[preds_progress['bin']==0,'binName']='LTBI & No progression'
        newgroupN, intval=self.disease_intervals(6)
        c=1
        for i in range(0,len(intval))[::-1]:
            preds_progress.loc[(preds_progress['time']<=intval[i][0]) & (preds_progress['time']>=intval[i][1]),'bin']=c
            preds_progress.loc[preds_progress['bin']==c,'binName']=newgroupN[i]
            c=c+1
        cmax=max(list(set(preds_progress.bin)))
        
        preds_treat=pd.read_csv(final_ml_dir+'/Com_treat_pred_'+modelN+'.csv',index_col=0)
        preds_treat=preds_treat.loc[preds_treat['group']!='LTBI',:]
        if inclusive==0:#exclude known MDR or treatment-failed subjects 
            preds_treat=preds_treat.loc[(preds_treat['outcome']!='Fail')&(preds_treat['group']!='TB Subjects MDR')&(preds_treat['group']!='TB Subjects TB Drug Resistance')&(preds_treat['group']!='TB Subjects Outbreak TB strain'),:]
        newgroupN, intval=self.disease_intervals(5)
        for c,name in enumerate(newgroupN):
            preds_treat.loc[preds_treat['binName']==name,'bin']=cmax+c+1
        preds_treat.loc[preds_treat['binName']=='Healthy Controls','bin']=cmax+c+2
            
        #Combine two data
        All=pd.concat([preds_progress.loc[:,['Cohort','time','bin','binName','Y_pred']],preds_treat.loc[:,['Cohort','time','bin','binName','Y_pred']]])
        groups=All.loc[~All.bin.duplicated(),['bin','binName']]
        
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(1, 1,figsize=(12,8))
        ax=sns.violinplot(x="bin", y="Y_pred", data=All, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="bin", y="Y_pred", data=All, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(groups.sort_values(by=['bin'])['binName'].values.tolist(),rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title(modelN+' | Score distribution over time intervals')
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_groupDisp_alldata.pdf')
        plt.show()
        
        return
    
    def viral_data_extract(self, vinf_dir, name, i_col, i_col_new, ageUnit='year'):
        data = pd.read_csv(vinf_dir+'/'+name+'_array_Exp_EachGene.csv',sep=',',index_col=0)
        data.index=[ut.genealias(x) for x in data.index]
        info = pd.read_csv(vinf_dir+'/'+name+'_array_Exp_Info.csv',sep=',',index_col=0)
        info=info.loc[:,i_col]
        info.columns=i_col_new

        if name=='GSE111368_GPL10558':
            info['disease']='Influenza'
            info.loc[info.viral_group=='HC','disease']='HC'
        if name=='GSE17156_GPL571':
            #Change pre-challenge (baseline) to healthy
            info.loc[info['disease'].str.contains('baseline'),'disease']='healthy'
            info.disease=info.disease.replace(to_replace=r' challenge sample at T.*$', value='', regex=True)
        if name=='GSE20346_GPL6947':
            #only keep influenza samples
            info=info.loc[info['disease'].str.contains('Severe Influenza'),:]
            info.disease=info.disease.replace(to_replace=r'^Severe ', value='', regex=True)
            info['timepoint']=info['disease']
            info.disease=info.disease.replace(to_replace=r'_day.*$', value='', regex=True)
            info.timepoint=info.timepoint.replace(to_replace=r'^Influenza_', value='', regex=True)
        if name=='GSE21802_GPL6102':
            info['disease']='Influenza'
            info.loc[info.viral_group=='none','disease']='none'
            info['viral_group'] = info.viral_group.str.extract(r'\((.*) new subtype\)', expand=True)
        if 'GSE38900' in name:
            #Remove follow up samples
            info=info.loc[~info['disease'].str.contains('follow up'),:]
            info.loc[info['disease'].str.contains('healthy'),'disease']='healthy'
            info.loc[info['disease'].str.contains('RSV'),'disease']='RSV'
            info.loc[info['disease'].str.contains('HRV'),'disease']='Rhinovirus'
            info.loc[info['disease'].str.contains('Influenza A'),'disease']='Influenza A'
        if name=='GSE40012_GPL6947':
            #Only select influenza A samples 
            info=info.loc[info['disease'].str.contains('influenza A')|info['disease'].str.contains('healthy control'),:]
            info.loc[info['disease'].str.contains('influenza A'),'disease']='Influenza A'
            info.loc[info['disease'].str.contains('healthy control'),'disease']='healthy'
        if 'GSE6269' in name:
            info=info.loc[info['disease'].str.contains('Influenza A')|info['disease'].str.contains('Influenza B')|info['disease'].str.contains('None'),:]
        if name=='GSE67059_GPL6947':
            info.loc[info.disease=='HRV-','disease']='healthy'
            info.loc[info.disease=='HRV+','disease']='Rhinovirus'
        if name=='GSE68004_GPL10558':
            info=info.loc[info['disease'].str.contains('HAdV')|info['disease'].str.contains('healthy'),:]
            info.loc[info.disease=='HAdV','disease']='adenovirus'
        if name=='GSE68310_GPL10558':
            #Exclude timpoint=spring
            info=info.loc[(info.timepoint!='Spring')&(info.disease!='our tests did not detect one of the viruses sought'),:]
            #Change sample's disease status whose timepoint equal to baseline to healthy (prior to infection) 
            info.loc[info.timepoint=='Baseline','disease']='healthy'
            info.disease=info.disease.replace(to_replace=r'respiratory syncytial virus', value='RSV', regex=True)
            info.disease=info.disease.replace(to_replace=r'human | virus', value='', regex=True)
        if name=='GSE73072_GPL14604':
            info['disease']=info['viral_group']
            info['time']=[int(re.sub('hour ','',e)) for e in info.timepoint]
            #Change sample's disease status whose timepoint equal or less than hr 0 to healthy (prior to challenge) 
            info.loc[info.time<=0,'disease']='healthy'
            info.loc[(info.viral_group=='H1N1')|(info.viral_group=='H3N2'),'disease']='Influenza'
            info.loc[(info.viral_group=='HRV'),'disease']='Rhinovirus'
        if name=='GSE61821_GPL10558':
            info['disease']='Influenza'
            #only keep the acute stage or day 0 before treatment
            info=info.loc[info.viral_group!='OFI',:]
            info=info.loc[(info.timepoint=='Acute')|(info.timepoint=='day_0'),:]
        if name=='GSE61754_GPL10558':
            #Exclude vaccinee 
            info['disease']='Influenza'
            info=info.loc[(info.vaccination_status=='Control')|((info.vaccination_status=='Vaccinee')&(info.timepoint=='Pre-challenge')),:]
            info.loc[(info.timepoint=='Pre-challenge'),'disease']='healthy'

        #Entry adjust
        if ageUnit=='month':
            info.age=info.age/12
        info.loc[(info.disease=='Control')|(info.disease=='Healthy')|(info.disease=='HC')|(info.disease=='none')|(info.disease=='None'),'disease']='healthy'
        if 'gender' in info.columns:
            info.gender=info.gender.str.lower()
            info.loc[info.gender=='m','gender']='male'
            info.loc[info.gender=='f','gender']='female'
        #Only keep samples only existing in info
        data=data.loc[:,info.index]
        return info, data

    def viral_infect_dataset(self, vinf_dir, select_fea, rescale, pair):
        all_info_col=['Cohort','disease','viral_group','age','gender']
        all_info=pd.DataFrame([],columns=all_info_col)
        all_gset=pd.DataFrame([])

        #GSE101702 (Influenza:: influenza patients with varying severity of infection)
        info, data=self.viral_data_extract(vinf_dir,'GSE101702_GPL21185',['age:ch1','diagnosis:ch1','severity:ch1','Sex:ch1','tissue:ch1'],['age','disease','severity','gender','tissue'])
        info['Cohort']='GSE101702'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE103842 (RSV:: RSV infected infants)
        info, data=self.viral_data_extract(vinf_dir,'GSE103842_GPL10558',['age (in months):ch1','condition1:ch1','condition2:ch1','gender:ch1','tissue:ch1'],
                                ['age','disease','viral_group','gender','tissue'],ageUnit='month')
        info['Cohort']='GSE103842'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE111368 (Influenza:: patients with severe influenza with or without bacterial co-infection)
        info, data=self.viral_data_extract(vinf_dir,'GSE111368_GPL10558',['age:ch1','bacterial_status:ch1','day of illness:ch1','flu_type:ch1','Sex:ch1','subject id:ch1',
                                't1severity:ch1','timepoint:ch1'],['age','bacterial_status','day_illness','viral_group','gender','subjectID','severity','timepoint'])
        info['Cohort']='GSE111368'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE117827 (Rhinovirus, RSV, Enterovirus, Coxsackievirus:: children with acute viral infection)
        info, data=self.viral_data_extract(vinf_dir,'GSE117827_GPL23126',['age (months):ch1','gender:ch1','infection:ch1','symptomatic:ch1'],['age','gender','disease','symptomatic'],ageUnit='month')
        info['Cohort']='GSE117827'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE17156 (Influenza, Rhinovirus, RSV:: challenge study cohort)
        info, data=self.viral_data_extract(vinf_dir,'GSE17156_GPL571',['title','subject id:ch1','symptom group:ch1','timepoint:ch1'],['disease','subjectID','symptomatic','timepoint'])
        info['Cohort']='GSE17156'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE20346 (Influenza:: adults with CAP)
        info, data=self.viral_data_extract(vinf_dir,'GSE20346_GPL6947',['title','tissue:ch1'],['disease','tissue'])
        info['Cohort']='GSE20346'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE21802 (Influenza:: patients attending to the participants ICUs with primary viral pneumonia during the acute phase of influenza virus illness with acute respiratory distress 
        #and unequivocal alveolar opacification involving two or more lobes with negative respiratory and blood bacterial cultures at admission)
        info, data=self.viral_data_extract(vinf_dir,'GSE21802_GPL6102',['disease phase:ch1','mechanical ventilation:ch1','patient:ch1','tissue:ch1','virus strain:ch1'],
                                ['disease_phase','mechanical_ventilation','subjectID','tissue','viral_group'])
        info['Cohort']='GSE21802'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE38900 (RSV, Rhinovirus, Influenza:: Children with acute lower respiratory tract infection)
        info, data=self.viral_data_extract(vinf_dir,'GSE38900_GPL10558',['source_name_ch1','age:ch1','gender:ch1','tissue:ch1'],['disease','age','gender','tissue'],ageUnit='month')
        info['Cohort']='GSE38900'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])
        info, data=self.viral_data_extract(vinf_dir,'GSE38900_GPL6884',['source_name_ch1','age (months):ch1','gender:ch1','tissue:ch1'],['disease','age','gender','tissue'],ageUnit='month')
        info['Cohort']='GSE38900'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE40012 (Influenza:: adults with CAP)
        info, data=self.viral_data_extract(vinf_dir,'GSE40012_GPL6947',['sample type:ch1','day:ch1','gender:ch1','ID:ch1','tissue:ch1'],['disease','day','gender','subjectID','tissue'])
        info['Cohort']='GSE40012'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE6269 (Influenza:: influenza and other acute respiratory viral infections)
        info, data=self.viral_data_extract(vinf_dir,'GSE6269_GPL96',['Age:ch1','Gender:ch1','Illness:ch1','Pathogen:ch1','Treatment:ch1'],['age','gender','illness','disease','Treatment'])
        info['Cohort']='GSE6269'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])
        info, data=self.viral_data_extract(vinf_dir,'GSE6269_GPL570',['Age:ch1','Gender:ch1','Illness:ch1','Pathogen:ch1','Treatment:ch1'],['age','gender','illness','disease','Treatment'])
        info['Cohort']='GSE6269'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])
        info, data=self.viral_data_extract(vinf_dir,'GSE6269_GPL2507',['Age:ch1','Gender:ch1','Illness:ch1','Pathogen:ch1','Treatment:ch1'],['age','gender','illness','disease','Treatment'])
        info['Cohort']='GSE6269'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE67059 (Rhinovirus)
        info, data=self.viral_data_extract(vinf_dir,'GSE67059_GPL6947',['age (months):ch1','gender:ch1','name:ch1','infection:ch1','tissue:ch1'],['age','gender','subjectID','disease','tissue'],ageUnit='month')
        info['Cohort']='GSE67059'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE68004 (Adenovirus:: )
        info, data=self.viral_data_extract(vinf_dir,'GSE68004_GPL10558',['age (mos.):ch1','final condition:ch1','gender:ch1','tissue:ch1'],['age','disease','gender','tissue'],ageUnit='month')
        info['Cohort']='GSE68004'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE68310 (Influenza, Rhinovirus, Coronavirus, RSV, Enterovirus:: influenza and other acute respiratory viral infections)
        info, data=self.viral_data_extract(vinf_dir,'GSE68310_GPL10558',['gender:ch1','infection:ch1','subject id:ch1','time point:ch1'],['gender','disease','subjectID','timepoint'])         
        info['Cohort']='GSE68310'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE77087 (RSV:: outpatient and inpatient RSV samples)
        info, data=self.viral_data_extract(vinf_dir,'GSE77087_GPL10558',['age (mos):ch1','disease:ch1','Sex:ch1','tissue:ch1'],['age','disease','gender','tissue'],ageUnit='month')
        info['Cohort']='GSE77087'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE73072 (Influenza, Rhinovirus, RSV:: Duke viral challenge study)
        info, data=self.viral_data_extract(vinf_dir,'GSE73072_GPL14604',['subject:ch1','time point:ch1','tissue:ch1','virus:ch1'],['subjectID','timepoint','tissue','viral_group'])
        info['Cohort']='GSE73072'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE61821 (Influenza:: influenza infected patients with different clinical outcomes)
        info, data=self.viral_data_extract(vinf_dir,'GSE61821_GPL10558',['age:ch1','severity:ch1','timepoint:ch1','tissue:ch1','virus type:ch1'],['age','severity','timepoint','tissue','viral_group'])
        info['Cohort']='GSE61821'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        #GSE61754 (Influenza:: influenza challenge study)
        info, data=self.viral_data_extract(vinf_dir,'GSE61754_GPL10558',['seroconversion:ch1','symptom severity:ch1','timepoint:ch1','vaccination status:ch1','viral shedding:ch1'],['seroconversion','severity','timepoint','vaccination_status','viral_shedding'])
        info['Cohort']='GSE61754'
        data=self.valdata_process(data,select_fea,rescale=rescale,pair=pair)
        all_info=pd.concat([all_info,info.loc[data.index,ut.intersection(all_info_col,info.columns.tolist())]])
        all_gset=pd.concat([all_gset, data])

        return all_info, all_gset
    
    def viralinfection_classification(self,preds,title,save):
        preds['class']=0
        preds.loc[preds['Status']=='Active TB','class']=1
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        ax=axs[0]
        #viral infection vs ATB
        subset=preds.loc[(preds.Status=='Viral infection') | (preds.Status=='Active TB'),:]
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset['class'].tolist(), subset['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, subset['class'].tolist())
        ax.plot(fpr, tpr, label='Viral infection vs ATB : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC')
        ax.legend(loc="lower right")
        ax=axs[1]
        sns.set_theme(style="ticks")
        ax=ut.violinplot_compare(preds, 'Status', 'Y_pred', ['Healthy control', 'Viral infection', 'Active TB'], title, ylabel='TB score',fig=0,ax=ax)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_viral_infect.pdf')
        plt.show()
        return 
    
    
    
    ######Other functions specific to the cohorts
    #########################################################################################################################################################################
    def ACSdata_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title, save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['ACS_'+modelN],columns=columns)
        cutoff=np.mean(np.array(list(cutoffs2.values())))
        
        progress_plotT=plotT.loc[plotT['Time_to_TB']!='---',:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        nonprogress_plotT=plotT.loc[plotT['Time_to_TB']=='---',:]
        nonprogress_plotT['sampleTime']=[int(e) for e in nonprogress_plotT['sampleTime']]
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        
        #Longitudinal dynamics of predicted score in progrssors and non-progressors
        self.longitudinal_dynamics(progress_plotT, nonprogress_plotT, title, cutoff)
        
        return plotT,outputT
    
    def GCdata_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title,save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['GC6_'+modelN],columns=columns)
        cutoff=np.mean(np.array(list(cutoffs2.values())))
        
        progress_plotT=plotT.loc[~pd.isna(plotT['Time_to_TB']),:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        nonprogress_plotT=plotT.loc[pd.isna(plotT['Time_to_TB']),:]
        nonprogress_plotT['time.from.exposure.months']=[int(e)*30 for e in nonprogress_plotT['time.from.exposure.months']]
        nonprogress_plotT=nonprogress_plotT.rename(columns={'time.from.exposure.months':'sampleTime'})
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        
        #Longitudinal dynamics of predicted score in progrssors and non-progressors
        self.longitudinal_dynamics(progress_plotT, nonprogress_plotT, title, cutoff)
        
        return plotT,outputT
    
    def Brizaldata_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title,save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Brizal_'+modelN],columns=columns)
        cutoff=np.mean(np.array(list(cutoffs2.values())))
        
        progress_plotT=plotT.loc[~pd.isna(plotT['Time_to_TB']),:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        
        #Longitudinal dynamics of predicted score in progrssors and non-progressors
        #no Longitudinal timepoint from individuals are available.
        return plotT,outputT
    
    def Leidata_assess(self, modelN, final_model, info, gset_m, title,save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title, save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Leicester_'+modelN],columns=columns)
        cutoff=np.mean(np.array(list(cutoffs2.values())))
        
        progress_plotT=plotT.loc[~pd.isna(plotT['Time_to_TB']),:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        nonprogress_plotT=plotT.loc[pd.isna(plotT['Time_to_TB']),:]
        nonprogress_plotT['sampleTime']=[int(e) for e in nonprogress_plotT['sampleTime']]
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        
        #Longitudinal dynamics of predicted score in progrssors and non-progressors
        self.longitudinal_dynamics(progress_plotT, nonprogress_plotT, title, cutoff)
        
        return plotT,outputT
    
    def Londondata_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(plotT, title,save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['London_'+modelN],columns=columns)
        cutoff=np.mean(np.array(list(cutoffs2.values())))
        
        progress_plotT=plotT.loc[~pd.isna(plotT['Time_to_TB']),:]
        progress_plotT['Time_to_TB']=[int(e) for e in progress_plotT['Time_to_TB']]
        nonprogress_plotT=plotT.loc[pd.isna(plotT['Time_to_TB']),:]
        
        #Data visualization
        self.timecourse_assess(plotT,progress_plotT,title,save)
        
        return plotT,outputT
    
    def Leidata2_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
            
        subplotT=plotT.copy()
        if arg=='progress':
            #Remove treatment timepoint
            subplotT=subplotT.loc[(subplotT['category']=='Control') | (subplotT['Time_to_TB']<=1),:]
            #AUCROC evaluation
            cutoffs1,cutoffs2,output = self.valmodel_assess_rocauc(subplotT, title, save)
            cutoff=np.mean(np.array(list(cutoffs2.values())))
            progress_subplotT=subplotT.loc[~pd.isna(subplotT['Time_to_TB']),:]
            progress_subplotT['Time_to_TB']=[int(e) for e in progress_subplotT['Time_to_TB']]
            nonprogress_subplotT=subplotT.loc[pd.isna(subplotT['Time_to_TB']),:]
            #Data visualization
            self.timecourse_assess(subplotT,progress_subplotT,title,save)
            
        elif arg=='tret':
            #Keep only treatment timepoint
            subplotT=subplotT.loc[(subplotT['category']=='Control') | (subplotT['Time_to_TB']>=-7),:]
            output, subplotT= self.valmodel_treatment_rocauc(subplotT, 'Time_to_TB', title, save)
            
            fig, axs = plt.subplots(3, 2,figsize=(15,20));                              
            #Baseline: 1) TB Score vs smear_results
            ax=axs[0,0]
            subset=subplotT.copy()
            subset=subset.loc[(subset['Time_to_TB']<=0),:]
            ut.violinplot_compare(subset, 'smear_results', 'Y_pred', ['Negative', 'Positive'], 'Smear results', 
                                  xlabel='', ylabel='Predicted score', fig=0, stats=1, ax=ax)
            
            #Compare TB scores among the groups (standard ATT (anti-TB treatment 200days), extended ATT (>200 days), diffacult TB cases, TB drug resistance, outbreak TB)
            #Baseline, early time point (1st, 2nd, first month), 6 month and time-series trend
            #Check longitudinal reponses of individuals 
            subset=pd.DataFrame([])
            regsubset=pd.DataFrame([])
            for ID in set(subplotT.loc[(subplotT['subgroup_att']=='Short_ATT')|(subplotT['subgroup_att']=='Long_ATT'),'ID']):
                if subplotT.loc[subplotT['ID']==ID,:].shape[0]>2:#more than 2 timepoints 
                    idset=subplotT.loc[subplotT['ID']==ID,:]
                    idset['Time_to_TB'] = pd.to_numeric(idset['Time_to_TB'])
                    idset=idset.sort_values(by=['Time_to_TB'])
                    subset=pd.concat([subset,idset])         
                    nx,ny=ut.linesmooth(idset['Time_to_TB'].tolist(),idset['Y_pred'].tolist(),smooth=0.999)   
                    regsubset=pd.concat([regsubset,pd.DataFrame({'ID':ID,'Time_to_TB':nx,'Y_pred':ny,'subgroup_att':idset['subgroup_att'][0]})])
            regsubset.index=list(range(0,regsubset.shape[0]))
            subset=subset.sort_values(by=['Time_to_TB'])
            #Compare ATT group at month 1
            ax=axs[0,1]
            subset1=subset.copy()
            subset1=subset1.loc[(subset1['Time_to_TB']>=25)&(subset1['Time_to_TB']<=35),:]
            ut.violinplot_compare(subset1, 'subgroup_att', 'Y_pred', ['Short_ATT', 'Long_ATT'], 'Month 1', 
                                  xlabel='', ylabel='Predicted score', fig=0, stats=1, ax=ax)
            #Compare ATT group at month 2
            ax=axs[1,0]
            subset1=subset.copy()
            subset1=subset1.loc[(subset1['Time_to_TB']>=52)&(subset1['Time_to_TB']<=60),:]
            ut.violinplot_compare(subset1, 'subgroup_att', 'Y_pred', ['Short_ATT', 'Long_ATT'], 'Month 2', 
                                  xlabel='', ylabel='Predicted score', fig=0, stats=1, ax=ax)
            
            #Compare ATT group at month 6
            ax=axs[1,1]
            subset1=subset.copy()
            subset1=subset1.loc[(subset1['Time_to_TB']>=145)&(subset1['Time_to_TB']<=180),:]
            ut.violinplot_compare(subset1, 'subgroup_att', 'Y_pred', ['Short_ATT', 'Long_ATT'], 'Month 6', 
                                  xlabel='', ylabel='Predicted score', fig=0, stats=1, ax=ax)
            #Compare ATT group longitudinally 
            ax=axs[2,0]
            colors=ut.subject_linecolor()
            regsubset=regsubset.loc[(regsubset['Time_to_TB']>=0)&(regsubset['Time_to_TB']<=365),:]#Look at the first year
            sns.lineplot(x='Time_to_TB', y='Y_pred', data=regsubset, hue="subgroup_att", markers=False, style=True, dashes=False, lw=2, legend=True, ax=ax, ci=90)
            #sns.lineplot(x='Time_to_TB', y='Y_pred', data=regsubset,hue="ID", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
            subset1=subset.copy()
            subset1=subset1.loc[(subset1['Time_to_TB']>=0)&(subset1['Time_to_TB']<=365),:]
            sns.scatterplot(x='Time_to_TB', y='Y_pred', data=subset1, legend=False, s=50,hue='subgroup_att',ax=ax)
            ax.legend(loc='upper right')
            ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')    
            ax.set_title(title)
            
            ax=axs[2,1]
            #AUCROC evaluation between short vs long ATT
            output=[]
            for i in [1,2,6]:
                subset1=subset.copy()
                if i==1:
                    subset1=subset1.loc[(subset1['Time_to_TB']>=25)&(subset1['Time_to_TB']<=35),:]
                elif i==2:
                    subset1=subset1.loc[(subset1['Time_to_TB']>=52)&(subset1['Time_to_TB']<=60),:]
                elif i==6:
                    subset1=subset1.loc[(subset1['Time_to_TB']>=145)&(subset1['Time_to_TB']<=180),:]
                subset1['group']=0
                subset1.loc[subset1['subgroup_att']=='Long_ATT','group']=1
                fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(subset1['group'].tolist(), subset1['Y_pred'].tolist())
                lower, upper=self.roc_auc_ci(roc, subset1['group'].tolist())
                output.append([title, 'short vs long ATT at Month '+str(i), roc, cutoff, tpr_cf, fpr_cf])
                ax.plot(fpr, tpr, label= 'Short vs long ATT at Month '+str(i)+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
                ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
        
            plt.tight_layout()
            plt.autoscale()
            plt.savefig(save+'_ind_dynamics.pdf')
            plt.show()
            
        
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Leicester2_'+modelN],columns=columns)
        
        return plotT,outputT
    
    def VANTdata_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            plotT=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        output=[]
        fig, axs = plt.subplots(1, 2,figsize=(15,8));
        ax=axs[0]
        #OD vs ATB (PTB, EPTB, PTB_EPTB)
        plotT['group']=0
        plotT.loc[plotT['eptborptb']!='OD','group']=1
        #plotT.loc[plotT['culture result']=='Positive;Positive','group']=1
        plotT.loc[plotT['group']==0,'groupName']='Non ATB disease'
        plotT.loc[plotT['group']==1,'groupName']='Active TB'
        fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(plotT['group'].tolist(), plotT['Y_pred'].tolist())
        lower, upper=self.roc_auc_ci(roc, plotT['group'].tolist())
        ax.plot(fpr, tpr, label='ATB vs OD : ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC')
        ax.legend(loc="lower right")
        output.append([roc,'({0:0.2f} to {1:0.2f})'.format(lower,upper)])

        ax=axs[1]
        sns.set_theme(style="ticks")
        ax=sns.violinplot(x="group", y="Y_pred", data=plotT, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=plotT, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(['Non ATB disease','Active TB'],rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title(title+' | score distribution')
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_ROCAUCs.pdf')
        plt.show()
        outputT=pd.DataFrame(output,index=['VANTDET_'+modelN],columns=['ATB vs OD ROCAUC','95% confidence'])
        return plotT, outputT
   
    def Cortis_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Fetch clinical assay data
        cdata=pd.read_excel(self.valdata_dir+'/GSE89403_clinical_data.xlsx', sheet_name='Results', index_col=0) 
        cdata2=pd.read_excel(self.valdata_dir+'/GSE89403_clinical_data.xlsx', sheet_name='clinical_data_clean', index_col=0) 
        
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
            
        fig, axs = plt.subplots(5, 2,figsize=(15,38));
        sns.set_theme(style="ticks")
        #AUCROC evaluation against healthy control
        subset=info.copy()
        subset=subset.loc[(subset.treatmentresult!='Not Cured') & (subset['disease state']!='Healthy Controls'),:]
        ax=axs[0,0]
        output=[]
        for i in [0,7,28,168]:
            g1=subset.loc[subset.time==i,:]
            g1['group']=1
            g0=info.loc[info['disease state']=='Healthy Controls',:]
            g0['group']=0
            testdata=pd.concat([g1,g0])
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(testdata['group'].tolist(), testdata['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, testdata['group'].tolist())
            output.append([title, 'Day'+str(i)+'_HC', roc, cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label='Day '+str(i)+' vs Healthy control: ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC against healthy control')
        ax.legend(loc="lower right")
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Cortis_'+modelN],columns=columns)
        
        #Boxplot predicted score and time point
        subset=info.copy()
        subset['group']=np.inf
        c=0
        for e in sorted(set(subset.loc[subset['disease state']!='Healthy Controls','time'])):
            subset.loc[subset.time==e,'group']=c
            subset.loc[subset.time==e,'groupName']='Day '+str(e)
            c=c+1
        subset.loc[subset['disease state']=='Healthy Controls','group']=c
        subset.loc[subset['disease state']=='Healthy Controls','groupName']='Healthy Control'  
        
        ax=axs[0,1]
        ut.violinplot_compare(subset, 'groupName', 'Y_pred', ['Day 0','Day 7','Day 28','Day 168','Healthy Control'], '', 
                                  xlabel='', ylabel='Predicted scores', fig=0, stats=0, ax=ax)
        
        ax=axs[1,0]
        subset_nohc=info.loc[(info['disease state']!='Healthy Controls'),:].copy()
        subset_nohc['Time_to_negativity']=''
        order=[]
        for e in sorted(set(subset_nohc['time'])):
            subset_nohc.loc[subset_nohc.time==e,'groupName']='Day '+str(e)
        for e in sorted(set(subset_nohc.loc[subset_nohc.treatmentresult!='Not Cured','timetonegativity'])):
            subset_nohc.loc[subset_nohc.timetonegativity==e,'Time_to_negativity']='Day '+str(int(e))
            order.append('Day '+str(int(e)))
        subset_nohc.loc[subset_nohc['treatmentresult']=='Not Cured','Time_to_negativity']='Not Cured'
        order.append('Not Cured')
        ut.violinplot_compare(subset_nohc, 'groupName', 'Y_pred', ['Day 0','Day 7','Day 28','Day 168'], '', 
                                  hue="Time_to_negativity", hue_order=order, xlabel='', ylabel='Predicted scores', fig=0, stats=0, stripplot=0, ax=ax)
        def gdetail():
            ax.set_xticks([1,2,3,4])
            ax.set_xticklabels(['Day 0','Day 7','Day 28','Day 168'],rotation=45)
            ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
            #ax.set_ylim([0,1])
            return   
        ax=axs[1,1] 
        subset=info.loc[(info['disease state']!='Healthy Controls'),:].copy()
        subset['group']=0
        c=1
        for e in sorted(set(subset.loc[subset['disease state']!='Healthy Controls','time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        c=1
        for e in sorted(set(subset.loc[subset['disease state']=='Healthy Controls','time'])):
            subset.loc[(subset['disease state']=='Healthy Controls') & (subset.time==e),'group']=c
            c=c+1
        subset['Time_to_negativity']=''
        for e in sorted(set(subset.loc[(subset.treatmentresult!='Not Cured')&(subset['disease state']!='Healthy Controls'),'timetonegativity'])):
            subset.loc[subset.timetonegativity==e,'Time_to_negativity']='Day '+str(int(e))
        subset.loc[subset['treatmentresult']=='Not Cured','Time_to_negativity']='Not Cured'
        subset.loc[subset['disease state']=='Healthy Controls','Time_to_negativity']='Healthy Controls'
        subset=subset.sort_values(by=['timetonegativity'])
        sns.lineplot(x="group", y="Y_pred", hue="Time_to_negativity", data=subset, ax=ax, ci=40)
        ax.legend(loc='upper right')
        gdetail()
        
        ax=axs[2,0]
        subset=subset_nohc.loc[(subset_nohc['time']==0),:].copy()
        ut.violinplot_compare(subset, 'Time_to_negativity', 'Y_pred', order, 'Day 0', 
                                  xlabel='', ylabel='Predicted scores', fig=0, stats=0, ax=ax)
        ax=axs[2,1]
        subset=subset_nohc.loc[(subset_nohc['time']==7),:].copy()
        ut.violinplot_compare(subset, 'Time_to_negativity', 'Y_pred', order, 'Day 7', 
                                  xlabel='', ylabel='Predicted scores', fig=0, stats=0, ax=ax)
        ax=axs[3,0]
        subset=subset_nohc.loc[(subset_nohc['time']==28),:].copy()
        ut.violinplot_compare(subset, 'Time_to_negativity', 'Y_pred', order, 'Day 28', 
                                  xlabel='', ylabel='Predicted scores', fig=0, stats=0, ax=ax)
        ax=axs[3,1]
        subset=subset_nohc.loc[(subset_nohc['time']==168),:].copy()
        ut.violinplot_compare(subset, 'Time_to_negativity', 'Y_pred', order, 'Day 168', 
                                  xlabel='', ylabel='Predicted scores', fig=0, stats=0, ax=ax)
        
        ax=axs[4,0]
        #AUCROC evaluation between cured vs failed
        subset=info.copy()
        subset=subset.loc[subset['disease state']!='Healthy Controls',:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata2.loc[:,['Age','Outcome']], how='inner')  
        output=[]
        for i in [0,7,28,168]:
            #if i==-1:#Delta between day 0 and 7
            #    bg=subset.loc[((subset.time==0)|(subset.time==7))&(subset['Outcome']!='Recur'),:]
            #    bg['group']=0
            #    bg.loc[bg['Outcome']=='Fail','group']=1
            #    subjects=[subject for subject in bg.subject if len(bg.loc[bg.subject==subject,:])==2]
            #    subjects=list(set(subjects))
            #    g=bg.loc[subjects,:].copy()
            #    g=g.loc[g.time==0,:]
            #    for subject in g.subject:
            #        g.loc[subject,'Y_pred']=bg.loc[(bg.subject==subject)&(bg.time==0),'Y_pred'].values-bg.loc[(bg.subject==subject)&(bg.time==7),'Y_pred'].values
            #else:
            g=subset.loc[(subset.time==i)&(subset['Outcome']!='Recur'),:]
            g['group']=0
            g.loc[g['Outcome']=='Fail','group']=1
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(g['group'].tolist(), g['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, g['group'].tolist())
            output.append([title, 'Fail vs Cured at Day'+str(i), roc, cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label= 'Fail vs Cured '+'at day '+str(i)+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        ax=axs[4,1]
        #AUCROC evaluation between cured vs recurrence
        subset=info.copy()
        subset=subset.loc[subset['disease state']!='Healthy Controls',:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata2.loc[:,['Age','Outcome']], how='inner')  
        output=[]
        for i in [0,7,28,168]:
            g=subset.loc[(subset.time==i)&(subset['Outcome']!='Fail'),:]
            g['group']=1
            g.loc[g['Outcome']=='Recur','group']=0    
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(g['group'].tolist(), g['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, g['group'].tolist())
            output.append([title, 'Cured vs Recurrence at Day'+str(i), roc, cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label= 'Cured vs Recurrence at day '+str(i)+': ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_roc_dis.pdf')
        
        #Check longitudinal reponses of individuals
        subset=pd.DataFrame([])
        #regsubset=pd.DataFrame([])
        for ID in set(info['subject']):
            if info.loc[info['subject']==ID,:].shape[0]>2:#more than 2 timepoints 
                idset=info.loc[info['subject']==ID,:]
                idset['time'] = pd.to_numeric(idset['time'])
                idset=idset.sort_values(by=['time'])
                subset=pd.concat([subset,idset])         
                #nx,ny=ut.linesmooth(idset['time'].tolist(),idset['Y_pred'].tolist(),smooth=0.99,initial_w=0.1)   
                #regsubset=pd.concat([regsubset,pd.DataFrame({'subject':ID,'time':nx,'Y_pred':ny,'timetonegativity':idset['timetonegativity'][0]})])
        #regsubset.index=list(range(0,regsubset.shape[0]))
        subset['group']=0
        c=1
        for e in sorted(set(subset.loc[subset['disease state']!='Healthy Controls','time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        c=1
        for e in sorted(set(subset.loc[subset['disease state']=='Healthy Controls','time'])):
            subset.loc[(subset['disease state']=='Healthy Controls') & (subset.time==e),'group']=c
            c=c+1
        
        fig, axs = plt.subplots(2, 3,figsize=(15,10));
        colors=ut.subject_linecolor()
        for c,t in enumerate([28,56,84,168]):
            if c==0:
                ax=axs[0,0]
            elif c==1:
                ax=axs[0,1]
            elif c==2:
                ax=axs[0,2]
            else:
                ax=axs[1,0]
            sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['timetonegativity']==t,:], 
                         hue="subject", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
            sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['timetonegativity']==t,:], lw=2, markers=False, legend=False, ax=ax)
            #ax.axhline(y=cutoff, color='r', linestyle='--')
            gdetail()
            ax.set_title('Time to negativity: Day '+str(t) )
        ax=axs[1,1]
        sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['treatmentresult']=='Not Cured',:], 
                         hue="subject", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
        sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['treatmentresult']=='Not Cured',:], lw=2, markers=False, legend=False, ax=ax)
        gdetail()
        ax.set_title('Not cured')
        ax=axs[1,2]
        sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['disease state']=='Healthy Controls',:], 
                         hue="subject", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
        sns.lineplot(x='group', y='Y_pred', data=subset.loc[subset['disease state']=='Healthy Controls',:], lw=2, markers=False, legend=False, ci=80, ax=ax)
        gdetail()
        ax.set_title('Healthy control')
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_pat_dynamics.pdf')
        
        ############################################################
        #plot correlations between predicted score and mgit, xpert and PET-CT output
        fig, axs = plt.subplots(3, 2,figsize=(15,18));
        #mgit vs tb score
        subset=info.copy()
        subset=subset.loc[(~subset.mgit.isnull()),:]#Only look for TTP <=42
        ax=axs[0,0]
        ax=ut.correlationplot(subset, 'mgit', 'Y_pred', 'TB score vs MGIT', xlabel='Time to positivity (days)', ylabel='Predicted scores', save='', fig=0, stats=1, ax=ax)
        #Xpert vs tb score
        subset=info.copy()
        subset=subset.loc[(~subset.xpert.isnull()),:]#
        ax=axs[0,1]
        ax=ut.correlationplot(subset, 'xpert', 'Y_pred', 'TB score vs Xpert', xlabel='Cycle values', ylabel='Predicted scores', save='', fig=0, stats=1, ax=ax)
        #TRGA at baseline vs tb score
        subset=info.copy()
        subset=subset.loc[(subset['disease state']!='Healthy Controls')&(subset.time==0),:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata.loc[:,'TGRA1'], how='inner')
        ax=axs[1,0]
        ax=ut.correlationplot(subset, 'TGRA1', 'Y_pred', 'TB score vs TGRA1', xlabel='Total Glycolytic Ratio Activity (Day 0)', ylabel='Predicted scores (Day 0)', save='', fig=0, stats=1, log_x=1, ax=ax)
        #TRGA at month1 vs tb score
        subset=info.copy()
        subset=subset.loc[(subset['disease state']!='Healthy Controls')&(subset.time==28),:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata.loc[:,'TGRA2'], how='inner')
        ax=axs[1,1]
        ax=ut.correlationplot(subset, 'TGRA2', 'Y_pred', 'TB score vs TGRA2', xlabel='Total Glycolytic Ratio Activity (Day 28)', ylabel='Predicted scores (Day 28)', save='', fig=0, stats=1, log_x=1, ax=ax)
        #TRGA at month6 vs tb score
        subset=info.copy()
        subset=subset.loc[(subset['disease state']!='Healthy Controls')&(subset.time==168),:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata.loc[:,'TGRA3'], how='inner')
        ax=axs[2,0]
        ax=ut.correlationplot(subset, 'TGRA3', 'Y_pred', 'TB score vs TGRA3', xlabel='Log2 Total Glycolytic Ratio Activity (Day 168)', ylabel='Predicted scores (Day 168)', save='', fig=0, stats=1, log_x=1, ax=ax)
        #M6 load vs tb score at baseline
        subset=info.copy()
        subset=subset.loc[(subset['disease state']!='Healthy Controls')&(subset.time==0),:]
        subset.index=subset.subject.tolist()
        subset=subset.join(cdata.loc[:,'M6 Load'], how='inner')
        ax=axs[2,1]
        ax=ut.violinplot_compare(subset, 'M6 Load', 'Y_pred', ['cleared','persist'], 'TB score vs Presistent lung inflammation', xlabel='Lung inflammation (Day 168)', ylabel='Predicted scores (Day 0)', fig=0, stats=1, ax=ax)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_correlation.pdf')
        plt.show()
        
        return info,outputT
    
    def SA2015_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #It has two outcomes from this cohort -relapse and cured
        
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        fig, axs = plt.subplots(1, 2,figsize=(10,5));
        sns.set_theme(style="ticks")
         #Boxplot predicted score and time point
        info['group']=np.inf
        c=0
        for e in sorted(set(info.loc[:,'time'])):
            info.loc[info.time==e,'group']=c
            c=c+1
        ax=axs[0]
        ax=sns.violinplot(x="group", y="Y_pred", data=info, inner=None, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=info, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(['Day 0','Day 14','Day 28'],rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title(modelN)

        #Check longitudinal reponses of individuals 
        subset1=pd.DataFrame([])
        for ID in set(info['patient']):
            if info.loc[info['patient']==ID,:].shape[0]>1:#more than 1 timepoints 
                subset1=pd.concat([subset1,info.loc[info['patient']==ID,:]])
        ax=axs[1]
        colors=ut.subject_linecolor()
        sns.lineplot(x='group', y='Y_pred', data=subset1, hue="patient", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
        sns.lineplot(x='group', y='Y_pred', data=subset1, markers=False, legend=False, ax=ax)
        #ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['Day 0','Day 14','Day 28'],rotation=45)
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
        #ax.set_ylim([0,1])
        ax.set_title('Individual responses')
        
        """
        fig, axs = plt.subplots(2, 4,figsize=(22.5,7.5));
        sns.set_theme(style="ticks")
        col=0
        for each in ['Relapse','Cured']:
            #Boxplot predicted score and time point
            subset=info.loc[info['outcome']!=each,:]
            subset['group']=np.inf
            c=0
            for e in sorted(set(subset.loc[:,'time'])):
                subset.loc[subset.time==e,'group']=c
                c=c+1
            ax=axs[0,col]
            col=col+1
            ax=sns.violinplot(x="group", y="Y_pred", data=subset, inner=None, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
            # Add in points to show each observation
            ax=sns.stripplot(x="group", y="Y_pred", data=subset, size=4, color=".2", linewidth=0, ax=ax)
            ax.set_xticklabels(['Day 0','Day 14','Day 28'],rotation=45)
            ax.set_xlabel('')
            ax.set_ylabel('predicted scores')
            ax.set_title(modelN)

            #Check longitudinal reponses of individuals 
            subset1=pd.DataFrame([])
            for ID in set(subset['patient']):
                if subset.loc[subset['patient']==ID,:].shape[0]>1:#more than 1 timepoints 
                    subset1=pd.concat([subset1,subset.loc[subset['patient']==ID,:]])
            ax=axs[0,col]
            col=col+1
            sns.lineplot(x='group', y='Y_pred', data=subset, hue="patient", style="patient",markers=True, dashes=False, legend=False, ax=ax)
            #ax.axhline(y=cutoff, color='r', linestyle='--')
            ax.set_xticks([0,1,2])
            ax.set_xticklabels(['Day 0','Day 14','Day 28'],rotation=45)
            ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
            #ax.set_ylim([0,1])
            ax.set_title('Individual responses')
        
        #AUCROC evaluation against healthy control
        #subset=info.copy()
        #ax=axs[0]
        #output=[]
        #for i in sorted(set(subset.loc[subset['status']!='untreated latent TB','time'])):
        #    g1=subset.loc[subset.time==i,:]
        #    g1['group']=1
        #    g0=info.loc[info['status']=='untreated latent TB',:]
        #    g0['group']=0
        #    testdata=pd.concat([g1,g0])
        #    fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(testdata['group'].tolist(), testdata['Y_pred'].tolist())
        #    lower, upper=self.roc_auc_ci(roc, testdata['group'].tolist())
        #    output.append([title, 'Day'+str(i)+'_LTBI', roc, cutoff, tpr_cf, fpr_cf])
        #    ax.plot(fpr, tpr, label='Day '+str(i)+' vs LTBI: ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
        #    ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        #ax.set_xlabel('False Positive Rate')
        #ax.set_ylabel('True Positive Rate')
        #ax.set_title(title+' | ROCAUC against LTBI')
        #ax.legend(loc="lower right")
        """
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
        outputT=pd.DataFrame([])
        
        return info,outputT
    
    def Borstel_assess(self, modelN, final_model, info, gset_m, title,save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        #AUCROC evaluation
        output,info = self.valmodel_treatment_rocauc(info, 'time', title, save)
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Borstel_'+arg+'_'+modelN],columns=columns)
        
        fig, axs = plt.subplots(1, 2,figsize=(15,7.5));
        sns.set_theme(style="ticks")
        #Check longitudinal reponses of individuals 
        subset=info.loc[info['disease_state']!='Healthy Controls',:]
        subset1=pd.DataFrame([])
        for ID in set(subset['ID']):
            if subset.loc[subset['ID']==ID,:].shape[0]>2:#more than 1 timepoints 
                subset1=pd.concat([subset1,subset.loc[subset['ID']==ID,:]])
        ax=axs[0]
        sns.lineplot(x='time', y='Y_pred', data=subset1, hue="ID", dashes=False, legend=False, ax=ax)
        #ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
        ax.set_title(title+'Individual responses')
        
        ax=axs[1]
        pr,rpval = scipy.stats.spearmanr(subset1['time'],subset1['Y_pred'])
        sns.regplot(x='time', y='Y_pred', data=subset1, color='g',ci=95,truncate=False,ax=ax)
        sns.scatterplot(x='time', y='Y_pred', data=subset1, legend=False, hue='description',s=50,ax=ax)
        ax.annotate("Spearman $\itr$ = {:.2f}".format(pr) + "\n$\itp$-value = {:.4f}".format(rpval),xy=(.05, .78), xycoords=ax.transAxes, fontsize=15)
        ax.set_xlabel('Days in treatment', fontsize=10, weight='bold')
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold')
        ax.set_title(title)
        #ax.set_ylim([0,1])
        
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_pat_dynamics.pdf')
        plt.show()
        
        return info,outputT
    
    def SUFK_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        fig, axs = plt.subplots(1, 3,figsize=(22.5,7.5));
        sns.set_theme(style="ticks")
        
        #AUCROC evaluation against healthy control
        subset=info.copy()
        ax=axs[0]
        output=[]
        for i in sorted(set(subset.loc[subset['status']!='untreated latent TB','time'])):
            g1=subset.loc[subset.time==i,:]
            g1['group']=1
            g0=info.loc[info['status']=='untreated latent TB',:]
            g0['group']=0
            testdata=pd.concat([g1,g0])
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(testdata['group'].tolist(), testdata['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, testdata['group'].tolist())
            output.append([title, 'Day'+str(i)+'_LTBI', roc, cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label='Day '+str(i)+' vs LTBI: ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC against LTBI')
        ax.legend(loc="lower right")
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['SUFK_'+modelN],columns=columns)
        
        #Boxplot predicted score and time point
        subset=info.copy()
        subset['group']=np.inf
        c=0
        for e in sorted(set(subset.loc[subset['status']!='untreated latent TB','time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        subset.loc[subset['status']=='untreated latent TB','group']=c
        ax=axs[1]
        ax=sns.violinplot(x="group", y="Y_pred", data=subset, inner=None, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=subset, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(['Day 0','Day 15','Day 60','Day180','Day360','LTBI'],rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title('')
        
        #Check longitudinal reponses of individuals 
        subset=pd.DataFrame([])
        for ID in set(info['ID']):
            if info.loc[info['ID']==ID,:].shape[0]>1:#more than 1 timepoints 
                subset=pd.concat([subset,info.loc[info['ID']==ID,:]])
        subset['group']=0
        c=1
        for e in sorted(set(subset.loc[subset['status']!='untreated latent TB','time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        ax=axs[2]
        sns.lineplot(x='group', y='Y_pred', data=subset, hue="ID", style="ID",markers=True, dashes=False, legend=False, ax=ax)
        #ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(['Day 0','Day 15','Day 60','Day180','Day360'],rotation=45)
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
        #ax.set_ylim([0,1])
        ax.set_title('Individual responses')
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
        
        
        return info, outputT
    
    def SA_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        fig, axs = plt.subplots(1, 3,figsize=(18,6));
        sns.set_theme(style="ticks")
        
        #AUCROC evaluation against healthy control
        subset=info.copy()
        ax=axs[0]
        output=[]
        for i in sorted(set(subset.loc[:,'time'])):
            if i!=182:
                g1=subset.loc[subset.time==i,:]
                g1['group']=1
                g0=info.loc[info['time']==182,:]
                g0['group']=0
                testdata=pd.concat([g1,g0])
                fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(testdata['group'].tolist(), testdata['Y_pred'].tolist())
                lower, upper=self.roc_auc_ci(roc, testdata['group'].tolist())
                output.append([title, 'Day'+str(i)+'_Day182', roc, cutoff, tpr_cf, fpr_cf])
                ax.plot(fpr, tpr, label='Day '+str(i)+' vs Day182: ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
                ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC against Day182')
        ax.legend(loc="lower right")
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['SA_'+modelN],columns=columns)
        
        #Boxplot predicted score and time point
        subset=info.copy()
        subset['group']=np.inf
        c=0
        for e in sorted(set(subset.loc[:,'time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        ax=axs[1]
        ax=sns.violinplot(x="group", y="Y_pred", data=subset, inner=None, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=subset, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(['Day 0','Day 7','Day 14','Day28','Day182'],rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title('')
        
        #Check longitudinal reponses of individuals 
        subset=pd.DataFrame([])
        for ID in set(info['patient_id']):
            if info.loc[info['patient_id']==ID,:].shape[0]>1:#more than 1 timepoints 
                subset=pd.concat([subset,info.loc[info['patient_id']==ID,:]])
        subset['group']=0
        c=1
        for e in sorted(set(subset.loc[:,'time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        ax=axs[2]
        colors=ut.subject_linecolor()
        sns.lineplot(x='group', y='Y_pred', data=subset, hue="patient_id", markers=False, style=True, dashes=[(2,1)], lw=1, palette=colors, legend=False, ax=ax)
        sns.lineplot(x='group', y='Y_pred', data=subset, markers=False, legend=False, ax=ax)
        #ax.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(['Day 0','Day 7','Day 14','Day28','Day182'],rotation=45)
        ax.set_ylabel('Predicted score', fontsize=10, weight='bold');ax.set_xlabel('')
        #ax.set_ylim([0,1])
        ax.set_title('Individual responses')
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
        
        return info, outputT
    
    def Jakarta_assess(self, modelN, final_model, info, gset_m, title, save, cutoff=0, pubmodel=0, arg=''):
        #Predict outcome given the whole data
        if pubmodel==1:#calculate model score based on the published model
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=modelN)
        else:
            info=self.modelscore(info, gset_m, pubmodel=pubmodel, model=final_model)
        
        fig, axs = plt.subplots(1, 2,figsize=(12,6));
        sns.set_theme(style="ticks")
        
        #AUCROC evaluation against healthy control
        subset=info.copy()
        ax=axs[0]
        output=[]
        for i in sorted(set(subset.loc[subset['condition']!='Control','time'])):
            g1=subset.loc[subset.time==i,:]
            g1['group']=1
            g0=info.loc[info['condition']=='Control',:]
            g0['group']=0
            testdata=pd.concat([g1,g0])
            fpr, tpr, roc, cutoff, tpr_cf, fpr_cf = ut.rocauc(testdata['group'].tolist(), testdata['Y_pred'].tolist())
            lower, upper=self.roc_auc_ci(roc, testdata['group'].tolist())
            output.append([title, 'Day'+str(i)+'_Control', roc, cutoff, tpr_cf, fpr_cf])
            ax.plot(fpr, tpr, label='Day '+str(i)+' vs Healthy Control: ROCAUC = {0:0.2f} ({1:0.2f} to {2:0.2f})'.format(roc,lower,upper),lw=2, alpha=.8)
            ax.scatter(fpr_cf, tpr_cf, s=20, c='black')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random chance', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title+' | ROCAUC against Healthy Control')
        ax.legend(loc="lower right")
        columns=[];vals=[]
        for i in range(0,len(output)):
            columns.append(output[i][1])
            vals.append(output[i][2])
        outputT=pd.DataFrame([vals],index=['Jakarta_'+modelN],columns=columns)
        
        #Boxplot predicted score and time point
        subset=info.copy()
        subset['group']=np.inf
        c=0
        for e in sorted(set(subset.loc[subset['condition']!='Control','time'])):
            subset.loc[subset.time==e,'group']=c
            c=c+1
        subset.loc[subset['condition']=='Control','group']=c
        ax=axs[1]
        ax=sns.violinplot(x="group", y="Y_pred", data=subset, inner=None, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
        # Add in points to show each observation
        ax=sns.stripplot(x="group", y="Y_pred", data=subset, size=4, color=".2", linewidth=0, ax=ax)
        ax.set_xticklabels(['Day 0','Day 56','Day 196','Healthy Control'],rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('predicted scores')
        ax.set_title('')
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
        
        return info, outputT