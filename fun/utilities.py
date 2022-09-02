import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from statannotations.Annotator import Annotator
import scipy
import statsmodels
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, roc_curve, auc, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from csaps import csaps
import requests, sys

def intersection(lst1,lst2): 
    return list(set(lst1) & set(lst2)) 

def get_official_gene_symbol(gs):
    """Get official gene symbol from HGNC
    :param gs: str, gene symbol
    :return: off_gs, official HGNC gene symbol
    """
    gene_id_info = get_gene_id(gs)
    new_symbol = ''
    if len(gene_id_info)>0:
        for g in gene_id_info:
            server = "https://rest.ensembl.org/"
            ext = 'xrefs/id/'+g['id']+'?external_db=HGNC;all_levels=1;'
            r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
            id_info = r.json()
            if len(id_info)>0:
                if id_info[0]['display_id'] == gs:
                    new_symbol = gs
                    break
                else:
                    new_symbol = id_info[0]['display_id']
            else:
                return 'Gene not found'
        return new_symbol
    else:
        return 'Gene not found'

def get_gene_id(gs):
    """Get Ensembl gene ID of symbol
    :param gs: str, gene symbol
    :return: gene_id, Ensembl gene ID
    """
    server = "https://rest.ensembl.org/"
    ext = 'xrefs/symbol/homo_sapiens/'+gs+'?external_db=HGNC;feature_type=gene;all_levels=1;'
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit("Gene not found")
    gene_id_info = r.json()
    return gene_id_info

def genealias(name,alt=0):
    alias=dict()
    alias['NOV']='CCN3'
    alias['SEPT4']='SEPTIN4'
    alias['PNUTL2']='SEPTIN4'
    #alias['C17orf47']='SEPTIN4'
    #alias['CE5B3']='SEPTIN4'
    #alias['ARTS']='SEPTIN4'
    #alias['MART']='SEPTIN4'
    #alias['H5']='SEPTIN4'
    #alias['hCDCREL-2']='SEPTIN4'
    #alias['hucep-7']='SEPTIN4'
    #alias['FLJ40121']='SEPTIN4'
    
    
    if alt==1:
        return alias
    else:
        if name in alias.keys():
            name=alias[name]
        return name

def MissingValueRecovery(X):
    #Function: using K-nearst neighbors approach to calculate the missing values
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    newX=pd.DataFrame(imputer.fit_transform(X),index=X.index,columns=X.columns)
    return newX

def rocauc(y_true, y_predict, plot=0):
    #Function: calculate fpr and tpr and aread under curve
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    #Claculate the cutoff point (Youden's point) on the curve closet to the top left corner.
    #cutoff=thresholds[np.argmax(tpr - fpr)]
    cutoff= thresholds[np.argmin((1 - tpr) ** 2 + fpr ** 2)]
    tpr_cf= tpr[np.argmin((1 - tpr) ** 2 + fpr ** 2)]
    fpr_cf= fpr[np.argmin((1 - tpr) ** 2 + fpr ** 2)]

    if plot==1:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return fpr, tpr, roc_auc, cutoff, tpr_cf, fpr_cf


####################################################################################################
def subject_linecolor():#Draw individual line colors in time-series plot
    #return sns.cubehelix_palette(100, start=-0.1, rot=0, light=0.5, dark=0.2, as_cmap=True)
    return sns.cubehelix_palette(100, hue=0.1, rot=0, light=0.8, dark=0.6, as_cmap=True)

def linesmooth(x,y,method='cubsmooth',smooth=0.6, normalizedsmooth=True, initial_w=0.1):
    xi = range(int(x[0]), int(x[-1]))
    if method=='cubsmooth':
        #cubic smoothing splines
        #https://csaps.readthedocs.io/en/latest/tutorial.html
        w = np.ones_like(x) * initial_w
        w[0] = 1 #full weight at first point
        yi = csaps(x, y, xi, weights=w, smooth=smooth, normalizedsmooth=normalizedsmooth)
        #https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.interpolate.UnivariateSpline.html
        #spl = UnivariateSpline(x, y, s=smooth)
        #yi=spl(xi)
    elif method=='polynomial':
        n=3
        polymodel = np.poly1d(np.polyfit(x, y, 3))
        yi=polymodel(xi)

    return xi, yi

def violinplot_compare(data, x, y, order, title, hue=None, hue_order=None, xlabel='', ylabel='', save='', fig=1, stats=0, multi_cor=0, ax='', stripplot=1):
    #fig=1: generate a independent figure 
    #stats=1: add statistical signtificant annotation
    #multi_cor=1: perform multiple testing correction
    if fig==1: #the figure is a independent one
        sns.set_theme(style="ticks")
        figure, ax = plt.subplots(1, 1,figsize=(8,8));
    
    ax=sns.violinplot(x=x, y=y, data=data, hue=hue, order=order, hue_order=hue_order, whis=[0, 100], width=.6, palette="vlag", scale='width', ax=ax)
    # Add in points to show each observation
    if stripplot==1:
        ax=sns.stripplot(x=x, y=y, data=data, order=order, size=4, color=".2", linewidth=0, dodge=True, ax=ax)
        #ax=sns.swarmplot(x=x, y=y, data=data, order=order, color="k", alpha=0.8)
    if hue:
        ax.legend(loc='upper right')
    if stats==1:
        #Add statistics annotation
        #https://github.com/trevismd/statannotations/blob/master/usage/example.ipynb
        comp = []
        for i in range (0,len(order)):
            for j in range (i+1,len(order)):
                comp.append((order[i],order[j]))
        comp = tuple(comp)   
        annot = Annotator(ax, comp, data=data, x=x, y=y, order=order)
        comparisons_correction = 'Bonferroni' if multi_cor==1 else None
        annot.configure(comparisons_correction=comparisons_correction, test='Mann-Whitney', text_format='star', loc='inside', verbose=False)
        annot.apply_test().annotate()
    #ax.set_ylim([0,1.2])
    ax.set_xticklabels(order,rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if fig==1:
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
    
    return ax

def correlationplot(data, x, y, title, xlabel='', ylabel='', save='', fig=1, stats=0, ax='',textx=0.05, texty=0.05,log_x=0,log_y=0):
    if fig==1: #the figure is a independent one
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(1, 1,figsize=(8,8));
    
    if log_x==1:
        data[x]=np.log2(data[x])
    if log_y==1:
        data[y]=np.log2(data[y])
    sns.regplot(x=x, y=y, data=data, color='g',ci=95,truncate=False,ax=ax)
    sns.scatterplot(x=x, y=y, data=data, legend=False, s=75,ax=ax)
    
    if stats==1:
        pr,rpval = scipy.stats.spearmanr(data[x],data[y])
        if rpval< 0.0001:
            ax.annotate("Spearman $\itr$ = {:.2f}".format(pr) + "\n$\itp$-value < 0.0001",xy=(.05, .08), xycoords=ax.transAxes, fontsize=12)
        else:
            ax.annotate("Spearman $\itr$ = {:.2f}".format(pr) + "\n$\itp$-value = {:.4f}".format(rpval),xy=(textx, texty), xycoords=ax.transAxes, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10, weight='bold')
    ax.set_ylabel(ylabel, fontsize=10, weight='bold')
    ax.set_title(title)
    
    if fig==1:
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf')
        plt.show()
        
    return ax