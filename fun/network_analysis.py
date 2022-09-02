import os,pickle,sys,re,glob
import pandas as pd
import numpy as np
from os import listdir
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as pl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hier
from scipy.stats import zscore
from scipy.optimize import curve_fit
from scipy.stats import beta
import fastcluster
import seaborn as sns; sns.set_theme(color_codes=True);sns.set_style("white")
from sklearn.metrics import roc_auc_score
from venn import venn

class geneAssemClass():
    def __init__(self,cwd,datapath,outputpath):
        self.cwd=cwd
        self.datapath=datapath
        self.outputpath=outputpath
        
    def intersection(self,lst1,lst2): 
        return list(set(lst1) & set(lst2)) 
    
    def allgene(self,folder):
        fcout=pd.Series()
        c=0
        for f in listdir(folder):
            gmod=self.dataFetch(folder+'/'+f)
            new=[x for x in gmod.index if x not in fcout.index]
            fcout=fcout.append(pd.Series(np.zeros(len(new)),index=new))
            fcout[gmod.index]=fcout[gmod.index]+1
            c=c+1
        return list(fcout.index)
        #return list(fcout[fcout>=c].index)
    
    def dataFetch(self,file):
        gmod=pd.read_csv(file,sep=',',index_col=0)
        return gmod
    
    def histplot(self,nums,title,save):
        fig , ax = plt.subplots()
        n , bins , patches = plt.hist(nums ,bins = np.arange(np.min(nums), np.max(nums) + 1.5) , 
                                      color = 'k' , rwidth = 1.0 , edgecolor='white', linewidth=1.35 , align = 'left')
        plt.xlabel('number of datasets gene is significantly differentially expressed in', fontweight = 'bold' , fontsize = 10, color = 'k')
        plt.ylabel('number of genes', fontweight = 'bold' , fontsize = 10, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelcolor = 'k')
        ax.set_yscale('log')
        fig = plt.gcf()
        fig.set_size_inches(7.5, 4.25)
        fig.tight_layout()
        plt.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        plt.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        plt.title(title)
        plt.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def edge_weight_distribution(self,logFC_table_filename):
        #*Function* for constructing distribution of edge weights from pooled (significant) logFC changes across datasets

        # read in CSV file with significant logFC changes for genes between comparison X samples collected from various datasets
        diff_gene_exp_df = pd.read_csv(logFC_table_filename,sep=',',index_col=0)
        
        # Construct simplified matrix of logFC direction from DataFrame with significant logFC changes across all analyses by converting values:
        # +1 if logFC > 0
        # 0 if logFC = 0
        # -1 if logFC < 0

        # store copy of array from dataframe with sig. logFC values (rows = genes, columns = GSE ID)
        direc_diff_gene_exp_matrix = diff_gene_exp_df.copy().values 

        # replace values in logFC matrix
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix > 0.0] = 1
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix == 0.0] = 0
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix < 0.0] = -1

        # convert to lower memory int8 datatype
        direc_diff_gene_exp_matrix = direc_diff_gene_exp_matrix.astype('int8')

        # compute the dot product between every two pairs of gene vectors (will calculate the edges weights for our network)
        # multiply direction logFC matrix by its transpose to get the dot products between all pairs of rows
        network_edge_weight_matrix = direc_diff_gene_exp_matrix.dot(direc_diff_gene_exp_matrix.T)

        # the row/column annotation (genes) can be copied from the logFC differential gene expression DataFrame
        network_edge_weight_matrix_labels = pd.Series(list(diff_gene_exp_df.index) , index = range(0 , len(diff_gene_exp_df.index)))

        #number of rows / columns
        num_genes = np.shape(network_edge_weight_matrix)[0]

        # retrieve the distribution of the Edge Weights by returning the upper triangular part of the matrix
        edge_weight_array = network_edge_weight_matrix[np.triu_indices(num_genes, k = 0)]

        #convert array to a Counter dict to save space (keys: edge weight values, values: count of edge weights in edge weight distribution)
        edge_weight_distr_counter_dict = Counter(list(edge_weight_array))

        return edge_weight_distr_counter_dict, network_edge_weight_matrix,network_edge_weight_matrix_labels
    
    def plot_distribution_of_edge_weights(self,ax, edge_weight_distr_counter_dict, plot_title, linecolor):
        #convert Counter dict to series
        edge_weight_distr_series = pd.Series(edge_weight_distr_counter_dict).sort_index()

        #get the keys & values from Counter dict (with Edge Weight distr)
        edge_weight_values = edge_weight_distr_series.index
        edge_weight_count = edge_weight_distr_series.values
        edge_weight_count_norm = np.array(edge_weight_count) / float(np.sum(edge_weight_count)) #normalize counts

        ax.bar(edge_weight_values , edge_weight_count_norm, color = 'white' , width = 1.0 , edgecolor='black', linewidth=0.5)
        ax.bar(edge_weight_values[edge_weight_distr_series.index <= -3] , edge_weight_count_norm[edge_weight_distr_series.index <= -3], 
               color = 'blue' , width = 1.0 , edgecolor='black', linewidth=0.5)
        ax.bar(edge_weight_values[edge_weight_distr_series.index >= 3] , edge_weight_count_norm[edge_weight_distr_series.index >= 3], 
               color = 'blue' , width = 1.0 , edgecolor='black', linewidth=0.5)
        ax.plot(edge_weight_values , edge_weight_count_norm, color = linecolor , linewidth=2.0)

        # calculate the proportion of the edges that had weight = 0 (~ sparsity of the edge weight matrix)
        proportion_edge_weight_zero = round(float(edge_weight_distr_counter_dict[0]) / float(np.sum(edge_weight_count)), 4) * 100
        
        # calculate the number of the edges that had weight <= -3
        negative_edge_weights_for_network = edge_weight_count[edge_weight_distr_series.index <= -3].sum()

        # calculate the number of the edges that had weight >= 3
        positive_edge_weights_for_network = edge_weight_count[edge_weight_distr_series.index >= 3].sum()
    
        ax.set_title(plot_title, fontsize = 12, color = 'k')
        ax.set_ylabel('Proportion of Edges', fontsize = 12, color = 'k')
        ax.set_xlabel(f'Edge Weights\n{proportion_edge_weight_zero}% = 0\nNum Edges <= -3: {negative_edge_weights_for_network}\nNum Edges >= 3: {positive_edge_weights_for_network}',
                      fontsize = 12, color = 'k')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        ax.set_yscale('log')

        ax.tick_params(labelcolor = 'k')
        ax.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')

        for axis in [ax.yaxis]:
            axis.set_major_formatter(FormatStrFormatter('%.5f'))

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return
    
    def shuffle_rows_within_each_column(self,array):
        '''
        This function takes in a 2D-array and shuffles the rows of the array seperately for each column.
        '''
        #get number of rows & columns for input array
        nrows, ncols = array.shape

        #get the column indices for each element (will keep the same)
        cols = np.indices((nrows, ncols))[1]

        #permute the row indices for each column
        rows = np.array([np.random.permutation(nrows) for _ in range(ncols)]).T

        #re-arrange elements in each column according to the chosen row indices for that column
        shuffled_array = array[rows, cols]

        return shuffled_array
    
    def edge_weight_null_distribution(self,logFC_table_filename, N):
    
        # read in CSV file with significant logFC changes for genes between comparison X samples collected from various datasets
        diff_gene_exp_df = pd.read_csv(logFC_table_filename,sep=',',index_col=0)

        # Construct simplified matrix of logFC direction from DataFrame with significant logFC changes across all analyses by converting values:
        # +1 if logFC > 0
        # 0 if logFC = 0
        # -1 if logFC < 0

        # store copy of array from dataframe with sig. logFC values (rows = genes, columns = GSE ID)
        direc_diff_gene_exp_matrix = diff_gene_exp_df.copy().values 

        # replace values in logFC matrix
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix > 0.0] = 1
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix == 0.0] = 0
        direc_diff_gene_exp_matrix[direc_diff_gene_exp_matrix < 0.0] = -1

        # convert to lower memory int8 datatype
        direc_diff_gene_exp_matrix = direc_diff_gene_exp_matrix.astype('int8')

        # construct null distribution of shuffling the columns within each row, extracting edge weights, then adding to bucket of edge weights from many iterations
        edge_weight_null_distr_counter_dict = Counter([]) #initialize empty Counter dict

        # (shuffle columns within each row > get edge weights) x N > add to bucket of edge weights (null distribution)
        for iter_i in range(1, N+1):

            # permute the rows within each column once
            direc_diff_gene_exp_matrix_shuffled = self.shuffle_rows_within_each_column(direc_diff_gene_exp_matrix)

            # compute the dot product between every two pairs of gene vectors (will plot_null_v_actual_distributionscalculate the edges weights for our network)
            # multiply direction logFC matrix by its transpose to get the dot products between all pairs of rows
            network_edge_weight_matrix = direc_diff_gene_exp_matrix_shuffled.dot(direc_diff_gene_exp_matrix_shuffled.T)

            # the row/column annotation (genes) can be copied from the logFC differential gene expression DataFrame
            network_edge_weight_matrix_labels = pd.Series(list(diff_gene_exp_df.index) , index = range(0 , len(diff_gene_exp_df.index)))

            #number of rows / columns
            num_genes = np.shape(network_edge_weight_matrix)[0]

            # retrieve the distribution of the Edge Weights by returning the upper triangular part of the matrix
            edge_weight_array = network_edge_weight_matrix[np.triu_indices(num_genes, k = 0)]

            #convert array to a Counter dict to save space (keys: edge weight values, values: count of edge weights in edge weight distribution)
            edge_weight_distr_counter_dict = Counter(list(edge_weight_array))

            # append to counter with distribution of edge weights
            edge_weight_null_distr_counter_dict = edge_weight_null_distr_counter_dict + edge_weight_distr_counter_dict

            #if iter_i % 1 == 10:
            #    print(f'finished loop {iter_i}')

        return edge_weight_null_distr_counter_dict
    
    def plot_null_v_actual_distributions(self,edge_weight_distr_counter_dict, edge_weight_null_distr_counter_dict, comparison_X, save):
        '''
        Input - Counter dict with distribution of edge weight counts for Actual Distribution of Edge Weights & Null Distribution of Edge Weights
        '''
        #GET PROPORTIONS FROM LEFT & RIGHT TAILS FOR EACH DISTRIBUTION (split denominator as -<<0 & 0>>+)

        #convert Counter dicts to pandas series for easier indexing
        edge_weight_distr_counter_series = pd.Series(edge_weight_distr_counter_dict)
        edge_weight_null_distr_counter_series = pd.Series(edge_weight_null_distr_counter_dict)

        #get the count of edge weights for each value (normalized by the number of observations in each distribution)
        edge_weight_distr_counter_norm_series = edge_weight_distr_counter_series.astype(float) / float(edge_weight_distr_counter_series.sum())
        edge_weight_null_distr_counter_norm_series = edge_weight_null_distr_counter_series.astype(float) / float(edge_weight_null_distr_counter_series.sum())

        #compare left and right tails of actual & null distributions
        max_all_edge_weights = np.max([edge_weight_distr_counter_series.index.max(), edge_weight_null_distr_counter_series.index.max()]) #largest edge weight observed across both actual & null distributions
        min_all_edge_weights = np.min([edge_weight_distr_counter_series.index.min(), edge_weight_null_distr_counter_series.index.min()]) #smallest edge weight observed across both actual & null distributions

        #get values of edge weights (both + & -)
        negative_edge_weights = np.arange(min_all_edge_weights, 0)
        positive_edge_weights = np.arange(1, max_all_edge_weights+1)

        #number of edge weights in ACTUAL distribution
        num_values_edge_weight_distr_left = float(edge_weight_distr_counter_series[edge_weight_distr_counter_series.index <= 0].sum()) #left part of distr
        num_values_edge_weight_distr_right = float(edge_weight_distr_counter_series[edge_weight_distr_counter_series.index >= 0].sum()) #right part of distr

        #number of edge weights in NULL distribution
        num_values_edge_weight_null_distr_left = float(edge_weight_null_distr_counter_series[edge_weight_null_distr_counter_series.index <= 0].sum()) #left part of distr
        num_values_edge_weight_null_distr_right = float(edge_weight_null_distr_counter_series[edge_weight_null_distr_counter_series.index >= 0].sum()) #right part of distr

        #LEFT TAIL - Examine the distribution of edge weights for Null and Actual distributions for Edge Weights =< -1
        #collect the proportion of observations to the left of each edge weight for negative edge weights
        edge_weight_distr_frac_values_below = []
        edge_weight_null_distr_frac_values_below = []

        for edge_weight_i in negative_edge_weights:

            edge_weight_distr_frac_values_below.append(float(edge_weight_distr_counter_series[edge_weight_distr_counter_series.index <= edge_weight_i].sum()) / num_values_edge_weight_distr_left)
            edge_weight_null_distr_frac_values_below.append(float(edge_weight_null_distr_counter_series[edge_weight_null_distr_counter_series.index <= edge_weight_i].sum()) / num_values_edge_weight_null_distr_left)

        #RIGHT TAIL - Examine the distribution of edge weights for Null and Actual distributions for Edge Weights >= 1
        #collect the proportion of observations to the right of each edge weight for positive edge weights
        edge_weight_distr_frac_values_above = []
        edge_weight_null_distr_frac_values_above = []

        for edge_weight_i in positive_edge_weights:
            edge_weight_distr_frac_values_above.append(float(edge_weight_distr_counter_series[edge_weight_distr_counter_series.index >= edge_weight_i].sum()) / num_values_edge_weight_distr_right)
            edge_weight_null_distr_frac_values_above.append(float(edge_weight_null_distr_counter_series[edge_weight_null_distr_counter_series.index >= edge_weight_i].sum()) / num_values_edge_weight_null_distr_right)

        # PLOT DISTRIBUTIONS AND TAIL PROPORTIONS

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(4, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax4 = fig.add_subplot(gs[2, :])
        ax5 = fig.add_subplot(gs[3, 0])
        ax6 = fig.add_subplot(gs[3, 1])

        ################### Null Distribution ###################
        self.plot_distribution_of_edge_weights(ax1, edge_weight_null_distr_counter_dict, comparison_X+': Null Distribution', 'xkcd:red')

        ################### Actual Distribution ###################
        self.plot_distribution_of_edge_weights(ax2, edge_weight_distr_counter_dict, comparison_X+': Actual Distribution', 'xkcd:green')

        ############## Actual vs. Null Distribution 1 ##############
        #NORMALIZED HISTOGRAMS
        #convert Counter dict to series
        edge_weight_actual_distr_series = pd.Series(edge_weight_distr_counter_dict).sort_index()
        edge_weight_null_distr_series = pd.Series(edge_weight_null_distr_counter_dict).sort_index()

        #get the index & values from Counter Series (with Edge Weight distr)
        edge_weight_values_actual = edge_weight_actual_distr_series.index
        edge_weight_count_actual = edge_weight_actual_distr_series.values
        edge_weight_count_norm_actual = np.array(edge_weight_count_actual) / float(np.sum(edge_weight_count_actual))

        edge_weight_values_null = edge_weight_null_distr_series.index
        edge_weight_count_null = edge_weight_null_distr_series.values
        edge_weight_count_norm_null = np.array(edge_weight_count_null) / float(np.sum(edge_weight_count_null))

        ax3.plot(edge_weight_values_null , edge_weight_count_norm_null, color = 'xkcd:red' , linewidth=1.5)
        ax3.plot(edge_weight_values_actual , edge_weight_count_norm_actual, color = 'xkcd:green' , linewidth=1.5)

        ax3.set_ylabel('Proportion of Edges', fontsize = 12, color = 'k')
        ax3.set_xlabel('Edge Weights' , fontsize = 12, color = 'k')

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.grid(False)
        ax3.set_yscale('log')

        ax3.tick_params(labelcolor = 'k')
        ax3.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax3.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')

        for axis in [ax3.yaxis]:
            axis.set_major_formatter(FormatStrFormatter('%.5f'))

        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

        ############## Actual vs. Null Distribution 2 ##############
        edge_weight_values = []
        edge_weight_count_null_over_actual = []

        for edge_weight_i in np.sort(list(set(edge_weight_null_distr_series.index).union(set(edge_weight_actual_distr_series.index)))):

            if (edge_weight_i in edge_weight_null_distr_series.index) and (edge_weight_i in edge_weight_actual_distr_series.index):
                edge_weight_values.append(edge_weight_i)
                edge_weight_count_null_over_actual.append( (float(edge_weight_null_distr_series[edge_weight_i]) / float(edge_weight_null_distr_series.sum())) / (float(edge_weight_actual_distr_series[edge_weight_i]) / float(edge_weight_actual_distr_series.sum())) )
            elif (edge_weight_i in edge_weight_null_distr_series.index) and (edge_weight_i not in edge_weight_actual_distr_series.index):
                edge_weight_values.append(edge_weight_i)
                edge_weight_count_null_over_actual.append( np.nan )
            elif (edge_weight_i not in edge_weight_null_distr_series.index) and (edge_weight_i in edge_weight_actual_distr_series.index):
                edge_weight_values.append(edge_weight_i)
                edge_weight_count_null_over_actual.append( 0.0 )

        #NORM DISTRIBUTION / ACTUAL DISTRIBUTION
        edge_weight_count_null_over_actual_series = pd.Series(edge_weight_count_null_over_actual, index = edge_weight_values)

        #largest negative edge weight <= 0.05
        neg_edge_weight_thresh = np.max(edge_weight_count_null_over_actual_series[np.array([(edge_weight < 0) and (null_over_actual <= 0.05) for edge_weight, null_over_actual in zip(edge_weight_count_null_over_actual_series.index, edge_weight_count_null_over_actual_series.values)])].index)
        #smallest positive edge weight <= 0.05
        pos_edge_weight_thresh = np.min(edge_weight_count_null_over_actual_series[np.array([(edge_weight > 0) and (null_over_actual <= 0.05) for edge_weight, null_over_actual in zip(edge_weight_count_null_over_actual_series.index, edge_weight_count_null_over_actual_series.values)])].index)
        
        ax4.plot(edge_weight_count_null_over_actual_series.index , edge_weight_count_null_over_actual_series.values , color = 'xkcd:black' , linewidth=1.5)
        ax4.axhline(0.05 , color = 'red' , linestyle = 'dashed')
        ax4.axvline(neg_edge_weight_thresh , color = '0.5' , linewidth = 0.5)
        ax4.axvline(pos_edge_weight_thresh , color = '0.5' , linewidth = 0.5)

        ax4.set_ylabel('Norm Null Count / Norm Actual Count', fontsize = 12, color = 'k')
        ax4.set_xlabel('Edge Weights' , fontsize = 12, color = 'k')

        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.grid(False)

        ax4.tick_params(labelcolor = 'k')
        ax4.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax4.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')

        for axis in [ax4.yaxis]:
            axis.set_major_formatter(FormatStrFormatter('%.5f'))

        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

        ################### Left Tail ###################
        ax5.scatter(negative_edge_weights, edge_weight_null_distr_frac_values_below , color = 'xkcd:red' , 
                    alpha = 1.0 , s = 60 , linewidth = 1.0 , edgecolor = 'white' , label = 'Null Distribution')
        ax5.scatter(negative_edge_weights, edge_weight_distr_frac_values_below , color = 'xkcd:green' , 
                    alpha = 1.0 , s = 60 , linewidth = 1.0 , edgecolor = 'white' , label = 'Actual Distribution')

        ax5.set_title('Left Tail', fontsize = 12, color = 'k')
        ax5.set_ylabel('(#Edge Weights <= X)\ \n(#Edge Weights <= 0)', fontsize = 12, color = 'k')
        ax5.set_xlabel('Negative Edge Weights' , fontsize = 12, color = 'k')

        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.grid(False)
        ax5.set_yscale('log')

        ax5.tick_params(labelcolor = 'k')
        ax5.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax5.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

        ################### Right Tail ###################
        ax6.scatter(positive_edge_weights, edge_weight_null_distr_frac_values_above , color = 'xkcd:red' , 
                    alpha = 1.0 , s = 60 , linewidth = 1.0 , edgecolor = 'white' , label = 'Null Distribution')
        ax6.scatter(positive_edge_weights, edge_weight_distr_frac_values_above , color = 'xkcd:green' , 
                    alpha = 1.0 , s = 60 , linewidth = 1.0 , edgecolor = 'white' , label = 'Actual Distribution')

        ax6.set_title('Right Tail', fontsize = 12, color = 'k')
        ax6.set_ylabel('(#Edge Weights <= X)\ \n(#Edge Weights <= 0)', fontsize = 12, color = 'k')
        ax6.set_xlabel('Positive Edge Weights' , fontsize = 12, color = 'k')

        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.grid(False)
        ax6.set_yscale('log')

        ax6.tick_params(labelcolor = 'k')
        ax6.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax6.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax6.xaxis.set_major_locator(MaxNLocator(integer=True))

        ################### Plot ###################
        fig = plt.gcf()
        fig.set_size_inches(9.5, 14.5)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def networkplot(self, G,title):
        pos = nx.spring_layout(G , k = 0.35 , weight = 'weight' , iterations = 100)
        
        fig , ax = plt.subplots()
        nx.draw_networkx(
            G, 
            pos = pos, 
            ax = ax,
            node_size = 20, 
            node_color = 'xkcd:yellow',
            edge_color = "0.6",
            alpha = 0.2, 
            with_labels = False)

        fig = plt.gcf()
        fig.set_size_inches(9.0, 9.0)
        fig.tight_layout()
        plt.title(title)
        plt.show()
        return
    
    def draw_graph_and_color_subset_nodes(self, G, nodes_in_group, save):
        '''
        This function takes in a graph, list of a subset of nodes and axis object,
        then draws the network with the subset of nodes colored seperately from the 
        rest of the nodes in the network
        '''
        fig , ax = plt.subplots()
        pos = nx.spring_layout(G , k = 0.35 , weight = 'weight' , iterations = 100)
        #draw nodes NOT in the (subset) group
        size_map = []
        for node in G:
            if node not in nodes_in_group:
                size_map.append(25)
            elif node in nodes_in_group:
                size_map.append(0)

        nx.draw_networkx(
            G, 
            ax = ax,
            pos=pos, 
            node_size=size_map, 
            node_color = 'xkcd:grey',
            linewidths = 0.0,
            edge_color="0.7",
            width=0.0,
            alpha=0.25, 
            with_labels=False)

        #draw nodes that ARE in the (subset) group
        size_map = []
        for node in G:
            if node not in nodes_in_group:
                size_map.append(0)
            elif node in nodes_in_group:
                size_map.append(25)

        nx.draw_networkx(
            G, 
            pos=pos, 
            ax = ax,
            node_size=size_map, 
            node_color = 'xkcd:black',
            linewidths = 0.0,
            edge_color="0.7",
            width=0.0,
            alpha=0.7, 
            with_labels=False)
        
        fig = plt.gcf()
        fig.set_size_inches(9.0, 9.0)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def centralityDist(self,title,degree_series, weighted_degree_series,eigenvector_centrality_series,save):
            #Plot the distibution of centrality values for all nodes
            
            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(1, 3, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            #DEGREE CENTRALITY
            ax1.hist(degree_series , bins = 40, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
            ax1.set_ylabel('Number of Nodes', fontsize = 12, color = 'k')
            ax1.set_xlabel(f'Degree' , fontsize = 12, color = 'k')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.grid(False)
            ax1.tick_params(labelcolor = 'k')
            ax1.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
            ax1.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
            #WEIGHTED DEGREE CENTRALITY
            ax2.hist(weighted_degree_series , bins = 40, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
            ax2.set_ylabel('Number of Nodes', fontsize = 12, color = 'k')
            ax2.set_xlabel(f'Weighted Degree' , fontsize = 12, color = 'k')
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.grid(False)
            ax2.tick_params(labelcolor = 'k')
            ax2.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
            ax2.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
            #EIGENVECTOR CENTRALITY
            ax3.hist(eigenvector_centrality_series , bins = 40, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
            ax3.set_ylabel('Number of Nodes', fontsize = 12, color = 'k')
            ax3.set_xlabel(f'Eigenvector Centrality' , fontsize = 12, color = 'k')
            ax3.spines['right'].set_visible(False)
            ax3.spines['top'].set_visible(False)
            ax3.grid(False)
            ax3.tick_params(labelcolor = 'k')
            ax3.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
            ax3.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
            plt.suptitle(title)
            ################### Plot ###################
            fig = plt.gcf()
            fig.set_size_inches(15, 5.0)
            fig.tight_layout()
            plt.autoscale()
            plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
            plt.show()
            
            return
    
    def cor_wdegree_eigenvector(self,weighted_degree_series,eigenvector_centrality_series,save):
        fig , ax = plt.subplots()

        ax.scatter(weighted_degree_series[eigenvector_centrality_series.index], eigenvector_centrality_series , 
                   color = 'white' , linewidth=0.75 , edgecolor = 'black' , s = 20 , alpha = 0.75)

        #ax.set_title('Degree vs. Eigenvector Centrality', fontsize = 10, color = 'k')
        ax.set_ylabel('Eigenvector Centrality', fontsize = 12, color = 'k')
        ax.set_xlabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelcolor = 'k')
        ax.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')

        fig = plt.gcf()
        fig.set_size_inches(7.5, 5.5)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def meanlogFC_Dis(self,mean_logFC_series,title,save):
        fig , ax = plt.subplots()
        n, bins, patches = ax.hist(mean_logFC_series , bins = 70, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
        ax.set_title(f'distribution of mean log2(FC) for genes in network', fontsize = 12, color = 'k')
        ax.set_ylabel(f'number of genes (nodes)', fontsize = 12, color = 'k')
        ax.set_xlabel('mean log2(FC) across datasets' , fontsize = 12, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelcolor = 'k')
        ax.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        plt.title(title)
        
        fig = plt.gcf()
        fig.set_size_inches(7.0, 5.5)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def dis_weighted_degree_ave_logfc(self,df,top_N_nodes,title,save):
        #Distributions of Weighted Degree and mean log2(Fold Change)
        
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        #WEIGHTED DEGREE CENTRALITY
        n, bins, patches = ax1.hist(df.weighted_degree , bins = 50, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
        ax1.hist(df.sort_values(by = 'weighted_degree', ascending = False).weighted_degree.head(n=top_N_nodes) , bins = bins, color = 'blue' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)

        ax1.set_ylabel(f'Number of Genes (nodes)', fontsize = 12, color = 'k')
        ax1.set_xlabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.grid(False)
        ax1.tick_params(labelcolor = 'k')
        ax1.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax1.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax1.set_title(title, fontsize = 12, color = 'k')
        #Log2(FC)
        n, bins, patches = ax2.hist(df.mean_log2FC , bins = 50, color = 'black' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)
        ax2.hist(df.sort_values(by = 'weighted_degree', ascending = False).mean_log2FC.head(n=top_N_nodes) , bins = bins, color = 'blue' , rwidth = 1.0 , edgecolor='white', linewidth=0.75)

        ax2.set_ylabel('Number of Genes (nodes)', fontsize = 12, color = 'k')
        ax2.set_xlabel('Mean log2(FC) Across Datasets' , fontsize = 12, color = 'k')

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.grid(False)
        ax2.tick_params(labelcolor = 'k')
        ax2.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ax2.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax2.set_title(title, fontsize = 12, color = 'k')
        ################### Plot ###################
        fig = plt.gcf()
        fig.set_size_inches(14, 7.0)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def volcanoplot(self,df,title,save):
        #Weighted Degree vs mean log2(Fold Change)
        
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        #PLOT 1: Log2(FC) vs. weighted degree
        ax1.scatter(df.mean_log2FC, df.weighted_degree,
                    color = "black" , linewidth=0.0 , edgecolor = 'black' , s = 35 , alpha = 0.4)

        ax1.set_title(title, fontsize = 12, color = 'k')
        ax1.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax1.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.grid(False)
        ax1.tick_params(labelcolor = 'k')
        ax1.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax1.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')

        #PLOT 2: absolute(Log2(FC)) vs. weighted degree
        ax2.scatter(abs(df.mean_log2FC), df.weighted_degree, color = "black" , linewidth=0.0 , edgecolor = 'black' , s = 35 , alpha = 0.4)

        ax2.set_title(title, fontsize = 12, color = 'k')
        ax2.set_xlabel('|Mean log2(FC)|', fontsize = 12, color = 'k')
        ax2.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.grid(False)
        ax2.tick_params(labelcolor = 'k')
        ax2.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax2.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')

        ################### Plot ###################
        fig = plt.gcf()
        fig.set_size_inches(15, 7.5)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def volcanoplot_highlight(self,df,top_N_nodes,title):
        fig , ax = plt.subplots()
        ax.scatter(df.mean_log2FC,df.weighted_degree, 
                   color = "0.7" , linewidth=0.0 , edgecolor = 'black' , s = 40 , alpha = 0.55)
        #highlight the top N nodes by weighted degree
        ax.scatter(df.sort_values(by = 'weighted_degree', ascending = False).mean_log2FC.head(n=top_N_nodes),
                   df.sort_values(by = 'weighted_degree', ascending = False).weighted_degree.head(n=top_N_nodes),
                   color = 'blue' , linewidth=0.5 , edgecolor = 'white' , s = 40 , alpha = 1.0)

        ax.set_title(title, fontsize = 12, color = 'k')
        ax.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelcolor = 'k')
        ax.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')

        fig = plt.gcf()
        fig.set_size_inches(10.5, 8.0)
        fig.tight_layout()
        plt.show()
        return
    
    def plot_logFC_v_meanlogFC_all_networks(self, ATB_HC_df, ATB_LTBI_df, ATB_OD_df, ATB_Tret_df, gene_list, gene_list_name,save):
        #Function to visualize a set of genes in mean log2(FC) vs. weighted degree for all networks
        
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        #ATB_HC
        ax1.scatter(ATB_HC_df.mean_log2FC, ATB_HC_df.weighted_degree, 
                    color = "0.7" , linewidth=0.0 , edgecolor = 'black' , s = 25 , alpha = 0.55)
        #filter for nodes in gene list
        gene_list_filter = [gene_i in gene_list for gene_i in ATB_HC_df.index]

        #highlight the top N nodes by weighted degree
        ax1.scatter(ATB_HC_df[gene_list_filter].mean_log2FC, ATB_HC_df[gene_list_filter].weighted_degree, 
                    color = 'blue' , linewidth=0.5 , edgecolor = 'white' , s = 25 , alpha = 1.0)
        ax1.set_title('ATB v HC', fontsize = 12, color = 'k')
        ax1.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax1.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.grid(False)
        ax1.tick_params(labelcolor = 'k')
        ax1.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax1.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')

        #ATB_LTBI
        ax2.scatter(ATB_LTBI_df.mean_log2FC, ATB_LTBI_df.weighted_degree, 
                    color = "0.7" , linewidth=0.0 , edgecolor = 'black' , s = 25 , alpha = 0.55)
        #filter for nodes in gene list
        gene_list_filter = [gene_i in gene_list for gene_i in ATB_LTBI_df.index]
        #highlight the top N nodes by weighted degree
        ax2.scatter(ATB_LTBI_df[gene_list_filter].mean_log2FC, ATB_LTBI_df[gene_list_filter].weighted_degree, 
                    color = 'blue' , linewidth=0.5 , edgecolor = 'white' , s = 25 , alpha = 1.0)
        ax2.set_title('ATB v LTBI', fontsize = 12, color = 'k')
        ax2.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax2.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.grid(False)
        ax2.tick_params(labelcolor = 'k')
        ax2.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax2.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')

        #ATB_OD
        ax3.scatter(ATB_OD_df.mean_log2FC, ATB_OD_df.weighted_degree, 
                    color = "0.7" , linewidth=0.0 , edgecolor = 'black' , s = 25 , alpha = 0.55)
        #filter for nodes in gene list
        gene_list_filter = [gene_i in gene_list for gene_i in ATB_OD_df.index]
        #highlight the top N nodes by weighted degree
        ax3.scatter(ATB_OD_df[gene_list_filter].mean_log2FC, ATB_OD_df[gene_list_filter].weighted_degree, 
                    color = 'blue' , linewidth=0.5 , edgecolor = 'white' , s = 25 , alpha = 1.0)
        ax3.set_title('ATB v OD', fontsize = 12, color = 'k')
        ax3.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax3.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.grid(False)
        ax3.tick_params(labelcolor = 'k')
        ax3.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax3.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        
        #ATB_Tret
        ax4.scatter(ATB_Tret_df.mean_log2FC, ATB_Tret_df.weighted_degree, 
                    color = "0.7" , linewidth=0.0 , edgecolor = 'black' , s = 25 , alpha = 0.55)
        #filter for nodes in gene list
        gene_list_filter = [gene_i in gene_list for gene_i in ATB_Tret_df.index]
        #highlight the top N nodes by weighted degree
        ax4.scatter(ATB_Tret_df[gene_list_filter].mean_log2FC, ATB_Tret_df[gene_list_filter].weighted_degree, 
                    color = 'blue' , linewidth=0.5 , edgecolor = 'white' , s = 25 , alpha = 1.0)
        ax4.set_title('ATB v Tret', fontsize = 12, color = 'k')
        ax4.set_xlabel('Mean log2(FC)', fontsize = 12, color = 'k')
        ax4.set_ylabel('Weighted Degree' , fontsize = 12, color = 'k')

        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.grid(False)
        ax4.tick_params(labelcolor = 'k')
        ax4.tick_params(axis='x', which='major', labelsize=12 , labelcolor = 'k')
        ax4.tick_params(axis='y', which='major', labelsize=12 , labelcolor = 'k')
        ################### Plot ###################
        fig = plt.gcf()
        fig.set_size_inches(16.5, 5.5)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'.pdf', dpi = 300 , transparent = True)
        plt.show()
        return
    
    def plot_logFC_heatmap(self, compares, pooled_diff_gene_exp_data, gene_set, gene_set_name,save,sameorder=0):
    
        '''
        Function to plot heatmap of diff gene exp (logFC) for a set of genes
        '''
        fig_width_dict = {'ATB_v_HC':2, 'ATB_v_LTBI':2, 'ATB_v_OD':3, 'ATB_v_Tret':1.5}
        fig_label_dict = {'ATB_v_HC':'ATB vs HC', 'ATB_v_LTBI':'ATB vs LTBI', 'ATB_v_OD':'ATB vs OD', 'ATB_v_Tret':'ATB vs Treatment' }
        mpl.rcParams['axes.linewidth'] = 0.35
        if sameorder==1:
            common_order=[]
        
        for e in compares:
            #load the matrix with the differential log2FC (that passed q-val & abs(logFC) criteria) for this comparison
            diff_gene_exp_df = pd.read_csv(pooled_diff_gene_exp_data+'/'+e+'_fll.csv', sep=',', index_col=0)
            diff_gene_exp_df = diff_gene_exp_df.sort_index(axis=1)
            #Convert logfc to zscore
            diff_gene_exp_df=diff_gene_exp_df.apply(zscore)
            #subset matrix to genes in gene set
            #gene_subset_filter = np.array([gene_id in gene_set for gene_id in diff_gene_exp_df.index])
            diff_gene_exp_subset_df = diff_gene_exp_df.loc[gene_set,:]
            #cluster the rows (genes)
            gene_exp_link = fastcluster.linkage(diff_gene_exp_subset_df, method='ward', metric='euclidean')
            #get the new order of the genes
            gene_order = hier.leaves_list(gene_exp_link)
            if e=='ATB_v_HC' and sameorder==1:
                common_order=gene_order
            if sameorder==0:
                #re-order the log2FC matrix according to the new clustering order
                reordered_data_subset = diff_gene_exp_subset_df.values[gene_order, :]
            else:
                #re-order the log2FC matrix according to the new clustering order
                reordered_data_subset = diff_gene_exp_subset_df.values[common_order, :]
            
            num_genes = float(len(gene_set))

            if num_genes <= 5:
                height_ratios = [15,num_genes]
                fig_height = 0.15*num_genes
            else:
                height_ratios = [35,45.0/num_genes]
                fig_height = 0.105*num_genes

            fig = plt.figure(figsize=(fig_width_dict[e], fig_height), dpi=300)
            
            gs = GridSpec(2, 2,
                       width_ratios=[1.5,10],                
                       height_ratios=height_ratios,
                       wspace=0.03,
                       hspace=1.5/num_genes)

            #DENDROGRAM
            if sameorder==0:
                ax2 = fig.add_subplot(gs[0,0], frameon=False)
                Z2 = dendrogram(Z=gene_exp_link, color_threshold=0, above_threshold_color = 'k', leaf_rotation=45, no_labels = True , orientation='left', ax=ax2) # adding/removing the axes
                ax2.set_xticks([])
                
            #HEATMAP
            if sameorder==0:
                axmatrix = fig.add_subplot(gs[0,1])
            else:
                axmatrix = fig.add_subplot(gs[0,:])
            abs_max_dataset = np.max( [diff_gene_exp_subset_df.max().max() , abs(diff_gene_exp_subset_df.min().min())] )
            norm = mpl.colors.Normalize(vmin = -1*abs_max_dataset , vmax = abs_max_dataset) #get the normalization
            im = axmatrix.matshow(reordered_data_subset, aspect='auto', origin='lower', cmap=plt.cm.seismic, interpolation='none', norm=norm)

            axmatrix.grid(False)
            axmatrix.tick_params(labelcolor = 'k')
            axmatrix.yaxis.set_label_position("right")
            axmatrix.yaxis.tick_right()
            
            if sameorder==0:
                gene_labels = list(diff_gene_exp_subset_df.index[gene_order])
            else:
                gene_labels = list(diff_gene_exp_subset_df.index[common_order])
            axmatrix.set_yticks(range(0,len(gene_labels)))
            axmatrix.set_yticklabels(gene_labels, rotation='0', fontsize = 4.5, color = 'k')
            plt.tick_params(axis = "y", which = "both", left = False, right = True, color = 'k', width = 0.5)

            dataset_labels = list(diff_gene_exp_subset_df.columns)
            axmatrix.set_xticks(range(0,len(dataset_labels)))
            axmatrix.set_xticklabels(dataset_labels, rotation='90', fontsize = 4.5, color = 'k')
            plt.tick_params(axis = "x", which = "both", bottom = False, top = True, color = 'k', width = 0.5)

            #COLORBAR
            if sameorder==0:
                ax_cbar = fig.add_subplot(gs[1,1])
            else:
                ax_cbar = fig.add_subplot(gs[1,:])
            cb2 = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.cm.seismic, norm = norm, orientation='horizontal')
            ax_cbar.tick_params(axis = "x", which = "both", bottom = False, top = False, color = 'k', width = 0.5, labelsize = 5)
            ax_cbar.tick_params(axis='x', which='major', pad=-2)
            ax_cbar.set_xlabel(fig_label_dict[e], fontsize = 5, color = 'k')
            
            fig.tight_layout()
            plt.autoscale()
            if sameorder==0:
                plt.savefig(save+'_'+e+'.pdf', dpi = 300 , transparent = True)
            else:
                plt.savefig(save+'_'+e+'_sameorder.pdf', dpi = 300 , transparent = True)
                
            plt.show() 
        return
    
    def rule_geneselection1(self,topN,sort,AH_pos,AH_neg,AL_pos,AL_neg,AT_pos,AT_neg,AO_pos,AO_neg):
        
        top_pos_AH = set(AH_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AH = set(AH_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AL = set(AL_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AL = set(AL_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AT = set(AT_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AT = set(AT_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AO = set(AO_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AO = set(AO_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        
        target = top_pos_AH.union(top_pos_AL).union(top_pos_AT).union(top_pos_AO)
        target = target.union(top_neg_AH).union(top_neg_AL).union(top_neg_AT).union(top_neg_AO)
        return list(target)
    
    def rule_geneselection2(self,topN,sort,AH_pos,AH_neg,AL_pos,AL_neg,AT_pos,AT_neg,AO_pos,AO_neg):
        #Top genes
        top_pos_AH = set(AH_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AH = set(AH_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AL = set(AL_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AL = set(AL_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AT = set(AT_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AT = set(AT_neg.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_pos_AO = set(AO_pos.sort_values(by = sort, ascending = False).head(n=topN).index)
        top_neg_AO = set(AO_neg.sort_values(by = sort, ascending = False).head(n=topN).index)

        #1 ATB_HC ∩ ATB_LTBI
        AH_AL_P = top_pos_AH.intersection(top_pos_AL)
        AH_AL_N = top_neg_AH.intersection(top_neg_AL)
        #ATB_HC ∩ ATB_OD'
        AH_AO_P = top_pos_AH.intersection(top_pos_AO)
        AH_AO_N = top_neg_AH.intersection(top_neg_AO)
        #'ATB_HC ∩ ATB_Tret
        AH_AT_P = top_pos_AH.intersection(top_pos_AT)
        AH_AT_N = top_neg_AH.intersection(top_neg_AT)
        #ATB_LTBI ∩ ATB_OD
        AL_AO_P = top_pos_AL.intersection(top_pos_AO)
        AL_AO_N = top_neg_AL.intersection(top_neg_AO)
        #ATB_LTBI ∩ ATB_Tret
        AL_AT_P = top_pos_AL.intersection(top_pos_AT)
        AL_AT_N = top_neg_AL.intersection(top_neg_AT)
        #ATB_OD ∩ ATB_Tret
        AO_AT_P = top_pos_AO.intersection(top_pos_AT)
        AO_AT_N = top_neg_AO.intersection(top_neg_AT)
        
        #(ATB_HC ∩ ATB_LTBI) ∪ (ATB_HC ∩ ATB_OD) ∪ (ATB_HC ∩ ATB_Tret) ∪ (ATB_LTBI ∩ ATB_OD) ∪ (ATB_LTBI ∩ ATB_Tret) ∪ (ATB_OD ∪ ATB_Tret)
        target = AH_AL_P.union(AH_AO_P).union(AH_AT_P).union(AL_AO_P).union(AL_AT_P).union(AO_AT_P)
        target = target.union(AH_AL_N).union(AH_AO_N).union(AH_AT_N).union(AL_AO_N).union(AL_AT_N).union(AO_AT_N)

        return list(target)
    
    def rule_geneselection3(self,top_percent,ATB_HC_df,ATB_LTBI_df,ATB_OD_df,ATB_Tret_df,save):
        #Determine top % genes from the degree distribution (assume it's a power-law distribution [a generalized beta distribution])
        fig = plt.figure(constrained_layout=True,figsize=(10, 8))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        top_AH=self.beta_distribution_fit(ATB_HC_df,top_percent,'ATB vs HC',ax1)
        top_AL=self.beta_distribution_fit(ATB_LTBI_df,top_percent,'ATB vs LTBI',ax2)
        top_AO=self.beta_distribution_fit(ATB_OD_df,top_percent,'ATB vs OD',ax3)
        top_AT=self.beta_distribution_fit(ATB_Tret_df,top_percent,'ATB vs Tret',ax4)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_degree_dist_fit.pdf', dpi = 300 , transparent = True)
        
        top_AH_set = set(top_AH.index)
        top_AL_set = set(top_AL.index)
        top_AO_set = set(top_AO.index)
        top_AT_set = set(top_AT.index)
        
        """
        #abs(logFC)>0.5
        top_pos_AH = set(top_AH.loc[top_AH['mean_log2FC']>=0.5,:].index)
        top_neg_AH = set(top_AH.loc[top_AH['mean_log2FC']<=-0.5,:].index)
        top_pos_AL = set(top_AL.loc[top_AL['mean_log2FC']>=0.5,:].index)
        top_neg_AL = set(top_AL.loc[top_AL['mean_log2FC']<=-0.5,:].index)
        top_pos_AT = set(top_AT.loc[top_AT['mean_log2FC']>=0.5,:].index)
        top_neg_AT = set(top_AT.loc[top_AT['mean_log2FC']<=-0.5,:].index)
        top_pos_AO = set(top_AO.loc[top_AO['mean_log2FC']>=0.5,:].index)
        top_neg_AO = set(top_AO.loc[top_AO['mean_log2FC']<=-0.5,:].index)
        print(top_pos_AO)
        
        vendigram_pos = {
            "ATB_vs_HC_up": top_pos_AH,
            "ATB_vs_LTBI_up": top_pos_AL,
            "ATB_vs_OD_up": top_pos_AO,
            "ATB_vs_Tret_up":top_pos_AT
        }
        vendigram_neg = {
            "ATB_vs_HC_down": top_neg_AH,
            "ATB_vs_LTBI_down": top_neg_AL,
            "ATB_vs_OD_down": top_neg_AO,
            "ATB_vs_Tret_down":top_neg_AT
        }
        venn(vendigram_pos)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_venndiagram_up.pdf', dpi = 300 , transparent = True)
        venn(vendigram_neg)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_venndiagram_down.pdf', dpi = 300 , transparent = True)
        plt.show()
        
        #1 ATB_HC ∩ ATB_LTBI
        AH_AL_P = top_pos_AH.intersection(top_pos_AL)
        AH_AL_N = top_neg_AH.intersection(top_neg_AL)
        #ATB_HC ∩ ATB_OD'
        #AH_AO_P = top_pos_AH.intersection(top_pos_AO)
        #AH_AO_N = top_neg_AH.intersection(top_neg_AO)
        #'ATB_HC ∩ ATB_Tret
        #AH_AT_P = top_pos_AH.intersection(top_pos_AT)
        #AH_AT_N = top_neg_AH.intersection(top_neg_AT)
        #ATB_LTBI ∩ ATB_OD
        #AL_AO_P = top_pos_AL.intersection(top_pos_AO)
        #AL_AO_N = top_neg_AL.intersection(top_neg_AO)
        #ATB_LTBI ∩ ATB_Tret
        #AL_AT_P = top_pos_AL.intersection(top_pos_AT)
        #AL_AT_N = top_neg_AL.intersection(top_neg_AT)
        #ATB_OD ∩ ATB_Tret
        #AO_AT_P = top_pos_AO.intersection(top_pos_AT)
        #AO_AT_N = top_neg_AO.intersection(top_neg_AT)
        
        #(ATB_HC ∩ ATB_LTBI) ∪ (ATB_HC ∩ ATB_OD) ∪ (ATB_HC ∩ ATB_Tret) ∪ (ATB_LTBI ∩ ATB_OD) ∪ (ATB_LTBI ∩ ATB_Tret) ∪ (ATB_OD ∪ ATB_Tret)
        #target = AH_AL_P.union(AH_AO_P).union(AH_AT_P).union(AL_AO_P).union(AL_AT_P).union(AO_AT_P)
        #target = target.union(AH_AL_N).union(AH_AO_N).union(AH_AT_N).union(AL_AO_N).union(AL_AT_N).union(AO_AT_N)
        
        #(ATB_HC ∩ ATB_LTBI) ∪ ATB_OD ∪ ATB_Tret
        target = AH_AL_P.union(top_pos_AT).union(top_pos_AO)
        target = target.union(AH_AL_N).union(top_neg_AT).union(top_neg_AO)
        """
        
        vendigram = {
            "ATB_vs_HC": top_AH_set,
            "ATB_vs_LTBI": top_AL_set,
            "ATB_vs_OD": top_AO_set,
            "ATB_vs_Tret":top_AT_set
        }
        venn(vendigram)
        fig.tight_layout()
        plt.autoscale()
        plt.savefig(save+'_venndiagram.pdf', dpi = 300 , transparent = True)
        
        AH_AL = top_AH_set.intersection(top_AL_set)
        AH_AO = top_AH_set.intersection(top_AO_set)
        AH_AT = top_AH_set.intersection(top_AT_set)
        AL_AO = top_AL_set.intersection(top_AO_set)
        AL_AT = top_AL_set.intersection(top_AT_set)
        AO_AT = top_AO_set.intersection(top_AT_set)
        target = AH_AL.union(AH_AO).union(AH_AT).union(AL_AO).union(AL_AT).union(AO_AT)
        
        target=target.intersection(ATB_HC_df.loc[abs(ATB_HC_df['mean_log2FC'])>=0.5,:].index)
        target=target.intersection(ATB_LTBI_df.loc[abs(ATB_LTBI_df['mean_log2FC'])>=0.5,:].index)
        target=target.intersection(ATB_Tret_df.loc[abs(ATB_Tret_df['mean_log2FC'])>=0.25,:].index)
        target=target.intersection(ATB_OD_df.loc[(ATB_OD_df['mean_log2FC']>=0.25)|(ATB_OD_df['mean_log2FC']<=-0.1),:].index)
        #print(len(target))
        return list(target)
    
    def beta_distribution_fit(self, df, top_percent, title, ax):
        #Funtion: fit the weighted degree data into a generalized beta distribution
        
        a, b, loc, scale = beta.fit(df.weighted_degree)
        x=np.linspace(5,int(max(df.weighted_degree)),500)
        yfit=beta.pdf(x, a,  b, loc, scale)
        cutoff=beta.ppf(1-float(top_percent/100), a, b,loc=loc, scale=scale)
        
        ax.hist(df.weighted_degree,bins = 50,density=True)
        ax.plot(x, yfit,'--')
        ax.axvline(cutoff)
        ax.set_xlabel('Weighted degree')
        ax.set_ylabel('Probability density function')
        ax.set_title(title)
        
        df=df.loc[df.weighted_degree>=cutoff,:]
        return df
    
    def sum_AUC_genelist(self, genes,nor_exp_data):
        #Function : Calculate averages of AUCs for TB disease stage differentiation given the gene sets
        
        #load all study list
        datalist = pd.read_csv(self.cwd+'/data_list.csv',sep=',',index_col=0)
        all=dict()
        for i in datalist.index:
            data1 = pd.read_csv(nor_exp_data + '/' + datalist.loc[i,'GSEID'] + '_' + datalist.loc[i,'Condition1'] + '_' + datalist.loc[i,'Type'] + '_Exp_EachGene.csv', sep=',',index_col=0)
            match=self.intersection(data1.index,genes)
            data1=data1.loc[match,]
            all[datalist.loc[i,'GSEID']+'_'+re.sub(' ','_',datalist.loc[i,'Condition1'])]=data1
            
            data2 = pd.read_csv(nor_exp_data + '/' + datalist.loc[i,'GSEID'] + '_' + datalist.loc[i,'Condition2'] + '_' + datalist.loc[i,'Type'] + '_Exp_EachGene.csv', sep=',', index_col=0)
            match=self.intersection(data2.index,genes)
            data2=data2.loc[match,]
            all[datalist.loc[i,'GSEID']+'_'+re.sub(' ','_',datalist.loc[i,'Condition2'])]=data2
        
        compares=['ATB_v_HC','ATB_v_LTBI','ATB_v_OD','ATB_v_Tret']
        results=pd.DataFrame(np.full((len(genes),len(compares)), np.nan),columns=compares,index=genes)
        for e in compares:
            subset=datalist.loc[datalist['Compare']==e,:]
            for eg in genes:
                rocaucs=[]
                for s in subset.index:
                    G1=all[subset.loc[s,'GSEID']+'_'+re.sub(' ','_',subset.loc[s,'Condition1'])]
                    G0=all[subset.loc[s,'GSEID']+'_'+re.sub(' ','_',subset.loc[s,'Condition2'])]
                    y=np.concatenate([np.ones(G1.shape[1]),np.zeros(G0.shape[1])])
                    if eg in G1.index and eg in G0.index: 
                        eg_exp=G1.loc[eg,:].tolist()+G0.loc[eg,:].tolist()
                        rocaucs.append(roc_auc_score(y, eg_exp))
                results.loc[eg,e]=np.mean(rocaucs)
        return results