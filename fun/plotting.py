
import plotnine
from plotnine import *
import pandas as pd
import numpy as np
import fastcluster
from . import network_analysis as na
from mizani.formatters import comma_format
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

blue = '#4E79A7'
orange = '#F28E2B'
green = '#59A14F'
lightgray = '#C7C7C7'

palette = [
    blue,
    orange,
    green,
    lightgray
]

def plot_expression_distributions(df_exprs, gse_id):
    """
    """
    plotnine.options.figure_size = (16, 3)
    df_ = df_exprs.copy()
    
    df_ = df_.loc[df_['dataset'] == gse_id]
    
    df_ = (df_
        .drop(['platform', 'gene_symbol'], axis=1)
        .melt(
            id_vars=['gsm_id', 'phenotype'],
            value_vars=['transformed', 'normalized'],
            var_name='data_type',
            value_name='expression')
        .groupby(['gsm_id', 'phenotype', 'data_type'])
        .quantile(q=[0.25, 0.5, 0.75])
        .reset_index()
        .rename({'level_3': 'quantile'}, axis=1)
        .pivot(
            index=['gsm_id', 'phenotype', 'data_type'],
            columns='quantile',
            values='expression')
        .reset_index()
        .rename({0.25: 'q25', 0.5: 'q50', 0.75: 'q75'}, axis=1))
    
    df_['iqr'] = df_['q75'] - df_['q25']
    df_['wmin'] = df_['q25'] - (1.5 * df_['iqr'])
    df_['wmax'] = df_['q75'] + (1.5 * df_['iqr'])
    
    df_ = df_.sort_values('phenotype')
    df_['gsm_id'] = pd.Categorical(
        df_['gsm_id'], categories=df_['gsm_id'].unique())
    df_['data_type'] = pd.Categorical(
        df_['data_type'], categories=['transformed', 'normalized'])

    p = (
        ggplot(df_) +
        geom_pointrange(
            aes(x='gsm_id',
                y='q50',
                ymin='q25',
                ymax='q75',
                color='phenotype')) +
        facet_grid(
            'data_type ~ .',
            scales='free_y') +
        scale_color_manual(
            values=palette,
            name='Phenotype') + 
        labs(
            x='Sample ID',
            y='Expression value',
            title=f'{gse_id} expression value distributions') +
        theme_bw() +
        theme(
            axis_text_x=element_text(rotation=90),
            strip_background_y=element_text(color='white'))
    )
    
    return p

def plot_n_datasets_differentially_expressed(merged_results):
    """
    """
    df_ = merged_results.copy()
    
    dfs = []
    for (control, case), df in merged_results.items():
        df_ = df.copy().set_index('gene_symbol')
        df_ = (df_ != 0)
        df_ = (df_
            .sum(axis=1)
            .reset_index()
            .rename({0: 'n_datasets'}, axis=1))
        df_['control'] = control
        df_['case'] = case
        dfs.append(df_)
        
    df_plot = pd.concat(dfs)
    
    """
    df_ = df_.set_index(['control', 'case', 'gene_symbol'])
    df_ = (df_ != 0)
    df_ = (df_
        .sum(axis=1)
        .reset_index()
        .rename({0: 'n_datasets'}, axis=1))
    """

    plotnine.options.figure_size = (12, 6)
    p = (
        ggplot(df_plot) +
        geom_bar(
            aes(x='n_datasets'),
            fill=blue) +
        facet_wrap(
            '~ control + case',
            ncol=2) +
        scale_x_continuous(
            breaks=np.arange(0, 15)) +
        scale_y_log10() +
        labs(
            x='# of datasets',
            y='# of genes',
            title='Distribution of differentially expressed datasets') +
        theme_bw()
    )
    
    return p

def plot_edge_weight_distributions(edge_weight_distrs, null_edge_weight_distrs):
    """
    """
    dfs = []
    
    for (control, case), edge_weights in edge_weight_distrs.items():
        edge_weights_ = (pd
            .DataFrame(edge_weights.copy())
            .reset_index())
        edge_weights_['distribution'] = 'actual'
        
        null_weights_ = (pd
            .DataFrame(null_edge_weight_distrs[(control, case)].copy())
            .reset_index())
        null_weights_['distribution'] = 'null'
        
        df_ = pd.concat([edge_weights_, null_weights_])
        df_ = df_.fillna(0)
        df_['control'] = control
        df_['case'] = case
        
        dfs.append(df_)
        
    df_plot = pd.concat(dfs)
    
    df_plot['is_sig'] = df_plot['edge_weight'].abs() >= 3
    
    plotnine.options.figure_size = (14, 6)
    
    p = (
        ggplot(df_plot) +
        geom_col(
            aes(x='edge_weight',
                y='count',
                fill='distribution',
                alpha='is_sig'),
            stat='identity',
            position='dodge') +
        #geom_smooth(
        #    df_plot.loc[df_plot['distribution'] == 'actual'],
        #    aes(x='edge_weight',
        #        y='count')) +
        facet_wrap(
            '~ control + case',
            ncol=2) +
        scale_fill_manual(
            values=[orange, blue],
            name='Distribution') +
        scale_color_manual(
            values=[orange, blue]) +
        scale_alpha_manual(
            values=[0.25, 1],
            name='Used in network construction') +
        scale_x_continuous(
            breaks=np.arange(-20, 20, 2)) +
        scale_y_log10(
            breaks=np.logspace(1, 10, num=10)) +
        labs(x='Edge weight',
             y='$\log_{10}$(# of edges)',
             title='Distributions of edge weights') +
        guides(
            color=False) +
        theme_bw()
    )

    return p

def plot_node_measure_distributions(networks, measure, top_n=100):
    """
    Parameters
    ------
    nodes : :class:`pd.DataFrame`
        A dataframe as generated by the ``nodes`` entry of the ``dict``
        returned by :func:`na.construct_networks()`.
    measure : str
        A string specifying which node measure to plot the distribution of.
        One of ``{'degree', 'weighted_degree', 'eigenvector_centrality', 'mean_log_fc'}``.
    top_n : int
        The ``top_n`` nodes as measured by ``measure`` are highlighted in
        the plot.
        
    Returns
    -------
    :class:`plotnine.ggplot`
    
    Example
    -------
    >>> import tb_gene_signature_pipeline.network_analysis as na
    >>> import tb_gene_signature_pipeline.plotting as tbplt
    >>>
    >>> diff_expr_results = na.run_differential_expression_analysis(
    ...     log_transform_all_geo_data=False)
    >>> merged_results = na.merge_differential_expression_results(
    ...     diff_expr_results, adj_pval_thresh=0.05, log_fc_thresh=np.log2(1.5))
    >>> networks = na.construct_networks(merged_results)
    >>>
    >>> weighted_degrees_distribution = tbplot.plot_node_measure_distributions(
    ...     networks['nodes'], measure='weighted_degree', top_n=100)
    >>> weighted_degrees_distribution.draw()
    
    """
    dfs = []
    
    for (control, case), df in networks['nodes'].items():
        
        df_ = df.copy()
        df_ = df_[['gene_symbol', measure]]
        
        df_top_n = df_.nlargest(top_n, measure)
        df_top_n['top_n'] = True
        df_top_n = df_top_n[['gene_symbol', 'top_n']]
        
        df_ = df_.merge(df_top_n, how='left', on='gene_symbol')
        df_['top_n'] = df_['top_n'].fillna(False)
        
        df_['control'] = control
        df_['case'] = case
        
        dfs.append(df_)
        
    df_plot = pd.concat(dfs)
    
    p = (
        ggplot() +
        geom_histogram(
            df_plot.loc[df_plot['top_n']],
            aes(x=measure),
            fill=orange,
            alpha=1.0,
            bins=50) +
        geom_histogram(
            df_plot.loc[~df_plot['top_n']],
            aes(x=measure),
            fill=blue,
            alpha=0.5,
            bins=50) +
        facet_wrap(
            '~ control + case',
            scales='free_y', ncol=2) +
        labs(x='Value',
             y='# of nodes',
             title=f'{measure} distributions') +
        theme_bw()
    )
    
    return p

def plot_mean_log_fc_vs_centrality_measure(networks, measure, top_n=100):
    """
    """
    dfs = []
    
    for (control, case), df in networks['nodes'].items():
        
        df_ = df.copy()
        df_ = df_[['gene_symbol', measure, 'mean_log_fc']]
        
        df_top_n = df_.nlargest(top_n, measure)
        df_top_n['top_n'] = True
        df_top_n = df_top_n[['gene_symbol', 'top_n']]
        
        df_ = df_.merge(df_top_n, how='left', on='gene_symbol')
        df_['top_n'] = df_['top_n'].fillna(False)
        
        df_['control'] = control
        df_['case'] = case
        
        dfs.append(df_)
        
    df_plot = pd.concat(dfs)
    
    """
    df_ = nodes.copy()
    df_ = df_.loc[:, ['control', 'case', 'gene_symbol', 'mean_log_fc', measure]]
    
    top_n_ = (df_
        .groupby(['control', 'case'])
        .apply(lambda x: x.nlargest(top_n, measure))
        .reset_index(drop=True))[['control', 'case', 'gene_symbol']]
    
    top_n_['top_n'] = True
    
    df_ = df_.merge(top_n_, how='left', on=['control', 'case', 'gene_symbol'])
    df_['top_n'] = df_['top_n'].fillna(False)
    """
    p = (
        ggplot(df_plot) +
        geom_point(
            aes(x=measure,
                y='mean_log_fc',
                color='top_n')) +
        facet_wrap(
            '~ control + case',
            scales='free_x', ncol=2) +
        scale_color_manual(
            values=palette,
            name=f'Top {str(top_n)} node by {measure}') +
        labs(
            x=measure,
            y='mean log(FC)',
            title=f'Mean log(FC) vs. {measure}') +
        theme_bw()
    )
    
    return p
    
def plot_log_fc_heatmap(df, gene_set, control, case):    
    """
    """
    plt.style.use('ggplot')
    plt.rcParams['lines.linewidth']=1.0
    plt.rcParams['axes.facecolor']='1.0'
    plt.rcParams['xtick.color']='black'
    plt.rcParams['axes.grid']=False
    plt.rcParams['axes.edgecolor']='black'
    plt.rcParams['grid.color']= '1.0'
    plt.rcParams.update({'font.size': 14})
    
    df_ = df.copy()
    df_ = (df_
        .loc[df_.gene_symbol.isin(gene_set)]
        .set_index('gene_symbol'))
    
    gene_linkage = fastcluster.linkage(df_, method='ward', metric='euclidean')
    gene_order = hierarchy.leaves_list(gene_linkage)
    
    df_ordered = df_.iloc[gene_order,:]
    
    if (control == 'hc') and (case == 'atb'):
        label = 'hc-atb'
    elif (control == 'ltbi') and (case == 'atb'):
        label = 'ltbi-atb'
    elif (control == 'od') and (case == 'atb'):
        label = 'od-atb'
    elif (control == 'hc') and (case == 'ltbi'):
        label = 'hc-ltbi'
        
    width = 2.0
    
    mpl.rcParams['axes.linewidth'] = 0.35

    num_genes = float(len(gene_set))
    
    if num_genes <= 5:
        height_ratios = [15,num_genes]
        fig_height = 0.15*num_genes
    else:
        height_ratios = [35,45.0/num_genes]
        fig_height = 0.105*num_genes
    
    fig = plt.figure(figsize=(width, fig_height), dpi=300)
    
    gs = GridSpec(2, 2,
               width_ratios=[1.5,10],                
               height_ratios=height_ratios,
               wspace=0.03,
               hspace=1.5/num_genes)

    #DENDROGRAM
    ax2 = fig.add_subplot(gs[0,0], frameon=False)
    Z2 = dendrogram(Z=gene_linkage, color_threshold=0, above_threshold_color = 'k', leaf_rotation=45, no_labels = True , orientation='left', ax=ax2) # adding/removing the axes
    ax2.set_xticks([])

    #HEATMAP
    axmatrix = fig.add_subplot(gs[0,1])
    abs_max_dataset = np.max( [df_.max().max() , abs(df_.min().min())] )
    norm = mpl.colors.Normalize(vmin = -1*abs_max_dataset , vmax = abs_max_dataset) #get the normalization
    im = axmatrix.matshow(df_ordered, aspect='auto', origin='lower', cmap=plt.cm.seismic, interpolation='none', norm=norm)

    axmatrix.grid(False)
    axmatrix.tick_params(labelcolor = 'k')
    axmatrix.yaxis.set_label_position("right")
    axmatrix.yaxis.tick_right()

    gene_labels = list(df_ordered.index)
    axmatrix.set_yticks(range(0,len(gene_labels)))
    axmatrix.set_yticklabels(gene_labels, rotation='0', fontsize = 4.5, color = 'k')
    plt.tick_params(axis = "y", which = "both", left = False, right = True, color = 'k', width = 0.5)

    dataset_labels = list(df_.columns)
    axmatrix.set_xticks(range(0,len(dataset_labels)))
    axmatrix.set_xticklabels(dataset_labels, rotation='90', fontsize = 4.5, color = 'k')
    plt.tick_params(axis = "x", which = "both", bottom = False, top = True, color = 'k', width = 0.5)

    #COLORBAR
    ax_cbar = fig.add_subplot(gs[1,1])
    cb2 = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.cm.seismic, norm = norm, orientation='horizontal')
    ax_cbar.tick_params(axis = "x", which = "both", bottom = False, top = False, color = 'k', width = 0.5, labelsize = 5)
    ax_cbar.tick_params(axis='x', which='major', pad=-2)
    ax_cbar.set_xlabel(label, fontsize = 5, color = 'k')

    plt.show()
