
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter
from subprocess import check_call

with (Path.cwd().resolve().parent / 'config.yml').open('r') as f:
    config = yaml.safe_load(f)

project_dir = Path.cwd().resolve().parent
this_dir = project_dir / 'tb_gene_signature_pipeline'

output_dir = Path(config['output_directory'])
output_dir.mkdir(parents=True, exist_ok=True)

with (project_dir / 'data' / 'datasets.json').open('r') as f:
    datasets = json.load(f)
    
with (project_dir / 'data' / 'comparisons.json').open('r') as f:
    comparisons = json.load(f)

def _edge_weight_computer(m):
    w = m @ m.T
    a = w[np.triu_indices(m.shape[0], k=1)]
    c = Counter(a)
    
    df = (pd
        .DataFrame
        .from_dict(c, orient='index')
        .rename({0: 'count'}, axis=1))
    df.index.name = 'edge_weight'
    
    return df.squeeze()

def _permute_columns(m):
    idx_i = np.random.sample(m.shape).argsort(axis=0)
    idx_j = np.tile(np.arange(m.shape[1]), (m.shape[0], 1))
    
    return m[idx_i, idx_j]

def compute_edge_weight_distributions(merged_results, overwrite=False):
    """
    Given a dataframe of significant log-FC results from each comparison,
    return the distribution of edge weights used to construct the networks.
    
    Parameters
    ----------
    merged_results : :class:`pd.DataFrame`
        A gene-by-dataset dataframe of significant log fold change
        effect sizes, as produced by :func:`merge_differential_expression_results`.
    overwrite : bool
        If output files already exist and ``overwrite`` is `False`, read
        in existing files instead of re-running analysis.
        
    Return
    ------
    :class:`pd.DataFrame`
        A dataframe with edge weight counts for each comparison
        
    Example
    -------
    >>> import tb_gene_signature.network_analysis as na
    >>> diff_expr_results = na.run_differential_expression_analysis()
    >>> merged_results = tb.merge_differential_expression_results(
    ...     adj_pval_thresh=0.05, log_fc_thresh=np.log2(1.5))
    >>> edge_weight_distributions = na.compute_edge_weight_distributions(
    ...     merged_results)
    >>> edge_weight_distributions.head()
    
    """
    edge_weights_file = output_dir / 'edge_weight.pkl'
    
    if any([not edge_weights_file.exists(), overwrite]):
            
        edge_weights_dict = {}
        
        for (control, case), df in merged_results.items():
            
            df_ = df.copy()
            df_ = df_.set_index('gene_symbol')

            df_[df_ < 0] = -1
            df_[df_ == 0] = 0
            df_[df_ > 0] = 1
            df_ = df_.astype(int)
            
            edge_weights_dict[(control, case)] = _edge_weight_computer(df_.values)

        with edge_weights_file.open('wb') as f:
            pickle.dump(edge_weights_dict, f)
            
    else:
        
        with edge_weights_file.open('rb') as f:
            edge_weights_dict = pickle.load(f)
            
    return edge_weights_dict

def generate_null_edge_weight_distributions(merged_results, n_iter=1, overwrite=False):
    """
    Generate null edge weight distributions via permutations of the
    ``merged_results`` matrix.
    
    Parameters
    ----------
    merged_results : :class:`pd.DataFrame`
        A gene-by-dataset dataframe of significant log fold change
        effect sizes, as produced by :func:`merge_differential_expression_results`.
    n_iter : int
        The number of iterations to perform in the null edge weight sampling process.
    overwrite : bool
        If output files already exist and ``overwrite`` is `False`, read
        in existing files instead of re-running analysis.
        
    Return
    ------
    :class:`pd.DataFrame`
        A dataframe with edge weight counts for each comparison
        
    Example
    -------
    >>> import tb_gene_signature.network_analysis as na
    >>> diff_expr_results = na.run_differential_expression_analysis()
    >>> merged_results = tb.merge_differential_expression_results(
    ...     adj_pval_thresh=0.05, log_fc_thresh=np.log2(1.5))
    >>> null_distributions = na.generate_null_edge_weight_distributions(
    ...     merged_results, n_iter=25)
    >>> null_distributions.head()
    """
    null_distributions_file = output_dir / 'edge_weight_null_distributions.pkl'
    
    if any([not null_distributions_file.exists(), overwrite]):
        
        null_distributions = {}
        
        for (control, case), df in merged_results.items():
            df_ = df.set_index('gene_symbol')
        
            df_[df_ < 0] = -1
            df_[df_ == 0] = 0
            df_[df_ > 0] = 1
            df_ = df_.astype(int)
            
            df_ = pd.concat([
                pd.DataFrame(_edge_weight_computer(_permute_columns(df_.values)))
                for _ in range(n_iter)], axis=1)
            
            srs = (df_
                .fillna(0)
                .sum(axis=1)
                .astype(int))
            
            srs.name = 'count'
            
            null_distributions[(control, case)] = srs
            
        with null_distributions_file.open('wb') as f:
            pickle.dump(null_distributions, f)
    
    else:
        
        with null_distributions_file.open('rb') as f:
            null_distributions = pickle.load(f)
        
    return null_distributions
         
# notebook 1
def run_differential_expression_analysis(overwrite=False, log_transform_all_geo_data=False,
                                         quantile_normalize_per_class=False, salmon_or_rsem='salmon'):
    """
    Run marginal, per-gene differential expression analyses.
    
    This function calls the differential analysis script
    ``tb-gene-signature-pipeline/R/differential_expression_analysis.R``.
    
    An independent case/control differential expression analysis is run for
    each dataset defined in ``data/datasets.json``, and for each group comparison
    defined in ``data/comparisons.json``.
    
    Below, `<output_dir>` refers to the path specified by the `output_directory` in the
    project config file (`confi.yml`).
    
    All transformed expression values for each dataset are written to files
    `<output_dir>/transformed-expression-matrices/<gse_id>.<dataset_platform>.transformed_expr_matrix.tsv`.
    
    All normalized expression values for each dataset are written to files
    `<output_dir>/normalized-expression-matrices/<gse_id>.<dataset_platform>.normalized_expr_matrix.tsv`.
    
    Combined transformed and expression values for all datasets are written to a single file
    `<output_dir>/differential_expression_values.tsv`.
    
    All differential expression results for each dataset are written to files
    `<output_dir>/differential-expression-results/<gse_id>.<control>_vs_<case>.diff_expr_results.tsv`.
    
    Combined differential expression results for all datasets are written to a single file
    `<output_dir>/differential_expression_results.tsv`.
    
    Parameters
    ----------
    overwrite : bool
        If output files already exist and ``overwrite`` is `False`, read
        in existing files instead of re-running analysis.
    log_transform_all_geo_data : bool
        If `True`, log-transform all microarray data downloaded from GEO,
        regardless of whether or not it's already been transformed. If `False`,
        only log-transform dataset if it is not already log-transformed.
    quantile_normalize_per_class : bool
        If `True`, quantile-normalize each log-transformed microarray dataset
        separately within each phenotype class. If `False`, quantile-normalize all
        samples together.
    salmon_or_rsem : str
        One of ``{'salmon', 'rsem'}``. Specifies which processed bulk RNA-seq expression
        matrices to use in differential expression analysis.
    
    Returns
    -------
    dict of :class:`pd.DataFrame`
        A dictionary with dataframes `exprs` and `results`.
        
        `exprs` contains preprocessed and normalized expression values
        for each sample and across datasets.
        
        `results` contains differential expression effect sizes and p-values
        for each dataset and case/control comparison.  

    Notes
    -----
    For each GSE microarray experiment, the gene expression data pulled from GEO
    is :math:`\log_2` transformed, if the GEO data is not already :math:`\log`
    transformed. The transformed data is then quantile-normalized
    within phenotype group* [1]_.
    
    References
    ----------
    .. [1] Zhao, Y., Wong, L. & Goh, W.W.B. How to do quantile normalization correctly for gene 
       expression data analyses. Sci Rep 10, 15534 (2020). 
       https://doi.org/10.1038/s41598-020-72664-6
    
    Examples
    --------
    >>> import tb_gene_signature_pipeline as tb
    >>> diff_expr_results = tb.run_differential_expression_analysis()
    >>> diff_expr_results['exprs'].head()
             gene_symbol      gsm_id  preprocessed  normalized phenotype    dataset platform
    0               A1BG  GSM2712676        71.001    0.738524       atb  GSE101705   rnaseq
    1               A1BG  GSM2712677        65.999    0.947318       atb  GSE101705   rnaseq
    2               A1BG  GSM2712678        43.000    0.269351       atb  GSE101705   rnaseq
    3               A1BG  GSM2712679        41.000    0.310476       atb  GSE101705   rnaseq
    4               A1BG  GSM2712680        60.000    0.830032       atb  GSE101705   rnaseq
    >>> diff_expr_results['results'].head()
        dataset control case gene_symbol    log_fc  adj_p_val
    0  GSE19439      hc  atb        A1BG  0.026045   0.917217
    1  GSE19439      hc  atb        A1CF  0.091435   0.468395
    2  GSE19439      hc  atb         A2M  0.007356   0.964635
    3  GSE19439      hc  atb       A2ML1 -0.171059   0.204401
    4  GSE19439      hc  atb     A3GALT2 -0.008727   0.967318
    """
    transformed_data_dir = output_dir / 'transformed-expression-matrices'
    transformed_data_dir.mkdir(parents=True, exist_ok=True)

    normalized_data_dir = output_dir / 'normalized-expression-matrices'
    normalized_data_dir.mkdir(parents=True, exist_ok=True)

    diff_exp_results_dir = output_dir / 'differential-expression-results'
    diff_exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    exprs_file = output_dir / 'differential_expression_values.tsv'
    results_file = output_dir / 'differential_expression_results.tsv'
    
    if any([not exprs_file.exists(), not results_file.exists(), overwrite]):
        r_script = this_dir / 'R' / 'differential_expression_analysis.R'
        check_call([
            r_script, str(log_transform_all_geo_data),
            str(quantile_normalize_per_class), salmon_or_rsem])

    df_exprs = pd.read_table(exprs_file, sep='\t')
    df_results = pd.read_table(results_file, sep='\t')
    
    return {
        'exprs': df_exprs,
        'results': df_results
    }

# notebook 2
def merge_differential_expression_results(
    differential_expression_df, adj_pval_thresh=0.05,
        log_fc_thresh=np.log2(1.5)):
    """
    Given a :class:`pd.DataFrame` of differential expression results
    (as generated by :func:`run_differential_expression_analysis`), return
    significant log fold change results.
    
    Parameters
    ----------
    differential_expression_df : :class:`pd.DataFrame`
        A dataframe of differential expression results, as generated by
        :func:`run_differential_expression_analysis()`.
    pval_thresh : float
        Adjusted p-value threshold for inclusion in merged dataframe.
    log_fc_thresh : float
        Effect size threshold for inclusion in merged dataframe.
        
    Return
    ------
    :class:`pd.DataFrame`
        A ``gene``-by-``dataset`` dataframe of log fold change effect sizes
        with the columns:
        
            * Name: control, dtype: object
            * Name: case, dtype: object
            * Name: gene_symbol, dtype: object
            * Name: ``<dataset GSE ID>``, dtype: float (one column per dataset)
        
    Example
    -------
    >>> import tb_gene_signature_pipeline as tb
    >>> df_results = tb.run_differential_expression_analysis()
    >>> merged_results = tb.merge_differential_expression_results(
    ...     adj_pval_thresh=0.05, log_fc_thresh=np.log2(1.5))
    >>> merged_results.head()
      control case gene_symbol  GSE107994  GSE29536  GSE34608  GSE42825  ...
    0      hc  atb   1060P11.3        0.0       0.0  0.000000  0.000000  ...
    1      hc  atb     A2M-AS1        0.0       0.0  0.000000  0.000000  ...
    2      hc  atb       AAED1        0.0       0.0  0.862417  0.000000  ...
    3      hc  atb       AAMDC        0.0       0.0  0.000000  0.694117  ...
    4      hc  atb        AAMP        0.0       0.0  0.000000  0.000000  ...
    """
    df = differential_expression_df.copy()
    
    df['is_sig'] = df.apply(
        lambda x: (np.abs(x['log_fc']) >= log_fc_thresh) & (x['adj_p_val'] <= adj_pval_thresh),
        axis=1)
    df['log_fc'] = df.apply(lambda x: x['log_fc'] if x['is_sig'] else 0.0, axis=1)
    
    dfs = {}
    
    for comparison in comparisons:
        
        control = comparison['control']
        case = comparison['case']
        
        df_ = (df
            .loc[
                (df['control'] == control) &
                (df['case'] == case)]
            .pivot(
                index='gene_symbol', columns='dataset', values='log_fc')
            .fillna(0.)
            .reset_index())
        
        dfs[(control, case)] = df_
    
    """
    df = (df
        .groupby(['control', 'case'])
        .apply(
            lambda x: x.pivot(
                index='gene_symbol', columns='dataset', values='log_fc'))
        .fillna(0.)
        .reset_index())
    """
    
    return dfs

# notebook 5
def construct_networks(merged_results, overwrite=False):
    """
    Construct networks based on shared association signal across
    differential expression analysis datasets. 
    
    One network is created for each case/control comparison
    (``hc`` vs. ``atb``, etc.).
    
    Within each network, a node represents a gene, and an edge between
    nodes represents indicates that those two genes had significant
    differential expression associations in the **same** direction in
    **at least 3** datasets.
    
    Pickled network graphs for each comparison network are written to
    `<output_dir>/network_graphs.pkl`.
    
    Measures for each node in all comparison networks are written to a single
    file `<output_dir>/network_nodes.tsv`.
    
    Parameters
    ----------
    merged_results : dict of :class:`pd.DataFrame`
        A dictionary of gene-by-dataset dataframes of significant log fold change
        effect sizes, as produced by :func:`merge_differential_expression_results`.
    overwrite : bool
        If output files already exist and ``overwrite`` is `False`, read
        in existing files instead of re-running analysis.
        
    Return
    ------
    dict
        A dictionary with the entries:
            graphs : :class:`pd.DataFrame`
                Columns:
                    * Name: control, dtype: object
                    * Name: case, dtype: object
                    * Name: graph, dtype: :class:`nx.Graph`
            nodes : :class:`pd.DataFrame`
                Columns:
                    * Name: control, dtype: object
                    * Name: case, dtype: object
                    * Name: gene_symbol, dtype: object
                    * Name: degree, dtype: float
                    * Name: weighted_degree, dtype: float
                    * Name: eigenvector_centrality, dtype: float
    
    Example
    -------
    >>> import tb_gene_signature_pipeline as tb
    >>> differential_expression_results = tb.run_differential_expression_analysis()
    >>> merged_results = tb.merge_differential_expression_results(
    ...     differential_expression_results, adj_val_thresh=0.05,
    ...     log_fc_thresh=np.log2(1.5))
    >>> networks = tb.construct_networks(merged_results)
    >>> print(networks['graphs'])
      control  case                                              graph
    0      hc   atb  (AAMDC, ABCA1, ACOT8, ACOT9, ACSL1, ACTA2, ADA...
    1      hc  ltbi                                     (ATP1B2, ETV7)
    2    ltbi   atb  (ABCA1, ABCA13, ABCC13, ACSL1, ACSL4, ADAM9, A...
    3      od   atb  (ADM, AIM2, ANKRD22, APOL6, ATF3, BATF2, BRSK1...
    >>> print(networks['nodes'])
         control case gene_symbol  degree  weighted_degree  eigenvector_centrality
    0         hc  atb       AAMDC     503        94.312500            1.404467e-02
    1         hc  atb       ABCA1    1318       349.187500            4.088542e-02
    2         hc  atb       ACOT8     503        94.312500            1.404467e-02
    3         hc  atb       ACOT9    1099       271.000000            3.351424e-02
    4         hc  atb       ACSL1    1327       354.187500            4.143864e-02
    ...      ...  ...         ...     ...              ...                     ...
    3409      od  atb        RGL1       5         1.666667            1.456176e-03
    3410      od  atb        TLR5       5         1.666667            1.456176e-03
    3411      od  atb        CD3G       4         1.333333            1.917126e-11
    3412      od  atb        GNLY       4         1.333333            1.917126e-11
    3413      od  atb        NRG1       4         1.333333            1.917126e-11
    
    """
    networks_file = output_dir / 'networks.pkl'
    
    if any([not networks_file.exists(), overwrite]):

        def generate_network_per_comparison(xdf):
            xdf_ = xdf.copy().reset_index()

            #m = xdf_.drop(['control', 'case', 'gene_symbol'], axis=1).values
            m = xdf_.drop('gene_symbol', axis=1).values
            n_datasets = (m.sum(axis=0) > 0).sum()

            m[m > 0] = 1
            m[m == 0] = 0
            m[m < 0] = -1

            edge_weights = m.dot(m.T)
            edge_weights_mask = np.abs(edge_weights) >= 3

            edge_indices = zip(*np.triu(edge_weights_mask).nonzero())
            gene_symbols = xdf_['gene_symbol']

            network_edge_list = [
                (gene_symbols[i], gene_symbols[j], float(edge_weights[i,j])/n_datasets)
                for i, j in edge_indices]

            graph = nx.Graph()
            graph.add_weighted_edges_from(network_edge_list)

            return graph
        
        graphs = {}
        nodes = {}
        
        for comparison, df in merged_results.items():
            control = comparison[0]
            case = comparison[1]
            
            graph = generate_network_per_comparison(df)
            
            degrees = (pd
                .DataFrame(graph.degree(weight=None))
                .rename({0: 'gene_symbol', 1: 'degree'}, axis=1))
            
            eigens = (pd
                .DataFrame([
                    (k, v) for k, v in 
                    nx.eigenvector_centrality(graph, weight='weight').items()])
                .rename({0: 'gene_symbol', 1: 'eigenvector_centrality'}, axis=1))
            
            weighted_degrees = (pd
                .DataFrame(graph.degree(weight='weight'))
                .rename({0: 'gene_symbol', 1: 'weighted_degree'}, axis=1))
            
            mean_log_fc = (df
                .copy()
                .set_index('gene_symbol')
                .mean(axis=1)
                .reset_index()
                .rename({0: 'mean_log_fc'}, axis=1))
            
            df_nodes = (degrees
                .merge(eigens, how='inner', on='gene_symbol')
                .merge(weighted_degrees, how='inner', on='gene_symbol')
                .merge(mean_log_fc, how='inner', on='gene_symbol'))

            graphs[(control, case)] = graph
            nodes[(control, case)] = df_nodes
            
        networks = {
            'graphs': graphs,
            'nodes': nodes
        }
        
        with networks_file.open('wb') as f:
            pickle.dump(networks, f)
            
    else:
        
        with networks_file.open('rb') as f:
            networks = pickle.load(f)
            
    return networks
        
def generate_gene_lists(networks, top_n=100):
    """
    Generate gene lists for each comparison by returning the
    top nodes in each comparison network by ``weighted_degree``.
    
    Parameters
    ----------
    nodes : :class:`pd.DataFrame`
        A dataframe of networks node metrics, as returned by the ``nodes``
        entry of :func:`generate_networks()`.
    top_n : int
        Return the ``top_n`` nodes in each comparison network.
        
    Return
    ------
    dict
        A dictionary of gene lists, keyed by comparison (e.g. ``'hc-atb'``.)
        
    Examples
    --------
    >>> import tb_gene_signature_pipeline.network_analysis as tb
    >>> diff_expr_results = na.run_differential_expression_analysis(
    ...     log_transform_all_geo_data=False)
    >>> merged_results = tb.merge_differential_expression_results(
    ...     differential_expression_results, adj_val_thresh=0.05,
    ...     log_fc_thresh=np.log2(1.5))
    >>> networks = tb.construct_networks(merged_results)
    >>> gene_lists = tb.generate_gene_lists(networks['nodes'], top_n=100)
        
    """
    top_nodes_dict = {}
    
    for (control, case), df in networks['nodes'].items():
        top_nodes = df.nlargest(top_n, 'weighted_degree')['gene_symbol']
        top_nodes_dict[(control, case)] = list(top_nodes)
        
    return top_nodes_dict
    
# notebook 6
def combine_networks_into_lists(networks, n_nodes=100):
    """
    Generate gene lists based on intersections and differences between the
    top ``n_nodes`` nodes in each comparison network (``hc`` vs. ``atb``,
    ``od`` vs. ``atb``, and `ltbi`` vs. ``atb`` - exclude the almost
    nonexistent ``hc`` vs. ``ltbi`` network).
    
    Parameters
    ----------
    graphs : dict
        A dictionary of network graphs and node metrics, as returned by
        :func:`construct_networks`.
    n_nodes : int
        Gene lists will be generated based on the top ``n_nodes`` nodes
        by weighted degree in each comparison network.
        
    Return
    ------
    dict
        A dictionary of gene lists with two entries:
        
        ``'top_genes_in_all_networks'``: The intersection of the top
            ``n_nodes`` genes by weighted degree in the ``hc`` vs. ``atb``,
            ``ltbi`` vs. ``atb``, and ``od`` vs. ``atb`` networks.
        
        ``'top_genes_not_in_od_network'``: The intersection of the top 
            ``n_nodes`` genes by weighted degree in the ``hc`` vs. ``atb``
            and ``ltbi`` vs. ``atb`` networks, minus the genes in the top
            ``n_nodes`` of the ``od`` vs. ``atb`` network.

    Example
    -------
    >>> import tb_gene_signature_pipeline as tb
    >>> differential_expression_results = tb.run_differential_expression_analysis()
    >>> merged_results = tb.merge_differential_expression_results(
    ...     differential_expression_results, adj_val_thresh=0.05,
    ...     log_fc_thresh=np.log2(1.5))
    >>> networks = tb.construct_networks(merged_results)
    >>> gene_lists = tb.combine_networks_into_lists(networks, n_nodes=100)
    >>> print(gene_lists['top_genes_in_all_networks'][:5])
    ['SAMD9L', 'KCNJ15', 'LAP3', 'CEACAM1', 'SERPING1']
    >>> print(gene_lists['top_genes_not_in_od_network'][:5])
    ['TLR5', 'JAK2', 'S100A12', 'ZNF438', 'SORT1']
    
    """
    
    df_ = networks['nodes'].copy()
    df_ = df_.loc[~((df_['control'] == 'hc') & (df_['case'] == 'ltbi'))]
    
    top_nodes = (df_
        .set_index(['control', 'case'])
        .sort_values('weighted_degree', ascending=False)
        .groupby(['control', 'case'])
        .head(n_nodes)
        .reset_index())[['control', 'case', 'gene_symbol']]
    
    hc_atb_nodes = set(top_nodes.loc[
        (top_nodes['control'] == 'hc') & (top_nodes['case'] == 'atb')]['gene_symbol'])
    
    ltbi_atb_nodes = set(top_nodes.loc[
        (top_nodes['control'] == 'ltbi') & (top_nodes['case'] == 'atb')]['gene_symbol'])
    
    od_atb_nodes = set(top_nodes.loc[
        (top_nodes['control'] == 'od') & (top_nodes['case'] == 'atb')]['gene_symbol'])
    
    top_genes_all_networks = hc_atb_nodes.intersection(ltbi_atb_nodes, od_atb_nodes)
    top_genes_not_od_network = hc_atb_nodes.intersection(ltbi_atb_nodes) - od_atb_nodes
    
    return {
        'top_genes_in_all_networks': list(top_genes_all_networks),
        'top_genes_not_in_od_network': list(top_genes_not_od_network)
    }
    