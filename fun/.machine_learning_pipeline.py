
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from joblib import Parallel, delayed
from plotnine import *
from mizani.formatters import comma_format
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, SelectFromModel, \
    mutual_info_classif, mutual_info_regression
from sklearn.compose import ColumnTransformer, \
    make_column_selector, make_column_transformer
from sklearn.utils import resample
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression, \
    ElasticNet, SGDClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, auc

colors = dict(
    blue = '#4E79A7',
    orange = '#F28E2B',
    green = '#59A14F',
    red = '#E15759',
    purple = '#B07AA1',
    pink = '#FF9DA7',
    gray = '#BAB0AC',
    yellow = '#EDC948',
    brown = '#9D7660',
    teal = '#499894',
    lightblue = '#A0CBE8'
)
palette = colors.keys()

def fit_bootstrap_sample(
    df_X, srs_y, pipeline, model_name, lambda_name,
        lambda_val, n_samples_in_bootstrap):
    """
    """
    df_X_, srs_y_ = resample(
        df_X.values, srs_y, n_samples=n_samples_in_bootstrap)
    
    pipeline.set_params(
        **{f'{model_name}__{lambda_name}': lambda_val})
    pipeline.fit(df_X_, srs_y_)
    
    coefs = pipeline.named_steps[model_name].coef_
    
    # if numeric outcome, len(coefs)==1; if categorical len(coefs)=# categories
    if len(coefs.shape) == 1:
        coefs = coefs.reshape((1, -1))

    # trickier when categorical
    bootstrap_mask = np.any(coefs != 0.0, axis=0)

    return bootstrap_mask
    
def compute_feature_stability_scores(
    df_X, srs_y, method='max', threshold=0.4, lambda_grid=np.logspace(-4, 0, num=25),
        n_bootstraps=100, n_samples_in_bootstrap=None, l1_ratio=1.0, random_state=None):
    """
    """
    if n_samples_in_bootstrap is None:
        n_samples_in_bootstrap = int(df_X.shape[0]/2)
        
    model_name = 'elasticnet'
    lambda_name = 'alpha'
        
    pipeline = Pipeline([
        ('imputer',
            KNNImputer(
                n_neighbors=5, weights='uniform')),
        ('scaler',
            StandardScaler()),
        (model_name, 
            ElasticNet(
                l1_ratio=1.0, max_iter=1e5,
                random_state=random_state))
    ])
    
    stability_scores_list = []
    
    for lambda_idx, lambda_val in enumerate(lambda_grid):
        bootstrap_mask_list = Parallel(n_jobs=-1)(
            delayed(fit_bootstrap_sample)(
                df_X, srs_y, pipeline, model_name, lambda_name,
                lambda_val, n_samples_in_bootstrap)
            for _ in range(n_bootstraps))
        
        bootstrap_mask_matrix = np.hstack([
            x.reshape((-1, 1)) for x in bootstrap_mask_list])

        stability_scores = bootstrap_mask_matrix.mean(axis=1)
        stability_scores_list.append(stability_scores)
        
    stability_scores_matrix = np.hstack([
        x.reshape((-1, 1)) for x in stability_scores_list])

    stability_scores_df = pd.DataFrame(
        stability_scores_matrix, index=df_X.columns, columns=lambda_grid)
    
    stability_scores_df = stability_scores_df.reset_index()
    stability_scores_df = stability_scores_df.rename({'index': 'feature'}, axis=1)
    
    stability_scores_df = stability_scores_df.melt(
        id_vars='feature', var_name='lambda', value_name='stability_score')

    stability_scores_df['lambda'] = stability_scores_df['lambda'].astype(float)
    stability_scores_df = stability_scores_df.sort_values(['lambda', 'feature'])
    stability_scores_df = stability_scores_df.set_index('feature')
    
    group = stability_scores_df.groupby(stability_scores_df.index)

    if method == 'max':
        stability_scores = pd.DataFrame(group['stability_score'].max())

    elif method == 'auc':
        stability_scores = pd.DataFrame(
            group[['lambda', 'stability_score']].apply(
                lambda x: auc(x['lambda'], x['stability_score'])))
        stability_scores = stability_scores.rename({0: 'stability_score'}, axis=1)
        
    stability_scores['is_stable'] = stability_scores['stability_score'] >= threshold
    stable_features = list(
        stability_scores['is_stable'].index[stability_scores['is_stable']])
    
    return {
        'scores': stability_scores,
        'lambda_path': stability_scores_df,
        'stable_features': stable_features
    }

def fit_inner_cv_regressor_model(X, y, preprocessor, regressor, params, cv):
    """
    """
    pipeline = clone(preprocessor)
    pipeline.steps.append(('regressor', regressor))
    
    param_grid = {f'regressor__{k}': v for k, v in params.items()}

    regression_model = GridSearchCV(
        estimator=pipeline, scoring='neg_root_mean_squared_error',
        n_jobs=-1, refit=True, param_grid=param_grid, cv=cv)
    
    regression_model.fit(X, y)
    
    return regression_model

class GeneSignatureDataset(object):
    
    def __init__(self, X, y, random_state=None):
        """A class holding data structures and methods used
        in a machine learning pipeline that uses gene expression
        data to predict phenotypic outcomes.
        
        # extended summary
        
        Parameters
        ----------
        X : `pd.DataFrame`
            A feature matrix of dimension (n_samples x n_features).
            Can contain gene expression features, as well as other
            potential predictors such as age and sex.
        y: `pd.Series`
            A vector of either `float` or `object` 
        """
        categorical_features = X.select_dtypes(
            include=object).columns
        
        self.X = pd.get_dummies(X)
        self.y = y
        self.random_state = random_state
        
        self.numeric_features = X.select_dtypes(
            include='number').columns
        self.categorical_features = X.columns[
            X.columns.map(
                lambda x: x.split('_')[0] in categorical_features)]
        
        self.features = self.numeric_features.union(
            self.categorical_features)
        
        self.is_regression = is_numeric_dtype(y)
        
        if self.is_regression:
            self.scorer = mutual_info_regression
        else:
            self.scorer = mutual_info_classif

    def _check_attribute(self, attribute, method):
        """
        """
        try:
            val = getattr(self, attribute)
        except:
            raise AttributeError(
                ' '.join([
                    f"'{attribute}' attribute is not set.",
                    f"Run '{method}()' method first."
                ])
            )
            
        return val
        
    def train_test_split(self, test_size=0.25):
        """
        """
        stratify = None if self.is_regression else self.y
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size,
            random_state=self.random_state)
        
        train_dataset = GeneSignatureDataset(
            X_train, y_train, random_state=self.random_state)
        test_dataset = GeneSignatureDataset(
            X_test, y_test, random_state=self.random_state)
        
        return train_dataset, test_dataset
    
    def compute_featurewise_outcome_correlations(self, method='pearson'):
        """
        """
        corr = self.X.apply(
            lambda feature: feature.corr(self.y, method=method))
        corr.name = f'{method}_correlations'
        
        self._featurewise_outcome_correlations = corr
        
        return corr
    
    def plot_feature_stability_scores_vs_outcome_correlations(
            self, title='', plot_corr_method='pearson'):
        """
        """
        correlations = self._check_attribute(
            '_featurewise_outcome_correlations',
            'compute_featurewise_outcome_correlations')
        
        stability_scores = self._check_attribute(
            '_feature_stability_scores',
            'compute_feature_stability_scores')
        
        df = pd.DataFrame({
            'correlations': correlations.abs(),
            'stability_scores': stability_scores
        })
        
        df = df.dropna()
        
        regression = LinearRegression().fit(
            df['correlations'].values.reshape(-1, 1),
            df['stability_scores'].values)
        
        intercept = regression.intercept_
        slope = regression.coef_[0]
        
        df['fitted'] = df['correlations']*slope + intercept
        
        r2 = r2_score(df['stability_scores'], df['fitted'])
        
        corr_method = correlations.name.split("_")[0]
        
        p = (
            ggplot(df) +
            geom_point(
                aes(x='correlations',
                    y='stability_scores'),
                color=colors['gray'],
                alpha=0.5) +
            geom_line(
                aes(x='correlations',
                    y='fitted'),
                linetype='dashed',
                color=colors['red'],
                size=1.5) +
            annotate(
                'text', x=0.1, y=0.9,
                label=f'$r^2 = {{{round(r2, 2)}}}$') + 
            scale_x_continuous(
                breaks=np.arange(0, 1.1, 0.1)) +
            scale_y_continuous(
                breaks=np.arange(0, 1.1, 0.1)) +
            coord_cartesian(
                xlim=[0, 1],
                ylim=[0, 1]) +
            labs(
                x=f'$|\\rho_{{{corr_method}}}|$',
                y='stability score',
                title=title) +
            theme_bw()
        )
        
        return p
    
    def filter_dataset(self, sample_mask=None, feature_mask=None):
        """
        """
        X_filter = self.X.copy()
        y_filter = self.y.copy()
        
        if sample_mask is not None:
            X_filter = X_filter.iloc[sample_mask,:]
            y_filter = y_filter.iloc[sample_mask]
            
        if feature_mask is not None:
            X_filter = X_filter.iloc[:,feature_mask]
            
        filtered_pipeline = GeneSignatureMLPipeline(
            X_filter, y_filter, random_state=self.random_state)
        
        return filtered_pipeline
    
    def preprocess_features(self):
        """
        """
        # user-defined parameters
        impute_strategy = 'knn'
        n_neighbors = 5
        weights = 'uniform'
        
        if impute_strategy == 'knn':
            imputer = KNNImputer(
                n_neighbors=n_neighbors, weights=weights)
            
        one_hot_encoder = OneHotEncoder()
        scaler = StandardScaler()
        
        numeric_pipeline = make_pipeline(imputer, scaler)
        categorical_pipeline = make_pipeline(one_hot_encoder, imputer)
        
        preprocessor = make_column_transformer(
            (numeric_pipeline, self.numeric_features),
            (categorical_pipeline, self.categorical_features),
            remainder='drop')
        
        return preprocessor
    
    def lasso_model(self):
        """
        """
        if self.is_regression:
            model = Lasso(max_iter=1e5, random_state=self.random_state)
            lambda_name = 'alpha'
        else:
            model = LogisticRegression(
                penalty='l1', multi_class='auto', max_iter=1e5,
                solver='saga', random_state=self.random_state, n_jobs=-1)
            lambda_name = 'C'
        
        return model, lambda_name
    
    def fit_bootstrap_lasso(
        self, pipeline, lambda_name, lambda_val,
            n_samples=None, with_replacement=True):
        """
        """
        stratify = None if self.is_regression else self.y
        
        if n_samples is None:
            n_samples = int(self.X.shape[0]/2)

        X_bootstrap, y_bootstrap = resample(
            self.X, self.y, replace=with_replacement,
            n_samples=n_samples, stratify=stratify)
        
        pipeline.set_params(**{f'lasso_model__{lambda_name}': lambda_val})
        pipeline.fit(X_bootstrap, y_bootstrap)
        
        coefs = pipeline.named_steps.lasso_model.coef_

        if len(coefs.shape) == 1:
            coefs = coefs.reshape((1, -1))
            
        bootstrap_mask = np.any(coefs != 0.0, axis=0)
        
        return bootstrap_mask
    
    def elasticnet_model(self, l1_ratio=1.0):
        """
        """
        if self.is_regression:
            model = ElasticNet(
                l1_ratio=l1_ratio, max_iter=1e5, 
                random_state=self.random_state)
            lambda_name = 'alpha'
        else:
            model = SGDClassifier(
                loss='log', penalty='elasticnet', l1_ratio=l1_ratio,
                max_iter=1e5, n_jobs=-1)
            lambda_name = 'alpha'
            
        return model, lambda_name
    
    def fit_bootstrap_elasticnet(
        self, pipeline, lambda_name, lambda_val,
            n_samples=None, l1_ratio=0.5, with_replacement=True):
        """
        """
        stratify = None if self.is_regression else self.y
        
        if n_samples is None:
            n_samples = int(self.X.shape[0]/2)

        X_bootstrap, y_bootstrap = resample(
            self.X, self.y, replace=with_replacement,
            n_samples=n_samples, stratify=stratify)
        
        pipeline.set_params(**{f'model__{lambda_name}': lambda_val})
        pipeline.fit(X_bootstrap, y_bootstrap)
        
        coefs = pipeline.named_steps.model.coef_

        if len(coefs.shape) == 1:
            coefs = coefs.reshape((1, -1))
            
        bootstrap_mask = np.any(coefs != 0.0, axis=0)
        
        return bootstrap_mask
    
    def compute_feature_stability_scores(
        self, method, lambda_grid=np.logspace(-4, 0, num=25), n_bootstraps=100,
            n_samples=None, l1_ratio=1.0, compute_lambda_path=True):
        """
        """
        if compute_lambda_path:
            
            if n_samples is None:
                int(self.X.shape[0]/2)

            preprocessor = self.preprocess_features()
            model, lambda_name = self.elasticnet_model()

            pipeline = Pipeline([
                ('preprocess_features', preprocessor),
                ('model', model)
            ])

            stability_scores_list = []
            for lambda_idx, lambda_val in enumerate(lambda_grid):

                bootstrap_mask_list = Parallel(n_jobs=-1)(
                    delayed(self.fit_bootstrap_elasticnet)(
                        clone(pipeline),
                        lambda_name,
                        lambda_val,
                        n_samples,
                        l1_ratio)
                    for _ in range(n_bootstraps))
                bootstrap_mask_matrix = np.hstack([
                    x.reshape((-1, 1)) for x in bootstrap_mask_list])

                stability_scores = bootstrap_mask_matrix.mean(axis=1)
                stability_scores_list.append(stability_scores)

            stability_scores_matrix = np.hstack([
                x.reshape((-1, 1)) for x in stability_scores_list])

            stability_scores_df = pd.DataFrame(
                stability_scores_matrix, index=self.features, columns=lambda_grid)

            stability_scores_df_long = (
                stability_scores_df
                .reset_index()
                .rename({'index': 'feature'}, axis=1)
                .melt(
                    id_vars='feature',
                    var_name='lambda',
                    value_name='stability_score')
            )

            stability_scores_df_long['lambda'] = (
                stability_scores_df_long['lambda'].astype(float))

            stability_scores_df_long = stability_scores_df_long.sort_values('lambda')
            
        else:
            stability_scores_df_long = self._check_attribute(
                '_feature_stability_scores_lambda_path',
                'compute_feature_stability_scores')
        
        group = stability_scores_df_long.groupby('feature')
        
        if method == 'max':
            stability_scores = group['stability_score'].max()
            
        elif method == 'auc':
            stability_scores = group[['lambda', 'stability_score']].apply(
                lambda x: auc(x['lambda'], x['stability_score']))
        
        self._feature_stability_scores_statistic = method
        self._feature_stability_scores_lambda_path = stability_scores_df_long
        self._feature_stability_scores = stability_scores
        
        return stability_scores
    
    def plot_feature_stability_scores_lambda_path(self, method, threshold=0.5, title=''):
        """
        """
        df = self._check_attribute(
            '_feature_stability_scores_lambda_path',
            'compute_feature_stability_scores')
        
        stability_statistic = self._check_attribute(
            '_feature_stability_scores_statistic',
            'compute_feature_stability_scores')
            
        if threshold is None:
            threshold = 0.0
            
        group = df.groupby('feature')
            
        if method == 'max':
            stability_scores = group['stability_score'].max()
            
            is_stable = (
                (stability_scores >= threshold)
                .reset_index()
                .rename({'stability_score': 'is_stable'}, axis=1))

        elif method == 'auc':
            stability_scores = group[['lambda', 'stability_score']].apply(
                lambda x: auc(x['lambda'], x['stability_score']))
            
            is_stable = (
                (stability_scores >= threshold)
                .reset_index()
                .rename({0: 'is_stable'}, axis=1))
        
        df = df.merge(is_stable, how='left', on='feature')
        df['lambda_neg_log10'] = -np.log10(df['lambda'])
        
        p = (
            ggplot() +
            geom_line(
                df,
                aes(x='lambda_neg_log10',
                    y='stability_score',
                    group='feature',
                    color='is_stable',
                    alpha='is_stable'),
                linetype='dashed') +
            scale_x_continuous(
                breaks=np.arange(0, 5, 1)) +
            scale_y_continuous(
                breaks=np.arange(0, 1.1, 0.1)) +
            scale_color_manual(
                values=[colors['gray'], colors['red']]) +
            scale_alpha_manual(
                values=[0.5, 1.0]) +
            coord_cartesian(
                ylim=[0, 1]) +
            labs(
                x='$-log_{10}(\lambda)$',
                y='stability score',
                title=title) +
            theme_bw()
        )
        
        return p
        
        
    def plot_feature_stability_scores_distribution(self, title=''):
        """
        """
        stability_scores = self._check_attribute(
            '_feature_stability_scores',
            'compute_feature_stability_scores')
        
        p = (
            ggplot(
                pd.DataFrame({
                    'stability_scores': stability_scores
                })) +
            geom_histogram(
                aes(x='stability_scores'),
                binwidth=0.05,
                fill=colors['blue'],
                color=colors['blue'],
                alpha=0.5) +
            scale_x_continuous(
                breaks=np.arange(0, 1.1, 0.1)) +
            scale_y_continuous(
                breaks=np.arange(0, 25000, 1000),
                labels=comma_format()) +
            labs(
                x='stability score',
                y='# of features',
                title=title) +
            theme_bw()
        )
        
        return p
    