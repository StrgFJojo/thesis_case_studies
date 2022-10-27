import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
Contains all statistic utilities used for the three case studies.
Mainly:
- DataManager: Handles data transformations in preparation for regression
- RegModelManager:  Takes data, creates possible reg models,
                    delegates creation to RegModels
- RegModel: Fits regression model based on specifications identified by 
            RegModelManager    
"""


class DataManager:
    """
    Handles data transformations in preparation for regression
    """

    def __init__(self, path, name, dv_name):
        if os.path.splitext(path)[1] == '.pkl':
            self.data = pd.read_pickle(path)
        elif os.path.splitext(path)[1] == '.csv':
            self.data = pd.read_csv(path)
        else:
            print(f'Cannot read {os.path.splitext(path)[1]} files')
            exit(1)
        self.name = name
        self.avg_data = None
        self.med_data = None
        self.var_data = None
        self.bin_data = None
        self.dv_name = dv_name

    def run(self):
        self.data = self.data.applymap(
            lambda x: [float(val) for val in x] if isinstance(x, list) else x)
        self.timeseries_to_avg()
        self.timeseries_to_med()
        self.timeseries_to_var()
        self.avg_to_bins()

    def timeseries_to_max(self):
        self.max_data = self.data.copy()
        self.max_data.replace(to_replace='', value=np.nan, inplace=True)
        self.max_data.iloc[:, self.max_data.columns != self.dv_name] = \
            self.max_data.iloc[:, self.max_data.columns != self.dv_name] \
                .applymap(lambda x: np.nanmax(x) if isinstance(x, list) else x)

    def timeseries_to_min(self):
        self.min_data = self.data.copy()
        self.min_data.replace(to_replace='', value=np.nan, inplace=True)
        self.min_data.iloc[:, self.min_data.columns != self.dv_name] = \
            self.min_data.iloc[:, self.min_data.columns != self.dv_name] \
                .applymap(lambda x: np.nanmin(x) if isinstance(x, list) else x)

    def timeseries_to_avg(self):
        self.avg_data = self.data.copy()
        self.avg_data.replace(to_replace='', value=np.nan, inplace=True)
        self.avg_data.iloc[:, self.avg_data.columns != self.dv_name] = \
            self.avg_data.iloc[:, self.avg_data.columns != self.dv_name] \
                .applymap(lambda x: np.nanmean(x) if isinstance(x, list) else x)

    def timeseries_to_med(self):
        self.med_data = self.data.copy()
        self.med_data.replace(to_replace='', value=np.nan, inplace=True)
        self.med_data.iloc[:, self.med_data.columns != self.dv_name] = \
            self.med_data.iloc[:, self.med_data.columns != self.dv_name] \
                .applymap(
                lambda x: np.nanmedian(x) if isinstance(x, list) else x)

    def timeseries_to_var(self):
        self.var_data = self.data.copy()
        self.var_data.replace(to_replace='', value=np.nan, inplace=True)
        self.var_data.iloc[:,
        self.avg_data.columns != self.dv_name] = \
            self.var_data.iloc[:, self.avg_data.columns != self.dv_name]\
                .applymap(lambda x: np.nanvar(x) if isinstance(x, list) else x)

    def avg_to_bins(self):
        self.bin_data = self.avg_data.copy()
        self.bin_data.replace(to_replace='', value=np.nan, inplace=True)
        excluded_cols = self.bin_data.columns[
            self.bin_data.isin([0, 1]).all()].append(pd.Index([self.dv_name]))
        self.bin_data.loc[:, ~self.bin_data.columns.isin(
            list(excluded_cols))] \
            = self.bin_data.loc[:, ~self.bin_data.columns.isin(
            list(excluded_cols))].apply(self.bin_by_quartiles)

    def val_to_nan(self, val):
        self.data = self.data.applymap(lambda cell:
                                       [x if (x != val) else np.nan for x in
                                        cell]
                                       if isinstance(cell, list)
                                       else cell)

    @staticmethod
    def bin_by_quartiles(col):
        bin_labels = ['quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']
        col = pd.to_numeric(col)
        bin_col = pd.qcut(col, q=4, duplicates='drop')
        return bin_col


class RegModelManager:
    """
    Takes data, creates possible reg models, delegates creation to RegModels
    One instance per dataset
    """

    def __init__(self, data, dv_name, data_type, data_name, avg_data=None):
        self.independent_vars = None
        self.data = data
        self.org_data = data
        self.avg_data = avg_data
        self.data_type = data_type
        self.dv_name = dv_name
        self.data_name = data_name
        self.estimations = []
        self.predictor_sets = []
        self.best_model = None
        self.thresh = 0.1

    def run(self):
        """
        creates model specification and delegates model fitting
        """
        self.data = self.data.dropna()

        if self.data_type == 'bin':
            self.avg_data = self.avg_data.dropna()
            X = self.avg_data.loc[:, self.avg_data.columns != self.dv_name]
        else:
            X = self.data.loc[:, self.data.columns != self.dv_name]
        y = self.data.loc[:, self.data.columns == self.dv_name]
        predictor_set = self.stepwise_selection(X.astype(float),
                                                y.astype(float))

        reg_model = RegModel(self.data,
                             predictor_set,
                             self.dv_name,
                             self.data_type)
        reg_model.fit_model()
        self.estimations.append(reg_model.est)

    def get_base_case(self, col):
        if self.data_type == 'bin':
            data_temp = self.avg_data.loc[:, self.avg_data.columns.str
                                                 .startswith(col)]
            unique_vals = self.org_data.col.unique()
        else:
            data_temp = self.data.loc[:, self.data.columns.str.startswith(col)]
            unique_vals = self.org_data.col.unique()
        data_temp = data_temp[(data_temp.T == 0).all()]
        data_temp.columns = data_temp.columns.str.replace(f'{col}_', '')
        cols = data_temp.columns
        base_case = list(set(unique_vals) - set(cols))
        return base_case

    def correlate(self):
        if self.data_type == 'bin':
            corr_data = self.avg_data
        else:
            corr_data = self.data
        # drop_cols = [col for col in corr_data if (('synch' not in col)
        # and (col != 'normalized_distance') and (col != self.dv_name))]
        # corr_data = corr_data.drop(columns=drop_cols)
        corr = corr_data.corr()
        p_matrix = np.zeros(shape=(corr_data.shape[1],
                                   corr_data.shape[1]))
        for col in corr_data.columns:
            for col2 in corr_data.drop(col, axis=1).columns:
                nas = np.logical_or(np.isnan(corr_data[col]),
                                    np.isnan(corr_data[col2]))
                _, p = scipy.stats.pearsonr(corr_data[col][~nas],
                                            corr_data[col2][~nas])
                p_matrix[corr_data.columns.to_list().index(
                    col), corr_data.columns.to_list().index(
                    col2)] = p
        p_values = pd.DataFrame(p_matrix)
        return corr, p_values

    def initial_predictors(self, corr, p_values):
        initial_predictors = []
        for idx, var in enumerate(corr[self.dv_name]):
            if list(corr.index)[idx] != self.dv_name and \
                    p_values.iloc[idx, corr.columns.get_loc(self.dv_name)] \
                    < self.thresh:
                initial_predictors.append(list(corr.index)[idx])
        return initial_predictors

    @staticmethod
    def correlated_predictors(corr, p_values, initial_predictors):
        # Get predictors that are correlated amongst each other
        corr_tri = np.tril(corr)
        corr_tri[np.triu_indices(corr_tri.shape[0], 0)] = np.nan
        correlated_predictors = []
        for i in range(corr_tri.shape[0]):
            for j in range(corr_tri.shape[1]):
                if (corr_tri[i, j] > 0.6 or corr_tri[i, j] < -0.6) and \
                        p_values.iloc[i, j] < 0.05 and (
                        corr.columns[i] in initial_predictors or
                        corr.columns[j] in initial_predictors):
                    correlated_predictors.append(
                        [corr.columns[i], corr.columns[j]])
        return correlated_predictors

    @staticmethod
    def get_predictor_sets(initial_predictors, correlated_predictors):
        # Create lists with predictors that are not correlated
        # amongst each other, but correlated with DV
        predictor_sets = [initial_predictors]
        for tuples in correlated_predictors:
            for idx, predictor_list in enumerate(predictor_sets):
                if tuples[0] in predictor_list and tuples[1] in predictor_list:
                    sublist1 = predictor_list.copy()
                    sublist1.remove(tuples[0])
                    sublist2 = predictor_list.copy()
                    sublist2.remove(tuples[1])
                    predictor_sets.append(sublist1)
                    predictor_sets.append(sublist2)
                    del predictor_sets[idx]
        predictor_sets = [x for x in predictor_sets if not any(
            set(x) <= set(y) for y in predictor_sets if x is not y)]
        return predictor_sets

    def plot_cor_matrix(self, corr, p_values):
        plt.rcParams['font.family'] = 'CMU Serif'
        plt.rcParams['font.sans-serif'] = 'CMU Serif Roman'
        mask = np.invert(np.tril(p_values < self.thresh))
        f, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr, ax=ax,
                    mask=mask,
                    # cosmetics
                    annot=True, vmin=-1, vmax=1, center=0,
                    cmap='coolwarm', linewidths=2, linecolor='black',
                    cbar_kws={'orientation': 'vertical'})

    def stepwise_selection(self, X, y,
                           initial_list=[],
                           threshold_in=0.15,
                           threshold_out=0.150001,
                           verbose=False):
        """ Forward-backward feature selection.
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
        Always set threshold_in < threshold_out to avoid infinite looping.
        """
        included = list(initial_list)
        while True:
            changed = False
            # forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded, dtype='float64')
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(
                    pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature,
                                                                 best_pval))

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()  # null if pvalues is empty
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature,
                                                                 worst_pval))
            if not changed:
                break
        return included


class RegModel:
    """
    Fits regression model based on specifications identified by RegModelManager
    Several instances per dataset
    """

    def __init__(self, data, predictor_set, dv_name, data_type):
        self.x = None
        self.est = None
        self.data = data
        self.predictor_set = predictor_set
        self.dv_name = dv_name
        self.data_type = data_type
        self.fit_model()

    def fit_model(self):
        if len(self.predictor_set) == 0:
            self.est = None
        else:
            y = self.data[self.dv_name]
            self.x = self.data[self.predictor_set]
            if self.data_type == 'bin':
                self.x = pd.get_dummies(self.x, prefix=self.predictor_set,
                                        columns=self.predictor_set,
                                        drop_first=True)
            self.x = sm.add_constant(self.x)
            self.est = sm.OLS(y, self.x).fit()

    def test_breusch_pagan(self):
        test = sms.het_breuschpagan(self.est.resid, self.est.model.exog)
        if test[1] > 0.05:
            print("Failed to reject the null hypothesis. "
                  "Not enough evidence to assume presence of "
                  "heteroscedasticity.")
            ret = 0
        else:
            print("Null hypothesis rejected. "
                  "Evidence for presence of heteroscedasticity.")
            ret = 1
        return ret

    def plot_predictor_line(self, predictor):
        fig = sm.graphics.plot_ccpr(self.est, predictor)
        plt.show(fig)

    def print_summary(self):
        print(self.est.summary())


class OverviewPlotter:
    """
    Creates overview tables for batch runs
    """

    def __init__(self, reg_managers):
        self.reg_managers = reg_managers
        self.overview = None

    def plot_all(self):
        for reg_manager in self.reg_managers:
            for est in reg_manager.estimations:
                print('─' * 80)
                print('─' * 80)
                print(f'Dataset: {reg_manager.data_name}\n'
                      f'Timeseries transformation: {reg_manager.data_type}')
                print('─' * 80)
                if est is None:
                    print('No significant predictors found.')
                else:
                    print(est.summary())

    def plot_best_r2(self):
        best_rsq = [0, 0, 0]
        for reg_manager_idx, reg_manager in enumerate(self.reg_managers):
            for est_idx, est in enumerate(reg_manager.estimations):
                if est is not None:
                    if est.rsquared_adj > best_rsq[0]:
                        best_rsq = [est.rsquared_adj, reg_manager_idx, est_idx]
        print(f'Dataset: {self.reg_managers[best_rsq[1]].data_name}\n'
              f'Timeseries transformation: '
              f'{self.reg_managers[best_rsq[1]].data_type}')
        print('─' * 80)
        print(self.reg_managers[best_rsq[1]].estimations[best_rsq[2]].summary())
        return self.reg_managers[best_rsq[1]].estimations[best_rsq[2]], \
               self.reg_managers[best_rsq[1]]

    def plot_overview(self):
        row = []
        for reg_manager in self.reg_managers:
            for idx, est in enumerate(reg_manager.estimations):
                predictor_overview = pd.concat([est.params.rename('param'),
                                                est.pvalues.rename('pval')],
                                               axis=1)
                significant_predictor_overview = predictor_overview \
                    .loc[predictor_overview['pval'] < 0.05]
                row.append(
                    {
                        'data_name': reg_manager.data_name,
                        'data_type': reg_manager.data_type,
                        'adj_r_squared': est.rsquared_adj,
                        'predictors': reg_manager.predictor_sets[idx],
                        'significant_predictors': significant_predictor_overview
                    }
                )
        overview = pd.DataFrame(row)
        return overview

    @staticmethod
    def get_vif(est):
        variables = est.model.exog
        vif = [variance_inflation_factor(variables, i) for i in
               range(variables.shape[1])]
        return vif

    @staticmethod
    def plot_3d():

        columns = ['bodyside_compared', 'min_point_of_synch', 'med_rval',
                   'var_rval']

        df = pd.DataFrame({'bodyside_compared': [1, 1, 2, 2],
                           'min_point_of_synch': [1, 2, 1, 2],
                           'med_rval': [0.147, 0.166, 0.192, 0.131],
                           'var_rval': [0.088, 0.095, 0.100, 0.107]},
                          columns=columns)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        df["med_rval"] = np.log(df["med_rval"] + 1)
        df["var_rval"] = np.log(df["var_rval"] + 1)

        colors = ['r', 'g', 'b']

        num_bars = 4
        x_pos = []
        y_pos = []
        x_size = np.ones(num_bars * 2) / 4
        y_size = np.ones(num_bars * 2) / 4
        c = ['med_rval', 'var_rval']
        z_pos = []
        z_size = []
        z_color = []
        for i, col in enumerate(c):
            x_pos.append(df["bodyside_compared"])
            y_pos.append(df["min_point_of_synch"] + i / 2.9
                         )
            z_pos.append([0] * num_bars)
            z_size.append(df[col])
            z_color.append([colors[i]] * num_bars)

        x_pos = np.reshape(x_pos, (8,))
        y_pos = np.reshape(y_pos, (8,))
        z_pos = np.reshape(z_pos, (8,))
        z_size = np.reshape(z_size, (8,))
        z_color = np.reshape(z_color, (8,))

        ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color=z_color)

        plt.xlabel('bodyside_compared')
        plt.ylabel('min_point_of_synch')
        ax.set_zlabel('r_squared')

        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='A',
                                  markerfacecolor='r', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='B',
                                  markerfacecolor='g', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='C',
                                  markerfacecolor='b', markersize=10)
                           ]

        # Make legend
        ax.legend(handles=legend_elements, loc='best')
        # Set view
        ax.view_init(elev=35., azim=35)
        # plt.show()
        locs, labels = plt.xticks()
        plt.xticks((min(locs) + (max(locs) - min(locs)) / 4,
                    min(locs) + (max(locs) - min(locs)) / 4 * 3),
                   ['same-side', 'opposite-side'])  # Set text labels.
        locs, labels = plt.yticks()
        plt.yticks((min(locs) + (max(locs) - min(locs)) / 4,
                    min(locs) + (max(locs) - min(locs)) / 4 * 3),
                   ['linear', 'perpendicular'])  # Set text labels.
        print(locs)
        print(labels)