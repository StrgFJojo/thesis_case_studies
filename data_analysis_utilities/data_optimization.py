import math

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

"""
Used for initial exploration of study data.
Not further used in later analysis.
"""


def fill_missing_values_synchrony(data):
    data_optimized = data.copy()
    for idx, val in enumerate(data_optimized):
        if val == -1:  # if element not available
            # find next non-na element
            idx_next_avail = min([x for x in range(idx, len(data_optimized))
                                  if data_optimized[x] > -1], default=-1)
            # find previous non-na element
            idx_previous_avail = max([x for x in range(0, idx)
                                      if data_optimized[x] > -1], default=-1)
            if idx_next_avail != -1 and idx_previous_avail != -1:
                data_optimized[idx] = (data_optimized[int(idx_next_avail)]
                                       + data_optimized[
                    int(idx_previous_avail)]) / 2
            elif idx_previous_avail != -1:
                data_optimized[idx] = data_optimized[idx_previous_avail]
            elif idx_next_avail != -1:
                data_optimized[idx] = data_optimized[idx_next_avail]
            else:
                data_optimized[idx] = -1
    data_optimized = [x if (x != -1) else np.nan for x in data_optimized]
    return data_optimized


def remove_missing_vals(data):
    data_temp = data.copy()
    data_temp = [x if (x != -1) else np.nan for x in data_temp]
    data_new = [x for x in data_temp if not np.isnan(x)]
    return data_new


def subsample_arr(data):
    data_subsampled = data.copy()
    for i in range(len(data_subsampled)):
        for j in range(13, 31):
            data_subsampled.iat[i, j] = (data_subsampled.iat[i, j])[::10]
    return data_subsampled


def remove_outliers(data):
    data_new = [x if (x != -1) else np.nan for x in data]
    if not (len(data_new) == 0 or np.isnan(data_new).all()):
        upper_bound = np.nanmean(data_new) + np.nanstd(data_new)
        lower_bound = np.nanmean(data_new) - np.nanstd(data_new)
        data_new = [x for x in data_new if lower_bound <= x <= upper_bound]
    return data_new


def smooth(data):
    wl = math.floor(len(data)/10)
    data_new = savgol_filter(data, wl, 1)
    return data_new


def avg(data):
    if len(data) > 90:  # min 30 secs of synch tracking in video material
        data_new = np.nanmean(data)
    else:
        data_new = np.NaN
    return data_new


def make_nan(data):
    data_new = [x if (x != -1) else np.nan for x in data]
    return data_new


def plot_cor_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, ax=ax,
                mask=mask,
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', linewidths=2, linecolor='black',
                cbar_kws={'orientation': 'vertical'})


def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1], df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col, axis=1).columns:
            _, p = stats.pearsonr(df[col], df[col2])
            p_matrix[df.columns.to_list().index(col), df.columns
                     .to_list().index(col2)] = p
    return p_matrix
