import argparse

import numpy as np
from tqdm import tqdm

from modules.data import download_data, ALL_DATASETS
from modules.plotting import get_object_to_plot, plot

from modules.upper_estimation import upper_estimation
from modules.utils import _get_optimalty
from modules.optimality import subset_covariance
from modules.dpp import dpp
from scipy.linalg import fractional_matrix_power as frac_power


for dataset in ALL_DATASETS:
    dataset_name = dataset
    print(f'Upper estimation for {dataset_name} starts')
    df = download_data(dataset_name)
    df = df.dropna()

    X = df.values[:, 1:]
    num_samples = X.shape[0]
    print(f'dataframe shape: {X.shape}')
    n, d = X.shape

    A = (1 / n) * np.eye(d)

    methods = ['uniform', 'bottom_up', 'bayessian_dpp', 'predictive_length']
    opt_types = ['A', 'C', 'D', 'V']

    k_linspace = np.arange(start=int(1.5 * d), stop=5 * d + 1, step=1)

    for method in methods:

        if method == 'bottom_up':
            to_plot = []
            for optimality in opt_types:
                upper_estimation_bottom_up = []
                for ix in tqdm(range(len(k_linspace))):
                    k = k_linspace[ix]
                    # Sigma
                    assert num_samples >= k, f'number of samples should be greater than k'
                    selected_ixs = []
                    full_samples_set = set(list(range(num_samples)))
                    candidate_samples = list(full_samples_set - set(selected_ixs))

                    for candidate_sample in candidate_samples:
                        candidate_ixs = np.append(selected_ixs, [candidate_sample]).astype(int)
                    Sigma=subset_covariance(X[candidate_ixs])

                    upper_estimation_bottom_up.append(
                        upper_estimation(X=X, A=A, k=k, Sigma=Sigma, opt_type=optimality))

                plot_prop = {
                    'label': optimality}

                object_to_plot = get_object_to_plot(
                    X=k_linspace, Y=upper_estimation_bottom_up, plot_prop=plot_prop)
                to_plot.append(object_to_plot)

            plot(to_plot, x_label='k', y_label=f'upper estimation',
                 title=f'Upper estimation: dataset="{dataset_name}" & {method}', save_path=f'upper_estimation/{method}/{dataset_name}; method is {method}.png')

        elif method == 'bayessian_dpp':
            to_plot = []
            for optimality in opt_types:
                upper_estimation_dpp = []

                optimalty_func = _get_optimalty(optimality)

                for ix in tqdm(range(len(k_linspace))):
                    k = k_linspace[ix]
                    p = k / num_samples * np.ones(num_samples)

                    Z = A + (X.T * p) @ X
                    B = (np.sqrt(p[:, None]) * X) @ frac_power(Z, -0.5)
                    DPP = dpp(B)
                    DPP.sample_exact()
                    all_n = np.arange(num_samples)
                    b = set(all_n[np.random.uniform(size=num_samples) < p])
                    selected_ixs = set(DPP.list_of_samples[0]) | b
                    selected_ixs = np.array(list(selected_ixs))
                    Sigma = subset_covariance(X[selected_ixs])

                    upper_estimation_dpp.append(
                        upper_estimation(X=X, A=A, k=k, Sigma=Sigma, opt_type=optimality))

                plot_prop = {
                    'label': optimality}

                object_to_plot = get_object_to_plot(
                    X=k_linspace, Y=upper_estimation_dpp, plot_prop=plot_prop)
                to_plot.append(object_to_plot)

            plot(to_plot, x_label='k', y_label=f'upper estimation',
                 title=f'Upper estimation: dataset="{dataset_name}" & {method}', save_path=f'upper_estimation/{method}/{dataset_name}; method is {method}.png')

        elif method == 'predictive_length':
            to_plot = []
            for optimality in opt_types:
                upper_estimation_predic = []

                optimalty_func = _get_optimalty(optimality)
                for ix in tqdm(range(len(k_linspace))):
                    k = k_linspace[ix]

                    probs = np.linalg.norm(X, axis=1)
                    probs = probs / np.sum(probs)
                    selected_ixs = np.random.choice(num_samples,
                                                    size=k,
                                                    replace=False,
                                                    p=probs)
                    Sigma = subset_covariance(X[selected_ixs])
                    upper_estimation_predic.append(
                        upper_estimation(X=X, A=A, k=k, Sigma=Sigma, opt_type=optimality))

                plot_prop = {
                    'label': optimality}

                object_to_plot = get_object_to_plot(
                    X=k_linspace, Y=upper_estimation_predic, plot_prop=plot_prop)
                to_plot.append(object_to_plot)

            plot(to_plot, x_label='k', y_label=f'upper estimation',
                 title=f'Upper estimation: dataset="{dataset_name}" & {method}',
                 save_path=f'upper_estimation/{method}/{dataset_name}; method is {method}.png')

        elif method == 'uniform':
            to_plot = []
            for optimality in opt_types:
                upper_estimation_uniform = []

                optimalty_func = _get_optimalty(optimality)
                for ix in tqdm(range(len(k_linspace))):
                    k = k_linspace[ix]

                    assert num_samples >= k, f'number of samples should be greater than k'
                    selected_ixs = np.random.choice(num_samples, size=k, replace=False)
                    Sigma = subset_covariance(X[selected_ixs])
                    upper_estimation_uniform.append(
                        upper_estimation(X=X, A=A, k=k, Sigma=Sigma, opt_type=optimality))

                plot_prop = {
                    'label': optimality}

                object_to_plot = get_object_to_plot(
                    X=k_linspace, Y=upper_estimation_uniform, plot_prop=plot_prop)
                to_plot.append(object_to_plot)

            plot(to_plot, x_label='k', y_label=f'upper estimation',
                 title=f'Upper estimation: dataset="{dataset_name}" & {method}',
                 save_path=f'upper_estimation/{method}/{dataset_name}; method is {method}.png')



