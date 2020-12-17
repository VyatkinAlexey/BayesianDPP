import argparse

import numpy as np
from tqdm import tqdm

from modules.data import download_data, ALL_DATASETS
from modules.plotting import get_object_to_plot, plot

from selection_methods.uniform import select_uniform
from selection_methods.bottom_up import select_bottom_up
from selection_methods.predictive_length import select_predictive_length
from selection_methods.bayesian_dpp import select_bayesian_dpp


def _get_selection(selection_name: str, optimality_name: str = 'A'):
    if selection_name == 'uniform':
        return lambda X, A, k: select_uniform(X, A, k, optimality=optimality_name)
    elif selection_name == 'bottom_up':
        return lambda X, A, k: select_bottom_up(X, A, k, optimality=optimality_name)
    elif selection_name == 'predictive_length':
        return lambda X, A, k: select_predictive_length(X, A, k, optimality=optimality_name)
    elif selection_name == 'bayesian_dpp':
        return lambda X, A, k: select_bayesian_dpp(X, A, k, optimality=optimality_name)
    elif selection_name == 'bayesian_dpp_sdp':
        return lambda X, A, k: select_bayesian_dpp(X, A, k, optimality=optimality_name, with_sdp=True)
    else:
        raise NotImplementedError(f'selection method {selection_name} is not implemeted')


METHOD_TO_COLOR = {
    'uniform': 'red',
    'bottom_up': 'green',
    'predictive_length': 'purple',
    'bayesian_dpp': 'orange',
    'bayesian_dpp_sdp': 'blue'
}

METHOD_TO_LINESTYLE = {
    'uniform': 'dashed',
    'bottom_up': 'dashdot',
    'predictive_length': 'dotted',
    'bayesian_dpp': 'solid',
    'bayesian_dpp_sdp': 'dashdot'
}

DATASET_TO_TOP = {
    'mg_scale': 20,
    'mpg_scale': 30,
    'housing_scale': 50,
    'bodyfat_scale': 80
}


def do_experiment(X: np.ndarray,
                  A: np.ndarray = None,
                  method: str = 'uniform',
                  bootsrtap_size: int = 25,
                  alpha: float = 0.05):
    """

    :param X: np.ndarray, matrix of features
    :param A: np.ndarray, prior precision matrix
    :param method: str, method to select subset of S
    :return:
    """
    n, d = X.shape
    if A is None:
        A = (1 / n) * np.eye(d)
    selection_func = _get_selection(method)
    k_linspace = np.arange(start=int(1.5 * d), stop=5 * d + 1, step=1)
    if method == 'bottom_up':
        scores_mean = np.empty(len(k_linspace), dtype=float)
        for ix in tqdm(range(len(k_linspace))):
            selected_samples, score = selection_func(X=X, A=A, k=k_linspace[ix])
            scores_mean[ix] = score
        scores_top = None
        scores_down = None
    elif method in ['bayessian_dpp', 'bayesian_dpp_sdp']:
        scores = dict(zip(k_linspace, [[] for _ in range(len(k_linspace))]))
        for bootsrtap_ix in tqdm(range(bootsrtap_size)):
            for ix, k in enumerate(k_linspace):
                selected_samples, score = selection_func(X=X, A=A, k=k)
                if len(selected_samples) in scores:
                    scores[len(selected_samples)].append(score)

        scores_mean = np.empty_like(k_linspace)
        scores_top = np.empty_like(k_linspace)
        scores_down = np.empty_like(k_linspace)

        for k_ix, k in k_linspace:
            current_scores = scores[k]
            scores_mean[k_ix] = np.mean(current_scores)
            scores_top[k_ix] = np.percentile(current_scores, 100 - 100 * (alpha/2))
            scores_down[k_ix] = np.percentile(current_scores, 100 * (alpha/2))
    else:
        scores = np.empty((len(k_linspace), bootsrtap_size), dtype=float)
        for bootsrtap_ix in tqdm(range(bootsrtap_size)):
            for ix, k in enumerate(k_linspace):
                selected_samples, score = selection_func(X=X, A=A, k=k)
                scores[ix, bootsrtap_ix] = score

        scores_mean = np.mean(scores, axis=1)
        scores_top = np.percentile(scores, 95, axis=1)
        scores_down = np.percentile(scores, 5, axis=1)

    plot_prop = {
        'label': method,  # 'label' is necessary
        'color': METHOD_TO_COLOR[method],
        'linestyle': METHOD_TO_LINESTYLE[method]}

    obj_to_plot = get_object_to_plot(k_linspace, scores_mean, plot_prop, scores_top, scores_down)

    return obj_to_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Script to perform experiment for all datasets')
    parser.add_argument('-d', '--dataset', type=str,
                        default='mg_scale',
                        choices=ALL_DATASETS,
                        help='name of given dataset')
    parser.add_argument('-b', '--bootsrt_size', type=int, default=25,
                        help='bootstrap size')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='level of confidence')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset_name = args.dataset
    print(f'Experiment for {dataset_name} starts')
    df = download_data(dataset_name)
    df = df.dropna()

    X = df.values[:, 1:] # zero is taget
    print(f'dataframe shape: {X.shape}')
    plot_objects = []
    for method in ['uniform', 'predictive_length', 'bottom_up', 'bayesian_dpp']:
        obj_to_plot = do_experiment(X, method=method,
                                    bootsrtap_size=args.bootsrt_size,
                                    alpha=args.alpha)
        plot_objects.append(obj_to_plot)
    plot(plot_objects,
         x_label='Subset size (multiple of d)',
         y_label='A-optimality',
         title=f'dataset="{dataset_name}"',
         d=X.shape[1],
         ytop=DATASET_TO_TOP[dataset_name],
         save_path=f'{dataset_name}.png')