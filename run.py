import argparse

import numpy as np
from tqdm import tqdm

from modules.data import download_data, ALL_DATASETS
from modules.plotting import get_object_to_plot, plot

from selection_methods.uniform import select_uniform
from selection_methods.bottom_up import select_bottom_up
from selection_methods.predictive_length import select_predictive_length


def _get_selection(selection_name: str):
    if selection_name == 'uniform':
        return select_uniform
    elif selection_name == 'bottom_up':
        return select_bottom_up
    elif selection_name == 'predictive_length':
        return select_predictive_length
    else:
        raise NotImplementedError(f'selection method {selection_name} is not implemeted')


METHOD_TO_COLOR = {
    'uniform': 'red',
    'bottom_up': 'green',
    'predictive_length': 'blue'
}


def do_experiment(X: np.ndarray, A: np.ndarray = None, method: str = 'uniform'):
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
    k_linspace = np.arange(start=d + 3, stop=5 * d + 1, step=1)
    if method == 'bottom_up':
        scores_mean = np.empty(len(k_linspace), dtype=float)
        for ix, k in enumerate(k_linspace):
            selected_samples, score = selection_func(X=X, A=A, k=k)
            scores_mean[ix] = score
        scores_top = None
        scores_down = None
    else:
        bootsrtap_size = 25
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
        'linestyle': 'dashed'}

    obj_to_plot = get_object_to_plot(k_linspace, scores_mean, plot_prop, scores_top, scores_down)

    return obj_to_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Script to perform experiment')
    parser.add_argument('-d', '--dataset', type=str,
                        default='mg_scale',
                        choices=ALL_DATASETS,
                        help='name of given dataset')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset_name = args.dataset
    print(f'Experiment for {dataset_name} starts')
    df = download_data(dataset_name)

    X = df.values[:, 1:] # zero is taget
    print(f'dataframe shape: {X.shape}')
    plot_objects = []
    for method in ['uniform', 'predictive_length', 'bottom_up']:
        obj_to_plot = do_experiment(X, method=method)
        plot_objects.append(obj_to_plot)
    plot(plot_objects, x_label='Subset size', y_label='A-optimality', title=f'dataset="{dataset_name}"')