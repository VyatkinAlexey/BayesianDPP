# TODO add functions to plot graphs from the article
import numpy as np
import matplotlib.pyplot as plt

sty = 'seaborn'

'''
# Example:

Y = np.linspace(1, 10, 29)
X = np.arange(1, 30)
Y_top = Y - 0.5
Y_down = Y + 0.3
plot_prop_1 = {
    'label': 'label_1', # 'label' is necessary 
    'color': 'green',
    'linestyle': 'dashed'}
    
obj_to_plot_1 = get_object_to_plot(X, Y, plot_prop_1, Y_top, Y_down)

Y_2 = np.linspace(1, 4.5, 29) * 2
X_2 = np.arange(1, 30)
Y_top_2 = Y_2 - 0.7
Y_down_2 = Y_2 + 0.8
plot_prop_2 = {'label': 'label_2'}

obj_to_plot_2 = get_object_to_plot(X_2, Y_2, plot_prop_2)

x_label = 'x_label'
y_label = 'y_label'
title = 'title'
to_plot_list = [obj_to_plot_1, obj_to_plot_2]

plot(to_plot_list, x_label=x_label, y_label=y_label, title=title)

'''


def get_object_to_plot(X, Y, plot_prop: dict,
                       Y_top=None, Y_down=None):
    """
    Prepare object - python set - for plot()
    :param X: np array
    :param Y: np array
    :param plot_prop: dict with properties for ax.plot() such as 'label', 'color', 'linestyle' ...
    'label' is not optional
    https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    :param Y_top: np array
    :param Y_down: np array
    :return: python set
    """
    if 'label' not in plot_prop.keys():
        raise Exception("Put 'label' in plot_prop argument to plot, "
                        "plot_prop = {'label': label}")

    if Y_top is not None and Y_down is not None:
        return ([X, Y, Y_top, Y_down], plot_prop)
    else:
        return ([X, Y], plot_prop)


def plot(to_plot: list, x_label: str, y_label: str, title: str):
    """
    Plot several plots
    :param to_plot: list from python sets. Sets gotten from get_object_to_plot()
    :param x_label: str
    :param y_label: str
    :param title: str
    :return: None
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for plot in to_plot:
        points = plot[0]
        prop = plot[1]
        ax.plot(points[0], points[1], **prop)
        print(len(points))
        print(points)
        if len(points) == 4:
            if 'color' in prop.keys():
                ax.fill_between(points[0], points[2], points[3],
                                facecolor=prop['color'], alpha=0.2)
            else:
                ax.fill_between(points[0], points[2], points[3], alpha=0.2)

    plt.grid(True, color='w', linewidth=1)
    plt.gca().patch.set_facecolor((233 / 256, 233 / 256, 242 / 256))
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    plt.legend(loc='best')
    plt.xlim(right=5)
    plt.xlim(left=1.2)
    plt.show()


# Example:
#
# Y = np.linspace(1, 10, 29)
# X = np.arange(1, 30)
# Y_top = Y - 0.5
# Y_down = Y + 0.3
# plot_prop_1 = {
#     'label': 'label_1',  # 'label' is necessary
#     'color': 'green',
#     'linestyle': 'dashed'}
#
# obj_to_plot_1 = get_object_to_plot(X, Y, plot_prop_1, Y_top, Y_down)
#
# Y_2 = np.linspace(1, 4.5, 29) * 2
# X_2 = np.arange(1, 30)
# Y_top_2 = Y_2 - 0.7
# Y_down_2 = Y_2 + 0.8
# plot_prop_2 = {'label': 'label_2'}
#
# obj_to_plot_2 = get_object_to_plot(X_2, Y_2, plot_prop_2)
#
# x_label = 'x_label'
# y_label = 'y_label'
# title = 'title'
# to_plot_list = [obj_to_plot_1, obj_to_plot_2]
#
# plot(to_plot_list, x_label=x_label, y_label=y_label, title=title)
