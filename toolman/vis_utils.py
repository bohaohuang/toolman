"""

"""


# Built-in

# Libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Own modules
from .img_utils import change_channel_order


def get_default_colors():
    """
    Get plt default colors
    :return: a list of rgb colors
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_color_list():
    """
    Get default color list in plt, convert hex value to rgb tuple
    :return:
    """
    colors = get_default_colors()
    return [tuple(int(a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for a in colors]


def decode_label_map(label, label_num=2, label_colors=None):
    """
    #TODO this could be more efficient
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :param label_colors: list of tuples with RGB value of label colormap
    :return:
    """
    if len(label.shape) == 3:
        label = np.expand_dims(label, -1)
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    if not label_colors:
        color_list = get_color_list()
        label_colors = {}
        for i in range(label_num):
            label_colors[i] = color_list[i]
        label_colors[0] = (255, 255, 255)
    for i in range(n):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def inv_normalize(img, mean, std):
    """
    Do inverse normalize for images
    :param img: the image to be normalized
    :param mean: the original mean
    :param std: the original std
    :return:
    """
    inv_mean = [-a / b for a, b in zip(mean, std)]
    inv_std = [1 / a for a in std]
    if len(img.shape) == 3:
        return (img - inv_mean) / inv_std
    elif len(img.shape) == 4:
        for i in range(img.shape[0]):
            img[i, :, :, :] = (img[i, :, :, :] - inv_mean) / inv_std
        return img


def make_image_banner(imgs, n_class, mean, std, max_ind=(2, ), decode_ind=(1, 2), chanel_first=True):
    """
    Make image banner for the tensorboard
    :param imgs: list of images to display, each element has shape N * C * H * W
    :param n_class: the number of classes
    :param mean: mean used in normalization
    :param std: std used in normalization
    :param max_ind: indices of element in imgs to take max across the channel dimension
    :param decode_ind: indicies of element in imgs to decode the labels
    :param chanel_first: if True, the inputs are in channel first format
    :return:
    """
    for cnt in range(len(imgs)):
        if cnt in max_ind:
            # pred: N * C * H * W
            imgs[cnt] = np.argmax(imgs[cnt], 1)
        if cnt in decode_ind:
            # lbl map: N * 1 * H * W
            imgs[cnt] = decode_label_map(imgs[cnt], n_class)
        if (cnt not in max_ind) and (cnt not in decode_ind):
            # rgb image: N * 3 * H * W
            imgs[cnt] = inv_normalize(change_channel_order(imgs[cnt]), mean, std) * 255
    banner = np.concatenate(imgs, axis=2).astype(np.uint8)
    if chanel_first:
        banner = change_channel_order(banner, False)
    return banner


def make_cmp_mask(lbl, pred, tp_mask_color=(0, 255, 0), fp_mask_color=(255, 0, 0), fn_mask_color=(0, 0, 255)):
    """
    Make compare mask for visualization purpose, the label and prediction maps should be binary and the truth value can
    only be 1
    :param lbl: the label map with dimension height * width
    :param pred: the prediction map with dimension height * width
    :param tp_mask_color: the rgb color of TP pixels, green by default
    :param fp_mask_color: the rgb color of FP pixels, red by default
    :param fn_mask_color: the rgb color of FN pixels, blue by default
    :return:
    """
    assert lbl.shape == pred.shape
    if np.max(lbl) != 1:
        lbl = lbl / np.max(lbl)
    if np.max(pred) != 1:
        pred = pred / np.max(pred)
    cmp_mask = 255 * np.ones((*lbl.shape, 3), dtype=np.uint8)
    tp_mask = (lbl == 1) * (lbl == pred)
    fp_mask = (pred - lbl) == 1
    fn_mask = (lbl - pred) == 1
    cmp_mask[tp_mask, :] = tp_mask_color
    cmp_mask[fp_mask, :] = fp_mask_color
    cmp_mask[fn_mask, :] = fn_mask_color
    return cmp_mask


def compare_figures(images, nrows_ncols, fig_size=(10, 8), show_axis=False, show_fig=True,
                    title_list=None, cmap=None, v_range=None):
    """
    Show images in grid pattern, link their x and y axis
    :param images: list of images to be displayed
    :param nrows_ncols: a tuple of (n_h, n_w) where n_h is #elements/row and n_w is #elements/col
    :param fig_size: a tuple of figure size
    :param show_axis: if True, each subplot will have its axis shown
    :param show_fig: if True, plt.show() will be called
    :param title_list: list of title names to be displayed on each sub images
    :param cmap: list of plt color maps to be used for each image
    :return:
    """
    from mpl_toolkits.axes_grid1 import Grid
    if title_list:
        assert len(title_list) == len(images)
    fig = plt.figure(figsize=fig_size)
    grid = Grid(fig, rect=111, nrows_ncols=nrows_ncols, axes_pad=0.25, label_mode='L', share_all=True)
    for i, (ax, img) in enumerate(zip(grid, images)):
        if cmap is not None:
            if v_range is not None and v_range[i] is not None:
                ax.imshow(img, cmap[i], vmin=v_range[i][0], vmax=v_range[i][1])
            else:
                ax.imshow(img, cmap[i])
        else:
            ax.imshow(img)
        if not show_axis:
            ax.axis('off')
        if title_list:
            ax.set_title(title_list[i])
    plt.tight_layout()
    if show_fig:
        plt.show()


def compare_bars(data, labels, xticks, x_axis_name=None, y_axis_name=None, save_name=None, width=0.3, figsize=(8, 6),
                 show_text=True, font_size=8, legend_loc='upper right', x_offset=-0.12, y_offset=0.0, ylim=None,
                 num_fmt='{:.2f}', show_fig=True):
    """
    Create grouped bar plots with legends and value displayed on top of each bar
    :param data: data to be displayed, should be a 2D np array, each row has a legend and column occupies one xtick
    :param labels: the labels to be displayed in the legends, should be the same as the row number of data
    :param xticks: the xticks text, should be the same as the column number of data
    :param x_axis_name: the name of the x axis
    :param y_axis_name: the name of the y axis
    :param save_name: if not None, the figure will be saved to this path
    :param width: the width of each bar
    :param figsize: the size of the figure
    :param show_text: if True, value will be displayed on top of each bar
    :param font_size: the size of the font for value texts, only works when show_text=True
    :param legend_loc: the location of the legends, see plt legends for available options
    :param x_offset: horizontal offsets of value text, only works when show_text=True
    :param y_offset: vertical offsets of value text, only works when show_text=True
    :param ylim: if None, set the y axis range by ylim, should be a tuple of (min, max)
    :param num_fmt: the format to display the numbers above the bars
    :param show_fig: if True, figure will be displayed, otherwise figure will be closed
    :return:
    """
    assert len(labels) == data.shape[0]
    assert len(xticks) == data.shape[1]

    x_len = data.shape[1]
    x = np.arange(x_len)

    plt.figure(figsize=figsize)
    for cnt in range(data.shape[0]):
        plt.bar(x+width*cnt, data[cnt, :], width, label=labels[cnt])
        if show_text:
            for cnt_t, d in enumerate(data[cnt, :]):
                plt.text(x[cnt_t]+width*cnt+x_offset, d+y_offset, num_fmt.format(d), fontsize=font_size)
    plt.xticks(x+width*(len(labels)/2-1/2), xticks)
    if ylim is not None:
        plt.ylim(ylim)

    if x_axis_name is not None:
        plt.xlabel(x_axis_name)
    if y_axis_name is not None:
        plt.ylabel(y_axis_name)

    plt.legend(loc=legend_loc)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)

    if show_fig:
        plt.show()
    else:
        plt.close()


def check_img(img, show_axis=False, show_stats=True, cmap=None, fig_size=(8, 6)):
    """
    Quickly check an given image
    :param img: input image, should be h*w*3 or h*w
    :param show_axis: if True, axis will be displayed
    :param show_stats: if True, the min and max will be displayed on top of the image
    :param cmap: the color map that will be used
    :param fig_size: the figure size of (width, height)
    :return:
    """
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    if not show_axis:
        plt.axis('off')
    if cmap is not None:
        plt.set_cmap(cmap)
    if show_stats:
        plt.title(f'vmin={np.min(img)}, vmax={np.max(img)}')
    plt.tight_layout()
    plt.show()


def overlay_bboxes(bboxes, ax=None, linewidth=1, edgecolor='r', facecolor='none'):
    """
    Add bounding boxes to the current figure
    :param bboxes: list of bounding boxes, each element should in range of [xmin, ymin, xmax, ymax]
    :param ax: the axes to add bouding boxes, if None, the current axes will be used
    :param linewidth: the line width of the bboxes to be drawn
    :param edgecolor: the edge color of the bboxes to be drawn
    :param facecolor: the face color of the bboxes to be drawn, default is None
    :return:
    """
    if ax is None:
        ax = plt.gca()

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)


def overlay_polygons(polygons, ax=None, linewidth=1, edgecolor='r', facecolor='none'):
    """
    Add polygons to the current figure
    :param bboxes: list of bounding boxes, each element should in range of [xmin, ymin, xmax, ymax]
    :param ax: the axes to add bouding boxes, if None, the current axes will be used
    :param linewidth: the line width of the bboxes to be drawn
    :param edgecolor: the edge color of the bboxes to be drawn
    :param facecolor: the face color of the bboxes to be drawn, default is None
    :return:
    """
    if ax is None:
        ax = plt.gca()

    for poly in polygons:
        x = [a[0] for a in poly]
        y = [a[1] for a in poly]
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)


if __name__ == '__main__':
    pass
