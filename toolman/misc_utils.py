# Built-in
import os
import time
import json
import pickle
import collections.abc
from glob import glob
from functools import wraps

# Libs
import numpy as np
from PIL import Image
from skimage import io
from natsort import natsorted

# Own modules


def make_dir_if_not_exist(dir_path):
    """
    Make the directory if it does not exists
    :param dir_path: absolute path to the directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def timer_decorator(func):
    """
    This is a decorator to print out running time of executing func
    :param func:
    :return:
    """
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time
        print('duration: {:.3f}s'.format(duration))
    return timer_wrapper


def str2list(s, sep=',', d_type=int):
    """
    Change a {sep} separated string into a list of items with d_type
    :param s: input string
    :param sep: separator for string
    :param d_type: data type of each element
    :return:
    """
    if type(s) is not list:
        s = [d_type(a) for a in s.split(sep)]
    return s


def load_file(file_name, **kwargs):
    """
    Read data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :return: file data, or IOError if it cannot be read by either numpy or pickle or imageio
    """
    try:
        if file_name[-3:] == 'npy':
            data = np.load(file_name)
        elif file_name[-3:] == 'pkl' or file_name[-6:] == 'pickle':
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'r') as f:
                data = f.readlines()
        elif file_name[-3:] == 'csv':
            data = np.genfromtxt(file_name, delimiter=',', dtype=None, encoding='UTF-8')
        elif file_name[-4:] == 'json':
            data = json.load(open(file_name))
        elif 'pil' in kwargs and kwargs['pil']:
            try:
                data = Image.open(file_name)
            except Image.DecompressionBombError:
                Image.MAX_IMAGE_PIXELS = None
                data = Image.open(file_name)
        else:
            try:
                data = io.imread(file_name)
            except Image.DecompressionBombError:
                Image.MAX_IMAGE_PIXELS = None
                data = io.imread(file_name)
            except ValueError or OSError:
                data = np.array(Image.open(file_name).convert('RGB'))

        return data
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem loading {}'.format(file_name))


def save_file(file_name, data, fmt='%.8e', sort_keys=True, indent=4):
    """
    Save data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :param data: data to save
    :return: file data, or IOError if it cannot be saved by either numpy or or pickle imageio
    """
    try:
        if file_name[-3:] == 'npy':
            np.save(file_name, data)
        elif file_name[-3:] == 'pkl':
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'w') as f:
                f.writelines(data)
        elif file_name[-3:] == 'csv':
            np.savetxt(file_name, data, delimiter=',', fmt=fmt)
        elif file_name[-4:] == 'json':
            json.dump(data, open(file_name, 'w'), sort_keys=sort_keys, indent=indent)
        else:
            data = Image.fromarray(data.astype(np.uint8))
            data.save(file_name)
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem saving this data')


def rotate_list(l):
    """
    Rotate a list of lists
    :param l: list of lists to rotate
    :return:
    """
    return list(map(list, zip(*l)))


def make_center_string(char, length, center_str=''):
    """
    Make one line decoration string that has center_str at the center and surrounded by char
    The total length of the string equals to length
    :param char: character to be repeated
    :param length: total length of the string
    :param center_str: string that shown at the center
    :return:
    """
    return center_str.center(length, char)


def float2str(f):
    """
    Return a string for float number and change '.' to character 'p'
    :param f: float number
    :return: changed string
    """
    return '{}'.format(f).replace('.', 'p')


def stem_string(s, lower=True):
    """
    Return a string that with spaces at the begining or end removed and all casted to lower cases
    :param s: input string
    :param lower: if True, the string will be casted to lower cases
    :return: stemmed string
    """
    if lower:
        return s.strip().lower()
    else:
        return s.strip()


def remove_digits(s):
    """
    Remove digits in the given string
    :param s: input string
    :return: digits removed string
    """
    return ''.join([c for c in s if not c.isdigit()])


def get_digits(s):
    """
    Get digits in the given string, cast to int
    :param s: input string
    :return: int from string
    """
    return int(''.join([c for c in s if c.isdigit()]))


def args_getter(inspect_class):
    """
    Inspect parameters inside a class
    :param inspect_class: the class to be inspected
    :return: a dict of key value pairs of all parameters in this class
    """
    params = {}
    for k, v in inspect_class.__dict__.items():
        if not k.startswith('__'):
            params[k] = v
    return params


def args_writer(file_name, inspect_class):
    """
    Save parameters inside a class into json file
    :param file_name: path to the file to be saved
    :param inspect_class: the class which parameters are going to be saved
    :return:
    """
    params = args_getter(inspect_class)
    save_file(file_name, params, sort_keys=True, indent=4)


def get_file_name_no_extension(file_name):
    """
    Handy function for getting the file name without path and extension
    :param file_name: the name or path to the file
    :return: filename without extension
    """
    return os.path.splitext(os.path.basename(file_name))[0]


def get_files(path_list, extension):
    """
    Get files in the given folder that matches certain regex
    :param path_list: list of path to the directory
    :param extension: regex that filters the desired files
    :return: list of files
    """
    if isinstance(path_list, str):
        path_list = [path_list]
    return natsorted(glob(os.path.join(*path_list, extension)))


def recursive_update(d, u):
    """
    Recursively update nested dictionary d with u
    :param d: the dictionary to be updated
    :param u: the new dictionary
    :return:
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_args(arg_list):
    """
    Parse the arguments in a recursive way
    :param arg_list: the arguments in a list where each element is either key or val
    :return: dictionary of arguments
    """
    def parse_args_helper(arg_l):
        """
        Recursively calling itself if it's a key or return the value otherwise
        :param arg_l: argument list
        :return:
        """
        while len(arg_l) > 0:
            item = arg_list.pop(0)
            if '--' in item:
                return {item[2:]: parse_args_helper(arg_l)}
            else:
                try:
                    return float(item)
                except ValueError:
                    return item

    arg_dict = {}
    while len(arg_list) > 0:
        item = parse_args_helper(arg_list)
        recursive_update(arg_dict, item)
    return arg_dict


def change_channel_order(data, to_channel_last=True):
    """
    Switch the image type from channel first to channel last
    :param data: the data to switch the channels
    :param to_channel_last: if True, switch the first channel to the last
    :return: the channel switched data
    """
    if to_channel_last:
        if len(data.shape) == 3:
            return np.rollaxis(data, 0, 3)
        else:
            return np.rollaxis(data, 1, 4)
    else:
        if len(data.shape) == 3:
            return np.rollaxis(data, 2, 0)
        else:
            return np.rollaxis(data, 3, 1)
