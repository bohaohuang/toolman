"""

"""


# Built-in
import os
import shutil

# Libs
import pytest
import numpy as np

# Own modules
from toolman import misc_utils


@pytest.mark.parametrize('s, sep, d_type, l_len', [
    ('a_b_c_d_e', '_', str, 5),
    ('12 3 119 2', ' ', int, 4),
    ('a 13_b 14', '_', str, 2)
])
def test_str2list(s, sep, d_type, l_len):
    l = misc_utils.str2list(s, sep, d_type)
    assert len(l) == l_len


@pytest.mark.parametrize('ext', ['npy', 'pkl'])
def test_io_funcs(ext):
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    data = np.random.random((512, 512, 3))
    save_name = os.path.join(test_dir, f'dat.{ext}')
    misc_utils.save_file(save_name, data)
    assert os.path.exists(save_name)

    data_read = misc_utils.load_file(save_name)
    np.testing.assert_array_almost_equal(data_read, data)

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('s', [
    'abcdefg',
    'a\nb\nc\n',
    ['abc', 'def', 'gh']
])
def test_io_funcs_txt(s):
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    save_name = os.path.join(test_dir, f'dat.txt')
    misc_utils.save_file(save_name, s)
    assert os.path.exists(save_name)

    data_read = misc_utils.load_file(save_name)
    assert ''.join(data_read) == ''.join(s)

    shutil.rmtree(test_dir)


def test_io_funcs_json():
    temp = 'c'
    d = {
        'a': 123,
        'b': 'abc',
        temp: [1, 2, 3]
    }

    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    save_name = os.path.join(test_dir, f'dat.json')
    misc_utils.save_file(save_name, d)
    assert os.path.exists(save_name)

    data_read = misc_utils.load_file(save_name)
    assert data_read == d

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('ext', ['png', 'tif'])
def test_io_funcs_image(ext):
    img = np.random.randint(0, 256, (512, 512, 3))

    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    save_name = os.path.join(test_dir, f'dat.{ext}')
    misc_utils.save_file(save_name, img)
    assert os.path.exists(save_name)

    data_read = misc_utils.load_file(save_name)
    np.testing.assert_array_almost_equal(data_read, img)

    shutil.rmtree(test_dir)


def test_io_funcs_image_pil():
    img = np.random.randint(0, 256, (512, 512, 3))

    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    save_name = os.path.join(test_dir, 'dat.png')
    misc_utils.save_file(save_name, img)
    assert os.path.exists(save_name)

    data_read = misc_utils.load_file(save_name, pil=True)
    np.testing.assert_array_almost_equal(np.array(data_read), img)

    data_read = misc_utils.load_file(save_name, pil=True, to_numpy=True)
    np.testing.assert_array_almost_equal(data_read, img)

    shutil.rmtree(test_dir)


def test_get_file_length():
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)

    for i in range(10):
        length = np.random.randint(100, 501)
        text = []
        for _ in range(length):
            text.append('abcdefg\n')

        save_name = os.path.join(test_dir, 'dat.txt')
        misc_utils.save_file(save_name, text)
        assert misc_utils.get_file_length(save_name) == length

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('array', [np.arange(100), ['a', 10, None, ('b', 2), 11, 12, 13, 14, 15, 16]])
@pytest.mark.parametrize('portions', [(0.1, 0.2), (0.1, )])
def test_randomsplit_fill(array, portions):
    final = misc_utils.random_split(array, portions)
    assert len(final) == len(portions) + 1
    for cnt in range(len(portions)):
        assert len(final[cnt]) == int(len(array) * portions[cnt])


@pytest.mark.parametrize('array', [np.arange(100), ['a', 10, None, ('b', 2), 11, 12, 13, 14, 15, 16]])
@pytest.mark.parametrize('portions', [(0.1, 0.2, 0.7), (0.1, 0.9)])
def test_randomsplit_all(array, portions):
    final = misc_utils.random_split(array, portions)
    assert len(final) == len(portions)
    for cnt in range(len(portions)):
        assert len(final[cnt]) == int(len(array) * portions[cnt])


@pytest.mark.parametrize('array', [np.arange(100), ['a', 10, None, ('b', 2), 11, 12, 13, 14, 15, 16]])
@pytest.mark.parametrize('portions', [(0.1, 0.2, 0.8), (0.1, -0.5)])
def test_randomsplit_error(array, portions):
    with pytest.raises(ValueError):
        misc_utils.random_split(array, portions)
