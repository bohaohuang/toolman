"""

"""


# Built-in
import os
import shutil

# Libs
import pytest
import numpy as np

# Own modules
from toolman import misc_utils, img_utils


@pytest.mark.parametrize('shape', [
    (512, 512, 3, 3),
    (512, 512, 1),
])
@pytest.mark.parametrize('ext', [
    'jpg',
    'png',
    'npy',
])
def test_image_channel(shape, ext):
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)

    img = np.random.randint(0, 256, shape[:-1])
    save_name = os.path.join(test_dir, f'dat.{ext}')
    misc_utils.save_file(save_name, img)
    assert os.path.exists(save_name)

    assert img_utils.get_img_channel_num(save_name) == shape[-1]

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('shape', [
    (512, 512, 4, 4),
    (512, 512, 10, 10),
])
def test_image_channel_non_image(shape):
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)

    img = np.random.randint(0, 256, shape[:-1])
    save_name = os.path.join(test_dir, 'dat.npy')
    misc_utils.save_file(save_name, img)
    assert os.path.exists(save_name)

    assert img_utils.get_img_channel_num(save_name) == shape[-1]

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('shape', [
    (512, 512, 5, 3),
    (512,),
])
def test_image_channel_raise_error(shape):
    test_dir = './temp'
    misc_utils.make_dir_if_not_exist(test_dir)

    img = np.random.randint(0, 256, shape)
    save_name = os.path.join(test_dir, f'dat.npy')
    misc_utils.save_file(save_name, img)
    assert os.path.exists(save_name)

    with pytest.raises(ValueError, match=r'Image can only have 2 or 3 dimensions'):
        img_utils.get_img_channel_num(save_name)

    shutil.rmtree(test_dir)


@pytest.mark.parametrize('shape', [
    (512, 512, 5),
    (10, 10, 100),
    (12, 12, 12, 12),
    (12, 1, 1, 12)
])
def test_change_order(shape):
    data = np.random.random(shape)
    data_new = img_utils.change_channel_order(img_utils.change_channel_order(data), False)
    np.testing.assert_array_almost_equal(data, data_new)
