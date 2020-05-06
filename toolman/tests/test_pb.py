"""

"""


# Built-in
import os
import shutil

# Libs
import pytest
import numpy as np

# Own modules
from toolman import process_block, misc_utils


@pytest.mark.parametrize('shape', [
    (512, 512, 3),
    (512, 512),
])
@pytest.mark.parametrize('ext', [
    'jpg',
    'png',
    'npy',
])
def test_pb(shape, ext):
    def func(shape):
        return np.random.randint(0, 256, shape)

    test_dir = './temp_pb'
    misc_utils.make_dir_if_not_exist(test_dir)
    assert os.path.exists(test_dir)
    save_name = os.path.join(test_dir, f'dat.{ext}')

    pb = process_block.ProcessBlock(func, save_name)
    img = pb.run(shape=shape)
    np.testing.assert_array_equal(shape, img.shape)

    shutil.rmtree(test_dir)
