"""

"""


# Built-in

# Libs
import pytest
import numpy as np

# Own modules
from toolman import vis_utils


@pytest.mark.parametrize('shape', [
    (1, 512, 512, 1),
    (1, 512, 512),
])
@pytest.mark.parametrize('label_num', [3, 8])
def test_decode_label_map(shape, label_num):
    label_num = np.random.randint(2, label_num)
    mask = np.random.randint(0, label_num, shape)
    label_colors = np.array(vis_utils.get_color_list(), dtype=np.uint8)
    label_colors[0] = (255, 255, 255)

    label_map = vis_utils.decode_label_map(mask, label_num)
    for i in range(shape[0]):
        for j in range(shape[0]):
            if len(shape) == 4:
                np.testing.assert_array_equal(label_map[0, i, j], label_colors[mask[0, i, j, 0]])
            else:
                print(label_map.shape, mask.shape)
                np.testing.assert_array_equal(label_map[0, i, j], label_colors[mask[0, i, j]])
