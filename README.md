# ToolMan: Python utility functions for R&D
See the source for this project [here](https://github.com/bohaohuang/toolman)
## Modules:
### *[misc_utils.py](./toolman/misc_utils.py)*: 
miscellaneous utility functions including data I/O and processing  
a) Read/write different formats of files in one function:
```python
import toolman as tm
data = tm.misc_utils.load_file(file_name)
tm.misc_utils.save_file(file_name, data)
```
Currently support extensions including: `.npy`, `.pkl`, `.txt`, `.csv`, `.json` and commonly used image formats.  
b) Argument parser, parse nested argument list:
```python
import sys
import argparse
import toolman as tm
parser = argparse.ArgumentParser()
args, extras = parser.parse_known_args(sys.argv[1:])
cfg_dict = tm.misc_utils.parse_args(extras)
```

### *[vis_utils.py](./toolman/vis_utils.py)*: 
Matplotlib utility functions for visualization  
a) Display images in side by side with axis linked
```python
import toolman as tm
fig1 = tm.misc_utils.load_file(img_name_1)
fig2 = tm.misc_utils.load_file(img_name_1)
tm.vis_utils.compare_figures([fig1, fig2], (1, 2), fig_size=(12, 5))
```
### *[img_utils](./toolman/img_utils.py)*: 
image specific utility functions

### *[pytorch_utils](./toolman/pytorch_utils.py)*: 
pytorch specific utility functions

### *[process_block](./toolman/process_block.py)*: 
A processing unit that do certain operations only if it has never done before. This is helpful avoid duplicate 
executing time consuming jobs.
```python
import toolman as tm
def foo(cnt_len):
    cnt = 0
    for i in range(cnt_len):
        cnt += 1
    return cnt

pb = tm.process_block.ProcessBlock(foo, file_dir)
pb.run(force_run=False, cnt_len=100)
```
