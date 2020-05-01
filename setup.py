from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='toolman',
      version='0.1a',
      description='Python utility tools for research',
      url='https://github.com/bohaohuang/toolman',
      author='bohaohuang',
      author_email='hbhzhuce@gmail.com',
      license='MIT',
      packages=['toolman'],
      install_requirements=[
            'numpy',
            'Pillow',
            'scikit-learn',
            'scikit-image',
            'natsort'
            'matplotlib',
            'torchsummary',
      ],
      zip_safe=False)
