import os
from typing import List

import setuptools


def get_long_description() -> str:
    with open('Readme.md') as fh:
        return fh.read()


def get_required() -> List[str]:
    with open('requirements.txt') as fh:
        return fh.read().splitlines()


def get_version():
    with open(os.path.join('attention_to_gif', '__init__.py')) as fh:
        for line in fh:
            if line.startswith('__version__ = '):
                return line.split()[-1].strip().strip("'")


setuptools.setup(
    name='attention_to_gif',
    packages=setuptools.find_packages(),
    version=get_version(),
    license='MIT',
    description='Create A Gif of The Attention From Various Layers in a Transformer',
    author='Valay Dave',
    include_package_data=True,
    author_email='valaygaurang@gmail.com',
    url='https://github.com/valayDave/attention_to_gif.git',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    keywords=[],
    install_requires=get_required(),
    python_requires='>=3.6',
    py_modules=['attention_to_gif', ],
)
