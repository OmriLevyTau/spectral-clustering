from gettext import install
from setuptools import setup, find_packages, Extension

setup(
    name="spkmeansmodule",
    version="0.1.0",
    author="Idan and Omri",
    author_email="omrilevy@mail.tau.ac.il",
    description="spkmeans capi",
    install_requires = ['invoke'],
    packages=find_packages(),
    headers=['spkmeans.h'],
    license="GPL-2",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension(
            'spkmeansmodule',
            ['spkmeansmodule.c'],
        ),
    ]
)
