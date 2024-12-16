# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='lmcsc',
    version='1.0.0',
    author='Houquan Zhou',
    author_email='Jacob_Zhou@outlook.com',
    license='MIT',
    description='Turn a causal language model into a Chinese Spelling Corrector',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Jacob-Zhou/llm-csc',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'pypinyin',
        'pypinyin-dict',
        'torch>=2.0.1',
        'transformers>=4.27.0',
        'sentencepiece',
        'numpy==1.24.4',
        'accelerate',
        'bitsandbytes',
        'modelscope',
        'opencc-python-reimplemented',
        'streamlit',
        'uvicorn',
        'fastapi',
        'loguru',
        'sse_starlette',
        'starlette>=0.40.0,<0.42.0'
    ],
    extras_require={},
    python_requires='>=3.7',
    zip_safe=False,
    include_package_data=True
)