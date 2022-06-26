from setuptools import setup, find_packages
setup(
    name='lm_2022_06',
    version='0.1.0',
    description='Hacky LM',
    long_description='This is a hacky transformer model based on RWKV',
    url='https://github.com/xloem/lm_2022_06',
    keywords=[],
    classifiers=[],   
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch'
    ]
)

