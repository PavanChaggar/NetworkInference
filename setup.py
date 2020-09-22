from setuptools import setup, find_packages

setup(
    name='netwin',
    version='0.1',
    description='A toolbox for performing modelling and inference on networks',
    license='MIT license',
    maintainer='Pavan Chaggar',
    maintainer_email='pavanjit.chaggar@maths.ox.ac.uk',
    include_package_data=True,
    packages = find_packages(include=('netwin', 'netwin.*')),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'networkx',
        'pandas',
        'plotly',
        'torch',
        'sbi'
    ],
)