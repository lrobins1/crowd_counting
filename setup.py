from setuptools import setup, find_packages

setup(
    name='Crowd_counting',
    version='0.6',
    packages=find_packages(exclude=['tests*']),
    package_dir={'Crowd_counting': 'Crowd_counting'},
    package_data={'Crowd_counting': ['data/*.tar']},
    include_package_data=True,
    license='MIT',
    description='A package for crowd-counting in Uclouvain auditorium',
    long_description=open('README.md').read(),
    install_requires=['numpy','tensorflow','scipy','h5py'],
    url='https://github.com/lrobins1/crowd_counting',
    author='Henri Collin & Louis Robins',
    author_email='henri.collin@student.uclouvain.be & louis.robins@tudent.uclouvain.be'
)
