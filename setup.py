from setuptools import setup, find_packages

setup(
    name='Crowd-counting',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='LGPL',
    description='A package for crowd-counting in Uclouvain auditorium',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/lrobins1/crowd_counting',
    author='Henri Collin & Louis Robins',
    author_email='henri.collin@student.uclouvain.be & louis.robins@tudent.uclouvain.be'
)
