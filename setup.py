from setuptools import setup, find_packages

def pull_first():
    """This script is in a git directory that can be pulled."""
    cwd = os.getcwd()
    gitdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(gitdir)
    g = git.cmd.Git(gitdir)
    try:
        g.execute(['git', 'lfs', 'pull'])
    except git.exc.GitCommandError:
        raise RuntimeError("Make sure git-lfs is installed!")
    os.chdir(cwd)

pull_first()

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
