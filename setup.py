import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements


def load_requirements(file_name):
    requirements = parse_requirements(file_name, session="test")
    return [str(item.req) for item in requirements]


setup(
    name="flopco-pytorch",
    version="v0.1.0",
    description="FLOPs and other statistics COunter for Pytorch neural networks",
    author="Julia Gusak",
    author_email="julgusak@gmail.com",
    url="https://github.com/juliagusak/flopco-pytorch",
    download_url="https://github.com/juliagusak/flopco-pytorch/archive/v0.1.0.tar.gz",
    keywords = ['pytorch', 'flops', 'macs', 'neural-networks', 'cnn'],
    license="MIT",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt")
)
