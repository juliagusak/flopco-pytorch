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

version="v0.1.1"
setup(
    name="flopco-keras",
    version=version,
    description="FLOPs and other statistics COunter for TF.keras neural networks",
    author="Evgeny Ponomarev (based on Julia Gusak's work)",
    author_email="evgps@ya.ru",
    url="https://github.com/evgps/flopco-keras",
    download_url=f"https://github.com/evgps/flopco-keras/archive/{version}.tar.gz",
    keywords = ['tensorflow', 'keras', 'flops', 'macs', 'neural-networks', 'cnn'],
    license="MIT",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt")
)
