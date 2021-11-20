from setuptools import find_packages
from setuptools import setup

setup(
    name="sort_sieu_xin",
    version="1.0",
    author="inspiros",
    author_email='hnhat.tran@gmail.com',
    description="SORT này chắc là xịn hơn SORT thường",
    packages=find_packages(exclude=("test",)),
    install_requires=["numpy", "scipy", "filterpy"],
)
