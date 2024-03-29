from setuptools import setup, find_packages

setup(
    name="clustering-project",
    version="0.0.1",
    description="Implementation of K-means, K-means++, DBSCAN clustering algorithms",
    author="Illia Nasiri",
    packages=find_packages(),
    install_requires=["numpy"]
)

