from setuptools import setup, find_packages

setup(
    name="urban_point_cloud_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "open3d",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "wandb",
        "tensorboard",
        "laspy",
        "pyproj",
        "shapely",
        "geopandas",
        "pandas",
        "dash",
        "plotly",
    ],
    python_requires=">=3.8",
)
