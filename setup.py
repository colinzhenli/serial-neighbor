from setuptools import setup, find_packages

setup(
    name="serial-neighbor",
    version="0.1.0",
    description="Serial order neighbor search module using spatial encoding.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zhen Li",
    author_email="zla247@sfu.ca",
    url="https://github.com/colinzhenli/serial-neighbor",
    packages=["serial_neighbor"],
    license="Apache-2.0",
    install_requires=[
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
