from setuptools import setup, find_packages
import os

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="decode",
    version="0.0",
    include_package_data=True,
    python_requires='>=3.10',
    packages=find_packages(),
    setup_requires=['setuptools-git-versioning'],
    install_requires=requirements,
    author="José Cação",
    author_email="josemaria@ua.pt",
    description="DECODE: DEcision tree Capturing Opaque DEcisions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    version_config={
       "dirty_template": "{tag}",
    }
)