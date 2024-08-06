from setuptools import setup, find_packages

# Read the contents of your README file
def read_readme():
    with open('README.md', 'r') as f:
        return f.read()

# Define the setup function
setup(
    name='Toyoko',  # Replace with your application's name
    version='0.1.0',
    author='Songlin Zhao',
    author_email='tagorezhao@berkeley.edu',  # Replace with your email address
    description='This repository contains a project that implements a Retrieval-Augmented Generation (RAG) system using the LLaMA3 model. The project focuses on creating embeddings for instructions of a professional bioinformatic software to help users conduct biology research.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/DNALinux/ai',  # Replace with your repository URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
    ],
    python_requires='>=3.10',  # Adjust based on the minimum Python version your project supports
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'myapp=src.main:main',  # Replace with your application's entry point
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
