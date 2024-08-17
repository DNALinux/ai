from setuptools import setup, find_packages

# Read the contents of your README file
def read_readme():
    with open('pkg.md', 'r') as f:
        return f.read()

# Define the setup function
setup(
    name='rag_llama3',  # Replace with your application's name
    version='0.1.2',
    author='Songlin Zhao',
    author_email='tagorezhao@berkeley.edu',  # Replace with your email address
    description='This repository contains a project that implements a Retrieval-Augmented Generation (RAG) system using the LLaMA3 model. The project focuses on creating embeddings for instructions of a professional bioinformatic software to help users conduct biology research.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/DNALinux/ai',  # Replace with your repository URL
    packages=find_packages(where='/home/tagore/repos/ai/src'),
    package_dir={'': 'src'},
    install_requires=[
        'ollama==0.2.1',
        'numpy',
        'chromadb==0.5.3',
        'pypdf==4.2.0',
        'beautifulsoup4==4.12.2',
        'nltk==3.6.2',
        'langchain-community==0.2.6'

    ],
    python_requires='>=3.10.0, <4', # Adjust based on the minimum Python version your project supports
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'rag-llama3 = src.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
