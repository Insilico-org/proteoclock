from setuptools import setup, find_packages

# Add error handling for README opening
try:
    long_description = open("README.md").read()
except FileNotFoundError:
    long_description = "A Python package for proteomic aging clock analysis"

setup(
    name="proteoclock",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0",
        "pandas>=2.2",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "plotly>=6.0",
        "scikit-learn>=1.5",
        "scikit-posthocs==0.11",
        "scipy>=1.7.0",
        "torch>=2.0",  # PyTorch renamed from pytorch in pip
    ],
    author="Fedor Galkin",
    author_email="f.a.galkin@gmail.com",
    description="A Python package for proteomic aging clock analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/f-galkin/proteoclock",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    package_data={
        'proteoclock': [
            'materials/**/*',
            'materials/simple_clocks/**/*',
            'materials/deep_clocks/**/*',
            'materials/scalers/*',
            'materials/test_data/**/*',
        ],
    },
    # Commented out due to missing CLI module
    # entry_points={
    #     'console_scripts': [
    #         'proteoclock-info=proteoclock.cli:main',
    #     ],
    # }
)
