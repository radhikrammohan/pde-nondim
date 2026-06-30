from setuptools import setup, find_packages

setup(
    name="pde-nondim",
    version="0.5.0",
    description="Symbolic non-dimensionalisation of PDEs with dimensionless group identification and PINN code generation",
    author="Radhik Rammohan",
    author_email="radhikrammohan@gmail.com",
    url="https://github.com/radhikrammohan/pde-nondim",
    packages=find_packages(),
    install_requires=["sympy>=1.10"],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=["PDE", "non-dimensionalisation", "dimensional analysis", "SymPy", "PINN", "Buckingham Pi"],
)
