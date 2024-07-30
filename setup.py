import setuptools


setuptools.setup(
    name="ivac",
    version="0.0.0",
    author="Chatipat Lorpaiboon",
    author_email="chatipat@uchicago.edu",
    license="MIT",
    description="Spectral estimation for Markov processes using the Integrated VAC algorithm.",
    url="https://github.com/chatipat/ivac",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "torch",
        "lightning",
    ],
)
