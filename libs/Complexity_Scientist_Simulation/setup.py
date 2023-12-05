from setuptools import setup, find_packages

setup(
    name='CS_Simulation',
    version='0.1.0',
    author="Munjung Kim",
    author_email='jennykim7369@gmail.com',
    url="https://github.com/MunjungKim/Complexity_Science",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(include=['CS_Simulation', 'CS_Simulation.*'])
)