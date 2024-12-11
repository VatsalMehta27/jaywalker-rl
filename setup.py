from setuptools import setup, find_packages

setup(
    name="jaywalker-rl",
    version="0.1",
    description="Jaywalker Reinforcement Learning Project",
    url="https://github.com/VatsalMehta27/jaywalker-rl",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "gymnasium",
        "matplotlib",
        "pre-commit",
        "torch",
        "opencv-python",
        "tqdm",
        "setuptools",
        "captum",
    ],
)
