import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genfact-swarna.kpaul",
    version="0.0.1",
    author="Swarna Kamal Paul",
    author_email="swarna.kpaul@gmail.com",
    description="A model agnostic and gradient-free optimization method for generating counterfactuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swarna-kpaul/genfact",
    project_urls={
        "Bug Tracker": "https://github.com/swarna-kpaul/genfact/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
	keywords = ['counterfactual', 'genetic algorithm', 'genfact'],
	install_requires=[ 'sklearn','pkg_resources','pandas','pickle','logging','scipy','numpy','random'],
)