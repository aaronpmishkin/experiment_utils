import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="experiment_utils",
    version="0.0.1",
    author="Aaron Mishkin",
    author_email="amishkin@cs.stanford.edu",
    description="A lightweight package for managing optimization and machine learning experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronpmishkin/experiment_utils",
    project_urls={
        "Bug Tracker": "https://github.com/aaronpmishkin/experiment_utils/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    package_dir={"": "src"},
    package_data={"experiment_utils": ["py.typed"]},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

