import os
import setuptools



setuptools.setup(
    name = "text_mining",
    version = "0.0.1",
    author = "Sai Swetha Pasam",
    author_email = "swethapasam@gmail.com",
    description = ("An demonstration of how to create, document, and publish "
                                   "to the cheese shop a5 pypi.org."),
    # license = "BSD",
    # keywords = "example documentation tutorial",
    # url = "http://packages.python.org/an_example_pypi_project",
    packages=setuptools.find_packages(),
    # long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)