from setuptools import find_packages,setup                               # Importing 'find_packages' and 'setup' functions from setuptools module.
                                                                          # 'setuptools' is a Python library used for packaging Python projects.
                                                                          # 'find_packages' automatically finds all sub-packages in the project directory.
                                                                          # 'setup' is used to define package metadata and installation configurations.

setup(                                                                    # Calling the setup() function to define the package information and configuration for distribution.
    name='Medical chatbot',                                               # The name of the package/project. This will be the name shown on PyPI if published.
    version='0.0.0',                                                      # The current version of the package. Helps in version tracking for updates.
    author='Abhijit Palai',                                               # The name of the author or maintainer of this package/project.
    author_email='pabhijit2021@gmail.com',                                # The email address of the author for contact or support.
    packages=find_packages(),                                             # Automatically detects all Python packages (directories with __init__.py) in the project folder.
                                                                          # This includes sub-packages so you don’t have to list them manually.
    install_requires=[]                                                   # A list of dependencies (external Python libraries) required by this project.
                                                                          # Currently, it's empty, meaning no extra packages are needed to run it.
)                                                                         # End of the setup() function call.
