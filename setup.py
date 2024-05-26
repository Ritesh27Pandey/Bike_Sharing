from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="RegressorProject",
    version='0.0.1',
    author="pandey",
    author_email='pandey@gmail.com',
    install_requires=['numpy','pandas'],
    packages= find_packages()
)