from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def main():
    package_name = 'mlpt'
    packages = find_packages(where=package_name)
    # Если хотим установить пакеты так, чтобы при импорте использовался просто "import <subpackage>"
    setup(
        name=package_name,
        version='0.0.1',
        author='HASKII',
        description='PPE-Detection package',
        package_dir={"": package_name},
        packages=find_packages(where=package_name),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=[str(req) for req in parse_requirements(
            open('requirements.txt'))],
    )


if __name__ == '__main__':
    main()
