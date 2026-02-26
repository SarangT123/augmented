from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Intended Audience :: Developers',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: BSD License'
]

setup(
    name='augmented',
    version='2.4.2',
    description='Augmented reality in python made easy',
    long_description=open('README.md', encoding='utf-8').read() + '\n\n' +
                     open('CHANGELOG.txt', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://www.github.com/sarangt123/augmented',
    author='Sarang T (github.com/sarangt123)',
    author_email='sarang.thekkedathpr@gmail.com',
    license='BSD-3-Clause',
    classifiers=classifiers,
    keywords='augmented reality',
    install_requires=[
        "numpy>=1.21.0",
        "opencv-contrib-python>=4.8.0"
    ],
    packages=find_packages(),
    python_requires=">=3.9"
)