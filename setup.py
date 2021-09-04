from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='ar-python',
    version='0.0.1',
    description='Augmented reality in python made easy',
    long_description=open('README.txt').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    url='',
    author='Sarang T (github.com/sarangt123)',
    author_email='sarang.thekkedathpr@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Augmented reality',
    packages=find_packages(),
    install_requires=['']
)
