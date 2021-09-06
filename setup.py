from setuptools import setup, find_packages, find_namespace_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Intended Audience :: Developers',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: BSD License'
]

setup(
    name='augmented',
    version='2.1.0',
    description='Augmented reality in python made easy',
    long_description=open('README.txt').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    long_description_content_type="text/markdown",
    url='https://www.github.com/sarangt123/augmented',
    author='Sarang T (github.com/sarangt123)',
    author_email='sarang.thekkedathpr@gmail.com',
    license='BSD-3-Clause License',
    classifiers=classifiers,
    keywords='Augmented reality',
    install_requires=["numpy==1.19.5", "opencv-contrib-python==4.5.3.56"],
    packages=find_packages(),
    python_requires=">=3.6"

)
