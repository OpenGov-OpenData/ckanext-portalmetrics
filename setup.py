# -*- coding: utf-8 -*-
from setuptools import setup, find_namespace_packages

version = '0.0.1'

setup(
    name='ckanext-portalmetrics',
    version=version,
    description="Plugin that brings Google Analytics 4-based usage statistics into the CKAN-portal itself",
    long_description='''\
    ''',
    classifiers=[],
    keywords='',
    author='Peter Vorman',
    author_email='pvorman@opengov.com',
    url='https://github.com/OpenGov-OpenData/ckanext-portalmetrics',
    license='AGPL',
    packages=find_namespace_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # -*- Extra requirements: -*-
    ],
    entry_points='''

    [ckan.plugins]
    portal_metrics=ckanext.portal_metrics.plugin:MetricsCliPlugin

    ''',
    message_extractors={
        'ckanext': [
            ('**.py', 'python', None),
            ('**.js', 'javascript', None),
            ('**/templates/**.html', 'ckan', None),
        ],
    },
)
