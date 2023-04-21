from setuptools import find_packages, setup
setup(
    name='riskill',
    packages=find_packages(include=['riskill']),
    version='0.1.0',
    description='My first Python library for quantitative risk management',
    author='Me',
    license='MIT',
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
