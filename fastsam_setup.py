from setuptools import setup, find_packages

setup(
    name='fastsam',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.2.2',
        'opencv-python>=4.6.0',
        'Pillow>=7.1.2',
        'PyYAML>=5.3.1',
        'requests>=2.23.0',
        'scipy>=1.4.1',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tqdm>=4.64.0',
        'pandas>=1.1.4',
        'seaborn>=0.11.0',
        'gradio==3.35.2',
        'ultralytics==8.0.120'
    ],
    package_data={
        '': ['*.*', '**/*'],
        'bpe': ['*.*', '**/*'],
        'models': ['*.*', '**/*'],
        '.assets': ['*.*', '**/*'],
    },
)