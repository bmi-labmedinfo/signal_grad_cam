from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="signal_grad_cam",
    version="2.0.4",
    description="SignalGrad-CAM aims at generalising Grad-CAM to one-dimensional applications, while enhancing usability"
                " and efficiency.",
    keywords="XAI, class activation maps, CNN, time series",
    author="Samuele Pe",
    author_email="samuele.pe01@universitadipavia.it",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelepe11/signal_grad_cam",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "torch",
        "keras",
        "tensorflow",
        "imageio"
    ],
    include_package_data=True,
    zip_safe=False
)
