<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<br />
<div align="center">
  <h2>
    SignalGrad-CAM
  </h2>

  <h3 align="center">SignalGrad-CAM aims at generalising Grad-CAM to one-dimensional applications, while enhancing usability and efficiency.</h3>

  <p align="center">
    <a href="https://github.com/samuelepe11/signal-grad-cam"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/samuelepe11/signal-grad-cam/issues">Report Bug</a>
    ·
    <a href="https://github.com/samuelepe11/signal-grad-cam/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#publications">Publications</a></li>
    <li><a href="#contacts-and-useful-links">Contacts And Useful Links</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Deep learning models have demonstrated remarkable performance across various domains; however, their black-box nature hinders interpretability and trust. As a result, the demand for explanation algorithms has grown, driving advancements in the field of eXplainable AI (XAI). However, relatively few efforts have been dedicated to developing interpretability methods for signal-based models. We introduce SignalGrad-CAM (SGrad-CAM), a versatile and efficient interpretability tool that extends the principles of Grad-CAM to both 1D- and 2D-convolutional neural networks for signal processing. SGrad-CAM is designed to interpret models for either image or signal elaboration, supporting both PyTorch and TensorFlow/Keras frameworks, and provides diagnostic and visualization tools to enhance model transparency. The package is also designed for batch processing, ensuring efficiency even for large-scale applications, while maintaining a simple and user-friendly structure.

**Keywords**: *explainable AI, explanations, local explanation, fidelity, interpretability, transparency, trustworthy AI, feature importance, saliency maps, CAM, Grad-CAM, black-box, deep learning, CNN, signals, time series*

<p align="right"><a href="#top">↰ Back To Top</a></p>

<!-- INSTALLATION -->
## Installation

1. Make sure you have the latest version of pip installed
   ```sh
   pip install --upgrade pip
    ```
2. Install araucanaxai through pip
    ```sh
    pip install signal-grad-cam
    ```

<p align="right"><a href="#top">↰ Back To Top</a></p>

<!-- USAGE EXAMPLES -->
## Usage

Here's a basic example with a built-in toy dataset that illustrates SignalGrad-CAM common usage.

You can also check the notebooks [here](https://github.com/samuelepe11/signal-grad-cam/blob/master/example.ipynb).

See the [open issues](https://github.com/samuelepe11/signal-grad-cam/issues) for a full list of proposed features (and known issues).

<p align="right"><a href="#top">↰ Back To Top</a></p>


If you use the AraucanaXAI software for your projects, please cite it as:

```
@software{Pe_SignalGrad_CAM_2025,
  author = {Pe, Samuele and Buonocore, Tommaso Mario and Giovanna, Nicora and Enea, Parimbelli},
  month = {3},
  title = {{SignalGrad-CAM}},
  url = {https://github.com/samuelepe11/signal-grad-cam},
  version = {1.0.0},
  year = {2025}
}
```

<p align="right"><a href="#top">↰ Back To Top</a></p>

<!-- CONTACTS AND USEFUL LINKS -->
## Contacts and Useful Links

*   **Repository maintainer**: Samuele Pe[![Gmail][gmail-shield]][gmail-url] [![LinkedIn][linkedin-shield]][linkedin-url]  

*   **Project Link**: [https://github.com/samuelepe11/signal-grad-cam](https://github.com/samuelepe11/signal-grad-cam)

*   **Package Link**: [https://pypi.org/project/signal-grad-cam/](https://pypi.org/project/signal-grad-cam/)

<p align="right"><a href="#top">↰ Back To Top</a></p>

<!-- LICENSE -->
## License

Distributed under MIT License. See `LICENSE` for more information.


<p align="right"><a href="#top">↰ Back To Top</a></p>
<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/samuelepe11/signal-grad-cam.svg?style=for-the-badge
[contributors-url]: https://github.com/samuelepe11/signal-grad-cam/graphs/contributors
[status-shield]: https://img.shields.io/badge/Status-pre--release-blue
[status-url]: https://github.com/samuelepe11/signal-grad-cam/releases
[forks-shield]: https://img.shields.io/github/forks/samuelepe11/signal-grad-cam.svg?style=for-the-badge
[forks-url]: https://github.com/samuelepe11/signal-grad-cam/network/members
[stars-shield]: https://img.shields.io/github/stars/samuelepe11/signal-grad-cam.svg?style=for-the-badge
[stars-url]: https://github.com/samuelepe11/signal-grad-cam/stargazers
[issues-shield]: https://img.shields.io/github/issues/samuelepe11/signal-grad-cam.svg?style=for-the-badge
[issues-url]: https://github.com/samuelepe11/signal-grad-cam/issues
[license-shield]: https://img.shields.io/github/license/samuelepe11/signal-grad-cam.svg?style=for-the-badge
[license-url]: https://github.com/samuelepe11/signal-grad-cam/LICENSE
[linkedin-shield]: 	https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/samuele-pe-818bbb307
[gmail-shield]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: mailto:samuele.pe01@universitadipavia.it
