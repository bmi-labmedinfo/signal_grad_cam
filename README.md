<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url]


<br />
<div align="center">
  <h1>
    SignalGrad-CAM
  </h1>

  <h3 align="center">SignalGrad-CAM aims at generalising Grad-CAM to one-dimensional applications, while enhancing usability and efficiency.</h3>

  <p align="center">
    <a href="https://github.com/bmi-labmedinfo/signal_grad_cam"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="https://github.com/bmi-labmedinfo/signal_grad_cam/issues">Report Bug or Request Feature</a>
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

<p align="justify">Deep learning models have demonstrated remarkable performance across various domains; however, their black-box nature hinders interpretability and trust. As a result, the demand for explanation algorithms has grown, driving advancements in the field of eXplainable AI (XAI). However, relatively few efforts have been dedicated to developing interpretability methods for signal-based models. We introduce SignalGrad-CAM (SGrad-CAM), a versatile and efficient interpretability tool that extends the principles of Grad-CAM to both 1D- and 2D-convolutional neural networks for signal processing. SGrad-CAM is designed to interpret models for either image or signal elaboration, supporting both PyTorch and TensorFlow/Keras frameworks, and provides diagnostic and visualization tools to enhance model transparency. The package is also designed for batch processing, ensuring efficiency even for large-scale applications, while maintaining a simple and user-friendly structure.</p>

<p align="justify">**Keywords:** *eXplainable AI, explanations, local explanation, fidelity, interpretability, transparency, trustworthy AI, feature importance, saliency maps, CAM, Grad-CAM, black-box, deep learning, CNN, signals, time series*</p>

<p align="right"><a href="#top">Back To Top</a></p>

<!-- INSTALLATION -->
## Installation

1. Make sure you have the latest version of pip installed
   ```sh
   pip install --upgrade pip
    ```
2. Install SignalGrad-CAM through pip
    ```sh
     pip install signal-grad-cam
    ```

<p align="right"><a href="#top">Back To Top</a></p>

<!-- USAGE EXAMPLES -->
## Usage
<p align="justify">
Here's a basic example that illustrates SignalGrad-CAM common usage.

First, train a classifier on the data or select an already trained model, then instantiate `TorchCamBuilder` (if you are working with a PyTorch model) or `TfCamBuilder` (if the model is built in TensorFlow/Keras).

Besides the model, `TorchCamBuilder` requires additional information to function effectively. For example, you may provide a list of class labels, a preprocessing function, or an index indicating which dimension corresponds to time. These attributes allow SignalGrad-CAM to be applied to a wide range of models.

The constructor displays a list of available Grad-CAM algorithms for explanation, as well as a list of layers that can be used as target for the algorithm. It also identifies any Sigmoid/Softmax layer, since its presence or absence will slightly change the algorithm's workflow.
</p>

```python
import numpy as np
import torch
from signal_grad_cam import TorchCamBuilder

# Load model
model = YourTorchModelConstructor()
model.load_state_dict(torch.load("path_to_your_stored_model.pt")
model.eval()

# Introduce useful information
def preprocess_fn(signal):
   signal = torch.from_numpy(signal).float()
   # Extra preprocessing: data resizing, reshaping, normalization...
   return signal
class_labels = ["Class 1", "Class 2", "Class 3"]

# Define the CAM builder
cam_builder = TorchCamBuilder(model=model, transform_fc=preprocess_fc, class_names=class_labels, time_axs=1)
```

<p align="justify">Now, you can use the `cam_builder` object to generate class activation maps from a list of input data using the *`get_cams`* method. You can specify multiple algorithm names, target layers, or target classes as needed.

The function's attributes allow users to customize the visualization (e.g., setting axis ticks or labels). If a result directory path is provided, the output is stored as a '.png' file; otherwise, it is displayed. In all cases, the function returns a dictionary containing the requested CAMs, along with the model's predictions and importance score ranges.

Finally, several visualization tools are available to gain deeper insights into the model's behavior. The display can be customized by adjusting line width, point extension, aspect ratio, and more:
* *`single_channel_output_display`* plots the selected channels using a color scheme that reflects the importance of each input feature.
* *`overlapped_output_display`* superimposes CAMs onto the corresponding input in an image-like format, allowing users to capture the overall distribution of input importance.
</p>

```python
# Prepare data
data_list = [x for x in your_numpy_data_x[:2]]
data_labels_list = [1, 0]
item_names = ["Item 1", "Item 2"]
target_classes = [0, 1]

# Create CAMs
cam_dict, predicted_probs_dict, score_ranges_dict = cam_builder.get_cam(data_list=data_list, data_labels=data_labels_list, 
									target_classes=target_classes, explainer_types="Grad-CAM", 
									target_layer="conv1d_layer_1", softmax_final=True,
                                                            		data_sampling_freq=25, dt=1, axes_names=("Time (s)", "Channels"))

# Visualize single channel importance
selected_channels_indices = [0, 2, 10]
cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels_list, predicted_probs=predicted_probs_dict,
					  cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes, 
					  target_layers="target_layer_name", desired_channels=selected_channels_indices, 
					  grid_instructions=(1, len(selected_channels_indices), bar_ranges=score_ranges_dict, 
				          results_dir="path_to_your_result_directoory", data_sampling_freq=25, dt=1, line_width=0.5, 
					  axes_names=("Time (s)", "Amplitude (mV)"))

# Visualize overall importance
cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels_list, predicted_probs=predicted_probs_dict,
                                      cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes, 
				      target_layers="target_layer_name", fig_size=(20 * len(your_data_X), 20), 
				      grid_instructions=(len(your_data_X), 1), bar_ranges=score_ranges_dict, data_names=item_names 
				      results_dir="path_to_your_result_directoory", data_sampling_freq=25, dt=1)
```

You can also check the python scripts [here](https://github.com/bmi-labmedinfo/signal_grad_cam/examples).

See the [open issues](https://github.com/bmi-labmedinfo/signal_grad_cam/issues) for a full list of proposed features (and known issues).

<p align="right"><a href="#top">Back To Top</a></p>


If you use the SignalGrad-CAM software for your projects, please cite it as:

```
@software{Pe_SignalGrad_CAM_2025,
  author = {Pe, Samuele and Buonocore, Tommaso Mario and Giovanna, Nicora and Enea, Parimbelli},
  title = {{SignalGrad-CAM}},
  url = {https://github.com/bmi-labmedinfo/signal_grad_cam},
  version = {0.0.1},
  year = {2025}
}
```

<p align="right"><a href="#top">Back To Top</a></p>

<!-- CONTACTS AND USEFUL LINKS -->
## Contacts and Useful Links

*   **Repository maintainer**: Samuele Pe  [![Gmail][gmail-shield]][gmail-url] [![LinkedIn][linkedin-shield]][linkedin-url]  

*   **Project Link**: [https://github.com/bmi-labmedinfo/signal_grad_cam](https://github.com/bmi-labmedinfo/signal_grad_cam)

*   **Package Link**: [https://pypi.org/project/signal-grad-cam/](https://pypi.org/project/signal-grad-cam/)

<p align="right"><a href="#top">Back To Top</a></p>

<!-- LICENSE -->
## License

Distributed under MIT License. See `LICENSE` for more information.


<p align="right"><a href="#top">Back To Top</a></p>

<!-- MARKDOWN LINKS -->

[contributors-shield]: https://img.shields.io/github/contributors/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[contributors-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/graphs/contributors

[status-shield]: https://img.shields.io/badge/Status-pre--release-blue

[status-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/releases

[forks-shield]: https://img.shields.io/github/forks/samuelepe11/signal_grad_cam.svg?style=for-the-badge

[forks-url]: https://github.com/samuelepe11/signal_grad_cam/network/members

[stars-shield]: https://img.shields.io/github/stars/samuelepe11/signal_grad_cam.svg?style=for-the-badge

[stars-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/stargazers

[issues-shield]: https://img.shields.io/github/issues/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[issues-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/issues

[license-shield]: https://img.shields.io/github/license/samuelepe11/signal_grad_cam.svg?style=for-the-badge

[license-url]: https://github.com/samuelepe11/signal_grad_cam/LICENSE

[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white

[linkedin-url]: https://linkedin.com/in/samuele-pe-818bbb307

[gmail-shield]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white

[gmail-url]: mailto:samuele.pe01@universitadipavia.it
