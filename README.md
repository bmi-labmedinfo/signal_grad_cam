<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url]


<br />
<div align="center">
  <h1>
    SignalGrad-CAM
  </h1>

  <h3 align="center">SignalGrad-CAM aims at generalising Grad-CAM to time-based applications, while enhancing usability and efficiency.</h3>

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

<p align="justify">Deep learning models have achieved remarkable performance across many domains, yet their black-box nature often limits interpretability and trust. This has fueled the development of explanation algorithms within the field of eXplainable AI (XAI). Despite this progress, relatively few methods target time-based convolutional neural networks (CNNs), such as 1D-CNNs for signals and 3D-CNNs for videos. We present SignalGrad-CAM (SGrad-CAM), a versatile and efficient interpretability tool that extends the principles of Grad-CAM to 1D, 2D, and 3D CNNs. SGrad-CAM supports model interpretation for signals, images, and video/volume data in both PyTorch and TensorFlow/Keras frameworks. It includes diagnostic and visualization tools to enhance transparency, and its batch-processing design ensures scalability for large datasets while maintaining a simple, user-friendly structure.</p>

<p align="justify"><i><b>Keywords:</b> eXplainable AI, XAI, explanations, local explanation, contrastive explanations, cXAI, fidelity, interpretability, transparency, trustworthy AI, feature importance, saliency maps, CAM, Grad-CAM, HiResCAM, black-box, deep learning, CNN, 1D-CNN, 2D-CNN, 3D-CNN, signals, time series, images, videos, volumes.</i></p>

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
<p align="justify">Here's a basic example that illustrates SignalGrad-CAM common usage.</p>

<p align="justify">First, train a CNN on the data or load a pre-trained model, then instantiate `TorchCamBuilder` (if you are working with a PyTorch model) or `TfCamBuilder` (if the model is built in TensorFlow/Keras).</p>

<p align="justify">Besides the model, `TorchCamBuilder` requires additional information to function effectively. For example, you may provide a list of class labels, a pre-processing function, or an index indicating which dimension corresponds to time (for signal elaboration). These attributes allow SignalGrad-CAM to be applied to a wide range of models.</p>

<p align="justify">The constructor displays a list of available Grad-CAM algorithms for explanation (Grad-CAM and HiResCAM at the moment), as well as a list of layers that can be used as target for the algorithm. It also identifies any Sigmoid/Softmax layer, since its presence or absence will slightly change the algorithm's workflow.</p>

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
cam_builder = TorchCamBuilder(model=model, transform_fn=preprocess_fn, class_names=class_labels, time_axs=1)
```

<p align="justify">Now, you can use the `cam_builder` object to generate class activation maps from a list of input data using the <i>`get_cams`</i> method. You can specify multiple algorithm names, target layers, or target classes as needed. As described in each function's documentation, every input (such as data and labels) need to be rearranged into lists for versatility.</p>

<p align="justify">The function's attributes allow users to customize the visualization (e.g., setting axes ticks or labels). If a result directory path is provided, the output is stored as a '.png' file; otherwise, it is simply displayed. In all cases, the function returns a dictionary containing the requested CAMs, along with the model's predictions and importance score ranges.</p>

<p align="justify">Finally, several visualization tools are available to gain deeper insights into the model's behavior. Their display can be customized by adjusting features like line width and point extension (for the drawing of signals and their explanation) along with others (e.g., aspect ratio) for a more general task:</p>

* <p align="justify"><i>`single_channel_output_display`</i> plots the selected input channels using a color scheme that reflects the importance of each input feature.</p>
* <p align="justify"><i>`overlapped_output_display`</i> superimposes CAMs onto the corresponding input in an image-like format, allowing users to capture the overall distribution of input importance.</p>

```python
# Prepare data
data_list = [x for x in your_numpy_data_x[:2]]
data_labels_list = [1, 0]
item_names = ["Item 1", "Item 2"]
target_classes = [0, 1]

# Create CAMs
cam_dict, predicted_probs_dict, score_ranges_dict = cam_builder.get_cam(data_list=data_list, data_labels=data_labels_list, 
																		target_classes=target_classes, explainer_types="Grad-CAM", 
																		target_layers="conv1d_layer_1", softmax_final=True,
																		data_sampling_freq=25, dt=1, axes_names=("Time (s)", "Channels"))

# Visualize single channel importance
selected_channels_indices = [0, 2, 10]
cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels_list, predicted_probs_dict=predicted_probs_dict,
										  cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes, 
										  target_layers="target_layer_name", desired_channels=selected_channels_indices, 
										  grid_instructions=(1, len(selected_channels_indices), bar_ranges_dict=score_ranges_dict, 
										  results_dir="path_to_your_result_directoory", data_sampling_freq=25, dt=1, line_width=0.5, 
										  axes_names=("Time (s)", "Amplitude (mV)"))

# Visualize overall importance
cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels_list, predicted_probs_dict=predicted_probs_dict,
                                      cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes, 
									  target_layers="target_layer_name", fig_size=(20 * len(your_data_X), 20), 
									  grid_instructions=(len(your_data_X), 1), bar_ranges_dict=score_ranges_dict, data_names=item_names 
									  results_dir_path="path_to_your_result_directoory", data_sampling_freq=25, dt=1)
```

<p align="justify">You can also explore the Python scripts available in the examples directory of the repository [here](https://github.com/bmi-labmedinfo/signal_grad_cam/examples), which provide complete, ready-to-run demonstrations for both PyTorch and TensorFlow/Keras models. These examples include open-source models for signal, image and video/volume classification using 1D, 2D, and 3D CNN architectures. Moreover, these tutorials illustrate how to deploy the recently added feature contrastive explanations in each scenario.</p>

See the [open issues](https://github.com/bmi-labmedinfo/signal_grad_cam/issues) for a full list of proposed features (and known issues).

## <i>NEW!</i> Updates in SignalGrad-CAM
<p align="justify">Compared to previous versions, SignalGrad-CAM now offers the following enhancements:</p>

* <p align="justify"><i>Support for regression tasks:</i> SGrad-CAM can now handle regression-based models. Previously, substantial adjustments were required for these tasks, similar to those still needed for segmentation or generative models, now it is only required to set as True the parameter <i>`is_regression_network`</i> in the constructor function.</p>
* <p align="justify"><i>Contrastive explanations:</i> Users can generate and visualize contrastive explanations by specifying one or more foil classes (for classification tasks) or foil values (for regression tasks) via the parameter <i>`contrastive_foils`</i>.</p>
* <p align="justify"><i>3D-CNN support for videos and volumetric data:</i> After expliciting the time axis in the constructor with the parameter <i>`time_axs`</i>, the same functions used for 1D and 2D data now work seamlessly for 3D-CNNs. Outputs include GIF files for quick visualization of 3D activation maps. For a more detailed analysis, users can also request separate PNG images for each volume slice (across the indicated time axis) or video frame using the parameter <i>`show_single_video_frames`</i>.</p>

<p align="right"><a href="#top">Back To Top</a></p>


If you use the SignalGrad-CAM software for your projects, please cite it as:

```
@inproceedings{pe_sgradcam_2025_paper,
  author = {Pe, Samuele and Buonocore, Tommaso Mario and Giovanna, Nicora and Enea, Parimbelli},
  title = {SignalGrad-CAM: Beyond Image Explanation},
  booktitle = {Joint Proceedings of the xAI 2025 Late-breaking Work, Demos and Doctoral Consortium co-located with the 3rd World Conference on eXplainable Artificial Intelligence (xAI 2025), Istanbul, Turkey, July 9-11, 2025},
  series = {CEUR Workshop Proceedings},
  volume = {4017},
  pages = {209--216},
  url = {https://ceur-ws.org/Vol-4017/paper_27.pdf},
  publisher = {CEUR-WS.org},
  year = {2025}
}
```

```   
@software{pe_sgradcam_2025_repo,
  author = {Pe, Samuele},
  title = {SignalGrad-CAM},
  url = {https://github.com/bmi-labmedinfo/signal_grad_cam},
  version = {1.0.1},
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

[forks-shield]: https://img.shields.io/github/forks/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[forks-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/network/members

[stars-shield]: https://img.shields.io/github/stars/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[stars-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/stargazers

[issues-shield]: https://img.shields.io/github/issues/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[issues-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/issues

[license-shield]: https://img.shields.io/github/license/bmi-labmedinfo/signal_grad_cam.svg?style=for-the-badge

[license-url]: https://github.com/bmi-labmedinfo/signal_grad_cam/LICENSE

[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white

[linkedin-url]: https://linkedin.com/in/samuele-pe-818bbb307

[gmail-shield]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white

[gmail-url]: mailto:samuele.pe01@universitadipavia.it
