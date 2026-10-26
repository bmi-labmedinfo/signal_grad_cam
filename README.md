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

SignalGrad-CAM generates Grad-CAM or HiResCAM explanations for convolutional layers in PyTorch and TensorFlow/Keras models, including 1D signals and token sequences, 2D images, and 3D videos or volumes. The workflow is the same across modalities: **initialize a CAM builder**, **compute CAMs with `get_cam()`**, and, if needed, **choose a specialized visualization**.

### 1. Initialize a CAM builder

Train or load your model, then instantiate `TorchCamBuilder` for PyTorch or `TfCamBuilder` for TensorFlow/Keras. In PyTorch, place the model in evaluation mode before passing it to the builder. Both constructors inspect the available layers and list the supported explainers (`"Grad-CAM"` and `"HiResCAM"`). Use one of the displayed convolutional layer names as `target_layers` in `get_cam()`.

```python
import numpy as np
import torch
from signal_grad_cam import TorchCamBuilder

# Replace these placeholders with your trained model and its checkpoint.
model = YourTorchModelConstructor()
model.load_state_dict(torch.load("path_to_your_stored_model.pt", map_location="cpu"))
model.eval()

def preprocess_fn(signal):
    return torch.as_tensor(signal, dtype=torch.float32)

class_labels = ["Class 0", "Class 1", "Class 2"]
cam_builder = TorchCamBuilder(model=model, transform_fn=preprocess_fn, class_names=class_labels, time_axs=1)
```

The constructor parameters are organized below by purpose. Only `model` is mandatory; the remaining parameters have defaults.

#### General setup

| Parameter | Default | Description                                                                               |
| --- | --- |-------------------------------------------------------------------------------------------|
| `model` | *Required* | Trained PyTorch or TensorFlow/Keras model to explain.                                     |
| `transform_fn` | `None` | Function that converts and preprocesses each input into the format expected by the model. |
| `class_names` | `None` | Class labels used in explanation titles; provide a name for every output class.      |

#### Data layout

| Parameter | Default | Description                                                                                                                                                                                                                                                             |
| --- | --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `time_axs` | `1` | Index of the time or sequence axis in each input; relevant to 1D signals, token sequences, and videos/volumes. It refers to the time or sequence dimension within one input, not the batch dimension (for example, `1` for a signal shaped `(channels, samples)`). |
| `input_transposed` | `False` | Set to `True` when preprocessing or model inference transposes the original input layout.                                                                                                                                                                               |
| `ignore_channel_dim` | `False` | Ignore the singleton channel dimension when the model expects an input without it.                                                                                                                                                                                      |
| `padding_dim` | `None` | Pad inputs to a specified size along the configured time axis when batching variable-length data.                                                                                                                                                                       |

#### Model structure

| Parameter | Default | Description                                                                                        |
| --- | --- |----------------------------------------------------------------------------------------------------|
| `is_regression_network` | `False` | Indicate that the model performs regression rather than classification.                            |
| `is_transformer_network` | `False` | Enable the builder's handling of supported convolutional-based transformer networks (e.g., Hyena). |
| `model_output_index` | `None` | In case the model returns multiple outputs, selects the index of the prediction scores.            |

#### Extra utils

| Parameter | Default | Description                                                                                       |
| --- | --- |---------------------------------------------------------------------------------------------------|
| `extend_search` | `False` | Expand the search for candidate layers if the standard inspection does not find the target layer. |
| `seed` | `11` | Seed used for reproducibility during CAM generation.                                              |
| `use_gpu` | `False` | Request CUDA execution when available; for **PyTorch (`TorchCamBuilder`) only**.                  |


### 2. Generate CAMs with `get_cam()`

Call `get_cam()` with one input or a list of inputs. Multiple target classes, layers, and explanation algorithms can be requested in a single call. The method calculates the explanations, produces its standard CAM visualization, and returns three dictionaries: `cams_dict`, `predicted_probs_dict`, and `bar_ranges_dict` (the original minimum and maximum importance scores). Setting `results_dir_path` saves plots to disk; otherwise, they are displayed.

Each returned dictionary is indexed by the same key, assembled from the explainer, layer, target class, and optional contrastive foil. Use `get_item_key()` rather than assembling the key manually.

The following parameter groups mirror the `get_cam()` API. **Only the first six parameters are mandatory**; all others have defaults.

#### General setup

| Parameter | Description                                                                                                |
| --- |------------------------------------------------------------------------------------------------------------|
| `data_list` | One input array/tensor or a list of inputs to explain.                                                     |
| `data_labels` | Ground-truth label for each input, used in figure titles.                                                  |
| `target_classes` | Output class index or indices to explain.                                                                  |
| `explainer_types` | Currently accepting `"Grad-CAM"`, `"HiResCAM"`, or a list of both.                                         |
| `target_layers` | Name(s) of convolutional layers to explain, as reported by the builder.                                    |
| `softmax_final` | `True` if the model output is already passed through Sigmoid/Softmax; `False` if the model outputs logits. |

#### Explanation settings

| Parameter | Default | Description                                                                                                                                                                                           |
| --- | --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `contrastive_foils` | `None` | Needed to generate contrastive explanations as it indicates the class(es), `Q`,  to contrast against the target class (`P`), or a reference value for regression; `None` generates conventional CAMs. |
| `quadratic_contrastive_score` | `False` | Use the quadratic rather than linear contrastive score; relevant only when a foil is supplied.                                                                                                        |
| `avoid_relu` | `False` | Skip the final CAM ReLU to retain negative as well as positive contributions.                                                                                                                         |

#### General visualization

| Parameter | Default | Description                                                                                                   |
| --- | --- |---------------------------------------------------------------------------------------------------------------|
| `data_names` | `None` | Human-readable names for the input items; autogenerated names in the form `Input i` are used otherwise.       |
| `results_dir_path` | `None` | Directory for generated figures; `None` displays them without saving.                                         |
| `axes_names` | `None` | Optional `(x_label, y_label)` to override default axis labels.                                                |
| `data_shape_list` | `None` | Original dimensions used to resize CAMs back to each input. The shape is inferred automatically when omitted. |
| `channel_names` | `None` | Labels for multi-channel inputs. |

#### Visualization of time-varying data

| Parameter | Default | Description                                                                                  |
| --- | --- |----------------------------------------------------------------------------------------------|
| `data_sampling_freq` | `None` | Sampling frequency in samples per second; converts sample indices into time-axis labels.     |
| `dt` | `10` | Interval, in seconds, between labeled time-axis ticks when `data_sampling_freq` is provided. |
| `time_names` | `None` | Custom time-axis tick labels when a sampling frequency is not supplied.                      |
| `aspect_factor` | `100` | Controls the aspect ratio of the standard one-dimensional CAM heatmap.                       |

#### Specific features for videos and volumetric data

| Parameter | Default | Description |
| --- | --- | --- |
| `video_fps_list` | `None` | Frames per second for each video, used when producing GIFs; the implementation defaults to 10 FPS if omitted. |
| `show_single_video_frames` | `False` | Save individual frames/slices in addition to the rendered GIF. |

#### Advanced preprocessing and numerical settings

| Parameter | Default | Description |
| --- | --- | --- |
| `extra_preprocess_inputs_list` | `None` | Per-input argument lists passed to `transform_fn` after the data item. |
| `extra_inputs_dict` | `None` | Extra keyword arguments passed to the model's forward call. |
| `eps` | `1e-6` | Clamping constant used when converting Sigmoid/Softmax probabilities into scores for gradient computation. |


### 3. Select a visualization

`get_cam()` already generates a standard CAM display. These additional methods use its returned dictionaries to produce different views **without recomputing CAMs**:

- **`single_channel_output_display()`** plots individual signal or video/volume channels as curves with CAM-colored points, allowing local inspection of the importance assigned along each channel. Relevant options include `desired_channels`, `line_width`, `marker_width`, `fig_size`, and `grid_instructions`.
- **`overlapped_output_display()`** superimposes a CAM onto the original input. It is useful for images, two-dimensional representations of multichannel signals, and video/volume frames. Relevant options include `alpha`, `fig_size`, `grid_instructions`, and `target_item_ids`.
- **`text_output_display()`** draws a bar plot of CAM importance scores for text or other tokenized sequences, labeling each token on the x-axis and coloring its bar according to the importance score. It accepts `token_data_list` as a list of tokens for one input or a list of token lists for multiple inputs. Relevant options include `target_item_ids`, `data_names`, `grid_instructions`, `fig_size`, and `axes_names`. Token order and length must match the corresponding CAMs.

These methods require the original `data_labels` and the same `explainer_types`, `target_classes`, `target_layers`, and `contrastive_foils` used in `get_cam()`. Supply `bar_ranges_dict` to retain the original importance-score scale. `grid_instructions` is passed to Matplotlib's `subplot()` as `(rows, columns)` in the current implementation: make sure the grid contains enough cells for the items or channels you want to show.

#### Example: multichannel time-series explanations

The following example assumes `your_numpy_data_x` contains signals of shape `(channels, samples)` and the loaded model returns **logits**, not Softmax probabilities. Replace `conv1d_layer_1` with the exact name of a convolutional layer in your model.

```python
# Two signals (adapt labels and names to your data).
data_list = [np.asarray(x) for x in your_numpy_data_x[:2]]
data_labels_list = [1, 0]
item_names = ["Item 1", "Item 2"]
target_classes = [0, 1]
target_layer = "conv1d_layer_1"

# Compute explanations.
cam_dict, predicted_probs_dict, score_ranges_dict = cam_builder.get_cam(
    data_list=data_list, data_labels=data_labels_list, target_classes=target_classes,
    explainer_types="Grad-CAM", target_layers=target_layer, softmax_final=False,
    data_names=item_names, data_sampling_freq=25, dt=1, axes_names=("Time (s)", "Channels"))

# Plot individual channels (choose indices present in your inputs).
selected_channels_indices = [0, 1]
cam_builder.single_channel_output_display(
    data_list=data_list, data_labels=data_labels_list, predicted_probs_dict=predicted_probs_dict,
    cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes,
    target_layers=target_layer, desired_channels=selected_channels_indices,
    bar_ranges_dict=score_ranges_dict, data_names=item_names, grid_instructions=(2, 1),
    fig_size=(14, 8), data_sampling_freq=25, dt=1, line_width=0.5,
    axes_names=("Time (s)", "Amplitude"))

# Show the overall CAM as a heatmap over the input.
cam_builder.overlapped_output_display(
    data_list=data_list, data_labels=data_labels_list, predicted_probs_dict=predicted_probs_dict,
    cams_dict=cam_dict, explainer_types="Grad-CAM", target_classes=target_classes,
    target_layers=target_layer, bar_ranges_dict=score_ranges_dict, data_names=item_names,
    fig_size=(16, 10), grid_instructions=(2, 1), data_sampling_freq=25, dt=1)
```

<p align="justify">You can also explore the Python scripts available in the examples directory of the repository <a href="https://github.com/bmi-labmedinfo/signal_grad_cam/tree/main/examples">here</a>, which provide complete, ready-to-run demonstrations for both PyTorch and TensorFlow/Keras models. These examples include open-source models for signal, image and video/volume classification using 1D, 2D, and 3D CNN architectures. Moreover, these tutorials illustrate how to deploy the recently added feature contrastive explanations in each scenario.</p>
<p align="justify">A step-by-step demonstration of SignalGrad-CAM for generating classical and contrastive explanations on image, cardiological signal, and echocardiographic video data was presented at the Medical Informatics Europe (MIE) 2026 conference. Three live scripts are now available in <a href="https://drive.google.com/drive/folders/17VKblpb7QO5HT1sBLgKRCrvh5CWlId62?usp=drive_link">Google Colab</a>.</p>

See the [open issues](https://github.com/bmi-labmedinfo/signal_grad_cam/issues) for a full list of proposed features (and known issues).

## <i>NEW!</i> Updates in SignalGrad-CAM
<p align="justify">Compared to previous versions, SignalGrad-CAM now offers the following enhancements:</p>

* <p align="justify"><i>Support for regression tasks:</i> SGrad-CAM can now handle regression-based models. Previously, substantial adjustments were required for these tasks, similar to those still needed for segmentation or generative models, now it is only required to set as True the parameter <i>`is_regression_network`</i> in the constructor function.</p>
* <p align="justify"><i>Contrastive explanations:</i> Users can generate and visualize contrastive explanations by specifying one or more foil classes (for classification tasks) or foil values (for regression tasks) via the parameter <i>`contrastive_foils`</i>. Both linear and quadratic contrastive scoring functions are supported, with the latter enabled by setting quadratic_contrastive_score=True in get_cam().</p>
* <p align="justify"><i>3D-CNN support for videos and volumetric data:</i> After expliciting the time axis in the constructor with the parameter <i>`time_axs`</i>, the same functions used for 1D and 2D data now work seamlessly for 3D-CNNs. Outputs include GIF files for quick visualization of 3D activation maps. For a more detailed analysis, users can also request separate PNG images for each volume slice (across the indicated time axis) or video frame using the parameter <i>`show_single_video_frames`</i>.</p>
* <p align="justify"><i>Support for textual and sequential data:</i> SGrad-CAM has been extended to support text and tokenized sequences, including DNA sequences, through convolutional layers embedded in compatible transformer-based architectures. By setting is_transformer_network=True, users can generate CAMs for these models and visualize token-level importance scores using the new text_output_display() function, which associates each token with its corresponding contribution to the model prediction.</p>
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
