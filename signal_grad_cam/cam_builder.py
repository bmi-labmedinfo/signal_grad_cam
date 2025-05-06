# Import dependencies
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import re
import torch
import tensorflow as tf
from typing import Callable, List, Tuple, Dict, Any, Optional


# Class
class CamBuilder:
    """
    Represents a generic Class Activation Map (CAM) builder, supporting multiple methods such as Grad-CAM and HiResCAM.
    """

    explainer_types = {"Grad-CAM": "Gradient-weighted Class Activation Mapping",
                       "HiResCAM": "High-Resolution Class Activation Mapping"}

    def __init__(self, model: torch.nn.Module | tf.keras.Model | Any,
                 transform_fn: Callable[[np.ndarray, Optional[List[Any]]], torch.Tensor | tf.Tensor] = None,
                 model_apply_fn: Callable[[torch.Tensor | tf.Tensor, Optional[List[Any]]],
                 torch.Tensor | Tuple[tf.Tensor, tf.Tensor]] = None, class_names: List[str] = None, time_axs: int = 1,
                 input_transposed: bool = False, ignore_channel_dim: bool = False, model_output_index: int = None,
                 extend_search: bool = False, padding_dim: int = None, seed: int = 11):
        """
        Initializes the CamBuilder class. The constructor also displays, if present and retrievable, the 1D- and
        2D-convolutional layers in the network, as well as the final Sigmoid/Softmax activation. Additionally, the CAM
        algorithms available for generating the explanations are shown.

        :param model: (mandatory) A torch.nn.Module, tf.keras.Model, or any object (with PyTorch or TensorFlow/Keras
            layers among its attributes) representing a convolutional neural network model to be explained.
            Unconventional models should always be set to inference mode before being provided as inputs.
        :param transform_fn: (optional, default is None) A callable function to preprocess np.ndarray data before model
            evaluation. This function is also expected to convert data into either PyTorch or TensorFlow tensors. The
            function may optionally take as a second input a list of objects required by the preprocessing method.
        :param model_apply_fn: (optional, default is None) A callable function that applies the model to the PyTorch or
            TensorFlow tensor input and returns target activations (only for TfCamBuilder) and model outputs. The
            function may optionally take as a second input a list of objects required by the model's forward method.
        :param class_names: (optional, default is None) A list of strings where each string represents the name of an
            output class.
        :param time_axs: (optional, default is 1) An integer index indicating whether the input signal's time axis is
            represented as the first or second dimension of the input array.
        :param input_transposed: (optional, default is False) A boolean indicating whether the input array is transposed
            during model inference, either by the model itself or by the preprocessing function.
        :param ignore_channel_dim: (optional, default is False) A boolean indicating whether to ignore the channel
            dimension. This is useful when the model expects inputs without a singleton channel dimension.
        :param model_output_index: (optional, default is None) An integer index specifying which of the model's outputs
            represents output scores (or probabilities). If there is only one output, this argument can be ignored.
        :param extend_search: (optional, default is False) A boolean flag indicating whether to deepend the search for
            candidate layers. It should be set true if no convolutional layer was found.
        :param padding_dim: (optional, default is None) An integer specifying the maximum length along the time axis to
            which each item will be padded for batching.
        :param seed: (optional, default is 11) An integer seed for random number generators, used to ensure
            reproducibility during model evaluation.
        """

        # Set seeds
        np.random.seed(seed)
        random.seed(seed)

        # Initialize attributes
        self.model = model
        self.transform_fn = transform_fn
        self.model_apply_fn = model_apply_fn
        self.class_names = class_names
        self.extend_search = extend_search

        self.time_axs = time_axs
        self.input_transposed = input_transposed
        self.ignore_channel_dim = ignore_channel_dim
        self.model_output_index = model_output_index
        self.padding_dim = padding_dim
        self.original_dims = []

        self.gradients = None
        self.activations = None

        # Show available explainers
        print()
        print("====================================================================================================")
        print("                                      Executing SignalGrad-CAM                                      ")
        print("====================================================================================================")
        print()
        print("AVAILABLE EXPLAINERS:")
        for k, v in self.explainer_types.items():
            print(f" - Explainer identifier '{k}': {v}")

        # Show available 1D or 2D convolutional layers
        print()
        print("SEARCHING FOR NETWORK LAYERS:")
        self.__print_justify("Please, verify that your network contains at least one 1D or 2D convolutional layer "
                             "and note the names of the layers that are of interest to you. If the desired layer is not"
                             " present in the following list, it can still be accessed by the name used in the network "
                             "to identify it.\nVerify whether the model ends with an activation function from the "
                             "Softmax family (such as Sigmoid o Softmax). Even if this activation function is not "
                             "present in the following list, ensure to check if it is applied at the end of the "
                             "network. Please make sure that the provided model is set in inference ('eval') mode for "
                             "PyTorch models and that TensorFlow/Keras models have been built (they must have the "
                             "specific 'inputs' and 'output' attributes)\nNetwork layers found (name: type)")
        self._get_layers_pool(show=True, extend_search=extend_search)
        print()

    def get_cam(self, data_list: List[np.ndarray], data_labels: List[int], target_classes: int | List[int],
                explainer_types: str | List[str], target_layers: str | List[str], softmax_final: bool,
                data_names: List[str] = None, data_sampling_freq: float = None, dt: float = 10,
                channel_names: List[str | float] = None, results_dir_path: str = None, aspect_factor: float = 100,
                data_shape_list: List[Tuple[int, int]] = None, time_names: List[str | float] = None,
                axes_names: Tuple[str | None, str | None] | List[str | None] = None) \
            -> Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray],  Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Allows the user to request Class Activation Maps (CAMs) for a given list of inputs, a set of algorithms,
        target classes, and target layers. Returns a standard visualization of each CAM along with representative
        outputs, enabling a customized display of CAMs. Optional inputs are employed for a more detailed
        visualization of the results.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param data_labels: (mandatory) A list of integers representing the true labels of the data to be explained.
        :param target_classes: (mandatory) An integer or a list of integers representing the target classes for the
            explanation.
        :param explainer_types: (mandatory) A string or a list of strings representing the desired algorithms for the
            explanation. These strings should identify one of the CAM algorithms allowed, as listed by the class
            constructor.
        :param target_layers: (mandatory) A string or a list of strings representing the target layers for the
            explanations. These strings should identify either PyTorch named modules, TensorFlow/Keras layers, or they
            should be class dictionary keys, used to retrieve each layer from the class attributes.
        :param softmax_final: (mandatory) A boolean indicating whether the network terminates with a Sigmoid/Softmax
            activation function.
        :param data_names: (optional, default is None) A list of strings where each string represents the name of an
            input item.
        :param data_sampling_freq: (optional, default is None) A numerical value representing the sampling frequency of
            signal inputs in samples per second.
        :param dt: (optional, default is 10) A numerical value representing the granularity of the time axis in seconds
            in the output display.
        :param channel_names: (optional, default is None) A list of strings where each string represents the name of a
            signal channel for tick settings.
        :param results_dir_path: (optional, default is None) A string representing the relative path to the directory
            for storing results. If None, the output will be displayed in a figure.
        :param aspect_factor: (optional, default is 100) A numerical value to set the aspect ratio of the output signal
            one-dimensional CAM.
        :param data_shape_list: (optional, default is None) A list of integer tuples storing the original input sizes,
            used to set the CAM shape after resizing during preprocessing. The expected format is number of rows x
             number of columns.
        :param time_names: (optional, default is None) A list of strings representing tick names for the time axis.
        :param axes_names: (optional, default is None) A tuple of strings representing names for X and Y axes,
            respectively.

        :return:
            - cams_dict: A dictionary storing a list of CAMs. Each list contains CAMs for each item in the input data
                list, corresponding to a given setting (defined by algorithm, target layer, and target class).
            - predicted_probs_dict: A dictionary storing a np.ndarray. Each array represents the inferred class
                probabilities for each item in the input list.
            - bar_ranges_dict: A dictionary storing a tuple of np.ndarrays. Each tuple contains two np.ndarrays
                corresponding to the minimum and maximum importance scores per CAM for each item in the input data list,
                based on a given setting (defined by algorithm, target layer, and target class).
        """

        # Check data names
        if data_names is None:
            data_names = ["item" + str(i) for i in range(len(data_list))]

        # Check input types
        target_classes, explainer_types, target_layers = self.__check_input_types(target_classes, explainer_types,
                                                                                  target_layers)
        for explainer_type in explainer_types:
            if explainer_type not in self.explainer_types:
                raise ValueError("'explainer_types' should be an explainer identifier or a list of explainer "
                                 "identifiers.")

        # Draw CAMs
        cams_dict = {}
        predicted_probs_dict = {}
        bar_ranges_dict = {}
        for explainer_type in explainer_types:
            for target_class in target_classes:
                for target_layer in target_layers:
                    cam_list, output_probs, bar_ranges = self.__create_batched_cams(data_list, target_class,
                                                                                    target_layer, explainer_type,
                                                                                    softmax_final, data_shape_list)
                    item_key = explainer_type + "_" + target_layer + "_class" + str(target_class)
                    cams_dict.update({item_key: cam_list})
                    predicted_probs_dict.update({item_key: output_probs})
                    bar_ranges_dict.update({item_key: bar_ranges})
                    self.__display_output(data_labels, target_class, explainer_type, target_layer, cam_list, output_probs,
                                          results_dir_path, data_names, data_sampling_freq, dt, aspect_factor,
                                          bar_ranges, channel_names, time_names=time_names, axes_names=axes_names)

        return cams_dict, predicted_probs_dict, bar_ranges_dict

    # Check data names
    def overlapped_output_display(self, data_list: List[np.ndarray], data_labels: List[int],
                                  predicted_probs_dict: Dict[str, np.ndarray], cams_dict: Dict[str, List[np.ndarray]],
                                  explainer_types: str | List[str], target_classes: int | List[int],
                                  target_layers: str | List[str], target_item_ids: List[int] = None,
                                  data_names: List[str] = None, grid_instructions: Tuple[int, int] = None,
                                  bar_ranges_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
                                  results_dir_path: str = None, data_sampling_freq: float = None, dt: float = 10,
                                  channel_names: List[str | float] = None, time_names: List[str | float] = None,
                                  axes_names: Tuple[str | None, str | None] | List[str | None] = None,
                                  fig_size: Tuple[int, int] = None) -> None:
        """
        Generates a superimposition of the input data and the selected CAMs, useful for visualizing image explanations
        and multichannel signals with numerous channels, such as frequency spectra.


        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param data_labels: (mandatory) A list of integers representing the true labels of the data to be explained.
        :param predicted_probs_dict: (mandatory) A dictionary storing a np.ndarray. Each array represents the inferred
            class probabilities for each item in the input list.
        :param cams_dict: (mandatory) A dictionary storing a list of CAMs. Each list contains CAMs for each item in the
            input data list, corresponding to a given setting (defined by algorithm, target layer, and target class).
        :param explainer_types: (mandatory) A string or a list of strings representing the desired algorithms for the
            explanation. These strings should identify one of the CAM algorithms allowed, as listed by the class
            constructor.
        :param target_classes: (mandatory) An integer or a list of integers representing the target classes for the
            explanation.
        :param target_layers: (mandatory) A string or a list of strings representing the target layers for the
            explanations. These strings should identify either PyTorch named modules, TensorFlow/Keras layers, or they
            should be class dictionary keys, used to retrieve each layer from the class attributes.
        :param target_item_ids: (optional, default is None) A list of integers representing the target item indices
            among the items in the input data list.
        :param data_names: (optional, default is None) A list of strings where each string represents the name of an
            input item.
        :param grid_instructions: (optional, default is None) A tuple of integers defining the desired tabular layout
            for figure subplots. The expected format is number of columns (width) x number of rows (height).
        :param bar_ranges_dict: A dictionary storing a tuple of np.ndarrays. Each tuple contains two np.ndarrays
                corresponding to the minimum and maximum importance scores per CAM for each item in the input data list,
                based on a given setting (defined by algorithm, target layer, and target class).
        :param results_dir_path: (optional, default is None) A string representing the relative path to the directory
            for storing results. If None, the output will be displayed in a figure.
        :param data_sampling_freq: (optional, default is None) A numerical value representing the sampling frequency of
            signal inputs in samples per second.
        :param dt: (optional, default is 10) A numerical value representing the granularity of the time axis in seconds
            in the output display.
        :param channel_names: (optional, default is None) A list of strings where each string represents the name of a
            signal channel for tick settings.
        :param time_names: (optional, default is None) A list of strings representing tick names for the time axis.
        :param axes_names: (optional, default is None) A tuple of strings representing names for X and Y axes,
            respectively.
        :param fig_size: (optional, default is None) A tuple of integers defining the dimensions of the output figure.
            The expected format is width x height in inches.
        """

        # Check input types
        target_classes, explainer_types, target_layers = self.__check_input_types(target_classes, explainer_types,
                                                                                  target_layers)
        if target_item_ids is None:
            target_item_ids = list(range(len(data_list)))

        # Define window size
        n_items = len(target_item_ids)
        w, h = self.__set_grid(n_items, grid_instructions)
        if w * h < n_items:
            n_items = w * h
            target_item_ids = target_item_ids[:n_items]
        fig_size = fig_size if fig_size is not None else (8 * w, 16 * h)

        for explainer_type in explainer_types:
            for target_layer in target_layers:
                for target_class in target_classes:
                    plt.figure(figsize=fig_size)
                    for i in range(n_items):
                        cam, item, batch_idx, item_key = self.__get_data_for_plots(data_list, i, target_item_ids,
                                                                                   cams_dict, explainer_type,
                                                                                   target_layer, target_class)

                        plt.subplot(w, h, i + 1)
                        plt.imshow(item)
                        aspect = "auto" if cam.shape[0] / cam.shape[1] < 0.1 else None

                        norm = self.__get_norm(cam)
                        map = plt.imshow(cam, cmap="jet", aspect=aspect, norm=norm)
                        self.__set_colorbar(bar_ranges_dict[item_key], i)
                        map.set_alpha(0.3)

                        self.__set_axes(cam, data_sampling_freq, dt, channel_names, time_names=time_names,
                                        axes_names=axes_names)
                        data_name = data_names[batch_idx] if data_names is not None else "item" + str(batch_idx)
                        plt.title(self.__get_cam_title(data_name, target_class, data_labels, batch_idx, item_key,
                                                       predicted_probs_dict))

                    # Store or show CAM
                    self.__display_plot(results_dir_path, explainer_type, target_layer, target_class)

    def single_channel_output_display(self, data_list: List[np.ndarray], data_labels: List[int],
                                      predicted_probs_dict: Dict[str, np.ndarray],
                                      cams_dict: Dict[str, List[np.ndarray]], explainer_types: str | List[str],
                                      target_classes: int | List[int], target_layers: str | List[str],
                                      target_item_ids: List[int] = None, desired_channels: List[int] = None,
                                      data_names: List[str] = None, grid_instructions: Tuple[int, int] = None,
                                      bar_ranges_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
                                      results_dir_path: str = None, data_sampling_freq: float = None, dt: float = 10,
                                      channel_names: List[str | float] = None, time_names: List[str | float] = None,
                                      axes_names: Tuple[str | None, str | None] | List[str | None] = None,
                                      fig_size: Tuple[int, int] = None, line_width: float = 0.1,
                                      marker_width: float = 30) -> None:
        """
        Displays input signal channels, coloring each with "jet" colormat according to the corresponding CAMs. This
        visualization is useful for interpreting signal explanations with a limited number of channels. If many channels
        are present, it is recommended to select only a subset.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param data_labels: (mandatory) A list of integers representing the true labels of the data to be explained.
        :param predicted_probs_dict: (mandatory) A dictionary storing a np.ndarray. Each array represents the inferred
            class probabilities for each item in the input list.
        :param cams_dict: (mandatory) A dictionary storing a list of CAMs. Each list contains CAMs for each item in the
            input data list, corresponding to a given setting (defined by algorithm, target layer, and target class).
        :param explainer_types: (mandatory) A string or a list of strings representing the desired algorithms for the
            explanation. These strings should identify one of the CAM algorithms allowed, as listed by the class
            constructor.
        :param target_classes: (mandatory) An integer or a list of integers representing the target classes for the
            explanation.
        :param target_layers: (mandatory) A string or a list of strings representing the target layers for the
            explanations. These strings should identify either PyTorch named modules, TensorFlow/Keras layers, or they
            should be class dictionary keys, used to retrieve each layer from the class attributes.
        :param target_item_ids: (optional, default is None) A list of integers representing the target item indices
            among the items in the input data list.
        :param desired_channels: (optional, default is None) A list of integers representing the selected channels
            to be displayed.
        :param data_names: (optional, default is None) A list of strings where each string represents the name of an
            input item.
        :param grid_instructions: (optional, default is None) A tuple of integers defining the desired tabular layout
            for figure subplots. The expected format is number of columns (width) x number of rows (height).
        :param bar_ranges_dict: A dictionary storing a tuple of np.ndarrays. Each tuple contains two np.ndarrays
                corresponding to the minimum and maximum importance scores per CAM for each item in the input data list,
                based on a given setting (defined by algorithm, target layer, and target class).
        :param results_dir_path: (optional, default is None) A string representing the relative path to the directory
            for storing results. If None, the output will be displayed in a figure.
        :param data_sampling_freq: (optional, default is None) A numerical value representing the sampling frequency of
            signal inputs in samples per second.
        :param dt: (optional, default is 10) A numerical value representing the granularity of the time axis in seconds
            in the output display.
        :param channel_names: (optional, default is None) A list of strings where each string represents the name of a
            signal channel for tick settings.
        :param time_names: (optional, default is None) A list of strings representing tick names for the time axis.
        :param axes_names: (optional, default is None) A tuple of strings representing names for X and Y axes,
            respectively.
        :param fig_size: (optional, default is None) A tuple of integers defining the dimensions of the output figure.
            The expected format is width x height in inches.
        :param line_width: (optional, default is 0.1) A numerical value representing the width in typographic points of
            the black interpolation lines in the plots.
        :param marker_width: (optional, default is 30) A numerical value representing the size in typographic points**2
            of the jet-colored markers in the plots.
        """

        # Check input types
        target_classes, explainer_types, target_layers = self.__check_input_types(target_classes, explainer_types,
                                                                                  target_layers)
        if desired_channels is None:
            try:
                desired_channels = list(range(data_list[0].shape[1]))
            except IndexError:
                desired_channels = [0]

        if target_item_ids is None:
            target_item_ids = list(range(len(data_list)))

        # Define window size
        n_items = len(target_item_ids)
        w, h = self.__set_grid(n_items, grid_instructions)
        fig_size = fig_size if fig_size is not None else (6 * w, 6 * h)
        if w * h < len(list(desired_channels)):
            n_channels = w * h
            desired_channels = list(desired_channels)[:n_channels]

        for explainer_type in explainer_types:
            for target_layer in target_layers:
                for target_class in target_classes:
                    for i in range(n_items):
                        plt.figure(figsize=fig_size)
                        cam, item, batch_idx, item_key = self.__get_data_for_plots(data_list, i, target_item_ids,
                                                                                   cams_dict, explainer_type,
                                                                                   target_layer, target_class)

                        # Cross-CAM normalization
                        minimum = np.min(cam)
                        maximum = np.max(cam)

                        data_name = data_names[batch_idx] if data_names is not None else "item" + str(batch_idx)
                        desired_channels = desired_channels if desired_channels is not None else range(cam.shape[1])
                        for j in range(len(desired_channels)):
                            channel = desired_channels[j]
                            plt.subplot(w, h, j + 1)
                            try:
                                cam_j = cam[channel, :]
                            except IndexError:
                                cam_j = cam[0, :]
                            item_j = item[:, channel] if item.shape[0] == len(cam_j) else item[channel, :]
                            plt.plot(item_j, color="black", linewidth=line_width)
                            plt.scatter(np.arange(len(item_j)), item_j, c=cam_j, cmap="jet", marker=".",
                                        s=marker_width, norm=None, vmin=minimum, vmax=maximum)
                            self.__set_colorbar(bar_ranges_dict[item_key], i)

                            if channel_names is None:
                                channel_names = ["Channel " + str(c) for c in desired_channels]
                            self.__set_axes(cam, data_sampling_freq, dt, channel_names, time_names,
                                            axes_names=axes_names, only_x=True)
                            plt.title(channel_names[j])
                        plt.suptitle(self.__get_cam_title(data_name, target_class, data_labels, batch_idx, item_key,
                                                          predicted_probs_dict))

                        # Store or show CAM
                        self.__display_plot(results_dir_path, explainer_type, target_layer, target_class, data_name,
                                            is_channel=True)

    def _get_layers_pool(self, show: bool = False, extend_search: bool = False) \
            -> Dict[str, torch.nn.Module | tf.keras.layers.Layer | Any]:
        """
        Retrieves a dictionary containing all the available PyTorch or TensorFlow/Keras layers (or instance attributes),
        with the layer (or attribute) names used as keys.

        :param show: (optional, default is False) A boolean flag indicating whether to display the retrieved layers
            along with their names.
        :param extend_search: (optional, default is False) A boolean flag indicating whether to deepend the search for
            candidate layers. It should be set true if no convolutional layer was found.

        :return:
            - layers_pool: A dictionary storing the model's PyTorch or TensorFlow/Keras layers (or instance attributes),
            with layer (or attribute) names as keys.
        """

        layers_pool = self.model.__dict__
        if show:
            for name, layer in layers_pool.items():
                self._show_layer(name, layer, potential=True)

        return layers_pool

    def _show_layer(self, name: str, layer: torch.nn.Module | tf.keras.layers.Layer | Any, potential: bool = False) \
            -> None:
        """
        Displays a single available layer (or instance attribute) in the model, along with its corresponding name.

        :param name: (mandatory) A string representing the name of the layer or attribute.
        :param layer: (mandatory) A PyTorch or TensorFlow/Keras layer, or an instance attribute in the model.
        :param potential: (optional, default is False) A flag indicating whether the object displayed is potentially
            a layer (i.e., a generic instance attribute, not guaranteed to be a layer).
        """

        addon = "(potential layer) " if potential else ""
        txt = " - " + addon + f"{name}:\t{type(layer).__name__}"
        print(txt)

    def _create_raw_batched_cams(self, data_list: List[np.ndarray], target_class: int, target_layer: str,
                                 explainer_type: str, softmax_final: bool, extra_inputs_list: List[Any]) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Retrieves raw CAMs from an input data list based on the specified settings (defined by algorithm, target layer,
        and target class). Additionally, it returns the class probabilities predicted by the model.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules, TensorFlow/Keras layers, or it should be a class dictionary key,
            used to retrieve the layer from the class attributes.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param softmax_final: (mandatory) A boolean indicating whether the network terminates with a Sigmoid/Softmax
            activation function.
        :param extra_inputs_list: (mandatory) A list of additional input objects required by the model's forward method.

        :return:
            - cam_list: A list of np.ndarray containing CAMs for each item in the input data list, corresponding to the
                given setting (defined by algorithm, target layer, and target class).
            - target_probs: A np.ndarray, representing the inferred class probabilities for each item in the input list.
        """

        raise AttributeError("The method '_create_raw_batched_cams' is not available for the parent class "
                             "'CamBuilder': you will need to instantiate either a 'TorchCamBuilder' or a 'TfCamBuilder'"
                             " instance to use it.")

    def _get_gradcam_map(self, is_2d_layer: bool, batch_idx: int) -> torch.Tensor | tf.Tensor:
        """
        Compute the CAM using the vanilla Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A PyTorch or TensorFlow/Keras tensor representing the Class Activation Map (CAM) for the
            batch_idx-th input, built with the Grad-CAM algorithm.
        """

        raise AttributeError("The method '__get_gradcam_map' is not available for the parent class 'CamBuilder': you "
                             "will need to instantiate either a 'TorchCamBuilder' or a 'TfCamBuilder' instance to use "
                             "it.")

    def _get_hirescam_map(self, is_2d_layer: bool, batch_idx: int) -> np.ndarray:
        """
        Compute the CAM using the High-Resolution Class Activation Mapping (HiResCAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A PyTorch or TensorFlow/Keras tensor representing the Class Activation Map (CAM) for the
            batch_idx-th input, built with the HiResCAM algorithm.
        """

        raise AttributeError("The method '__get_hirecam_map' is not available for the parent class 'CamBuilder': you "
                             "will need to instantiate either a 'TorchCamBuilder' or a 'TfCamBuilder' instance to use "
                             "it.")

    def __create_batched_cams(self, data_list: List[np.ndarray], target_class: int, target_layer: str,
                              explainer_type: str, softmax_final: bool, extra_preprocess_inputs_list: List[Any],
                              extra_inputs_list: List[Any], data_shape_list: List[Tuple[int, int]] = None) \
            -> Tuple[List[np.ndarray], np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepares the input data list and retrieves CAMs based on the specified settings (defined by algorithm, target
        layer, and target class), along with class probabilities predicted by the model. Additionally, it adjusts the
        output CAMs in both shape and value range (0-255), and returns the original importance score range.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param target_class: (mandatory) An integer representing the target classe for the explanation.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules, TensorFlow/Keras layers, or it should be a class dictionary key,
            used to retrieve the layer from the class attributes.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param softmax_final: (mandatory) A boolean indicating whether the network terminates with a Sigmoid/Softmax
            activation function.
        :param extra_preprocess_inputs_list: (mandatory) A list of additional input objects required by the
            preprocessing method.
        :param extra_inputs_list: (mandatory) A list of additional input objects required by the model's forward method.
        :param data_shape_list: (optional, default is None) A list of integer tuples storing the original input sizes,
            used to set the CAM shape after resizing during preprocessing. The expected format is number of rows x
            number of columns.

        :return:
            - cam_list: A list of np.ndarray containing CAMs for each item in the input data list, corresponding to the
                given setting (defined by algorithm, target layer, and target class).
            - target_probs: A np.ndarray, representing the inferred class probabilities for each item in the input list.
            - bar_ranges: A tuple containing two np.ndarrays, corresponding to the minimum and maximum importance scores
                per CAM for each item in the input data list, based on a given setting (defined by algorithm, target
                layer, and target class).
        """

        # Select target layer
        layers_pool = self._get_layers_pool(extend_search=self.extend_search)
        target_layer = layers_pool[target_layer]

        # Preprocess data
        if data_shape_list is None:
            data_shape_list = [data_element.shape for data_element in data_list]
        if self.transform_fn is not None:
            extra_preprocess_inputs_list = extra_preprocess_inputs_list or []
            data_list = [self.transform_fn(data_element, *extra_preprocess_inputs_list) for data_element in data_list]

        # Ensure data have consistent size for batching
        if len(data_list) > 1 and self.padding_dim is None:
            data_shape_list_processed = [data_element.shape for data_element in data_list]
            if len(np.unique(np.array(data_shape_list_processed, dtype=object))) != 1:
                data_list = [np.resize(x, data_shape_list_processed[0]) for x in data_list]
                self.__print_justify("Input data items have different shapes. Each item has been reshaped to match the "
                                     "first item's dimensions for batching. To prevent this, provide one item at a "
                                     "time.")

        cam_list, target_probs = self._create_raw_batched_cams(data_list, target_class, target_layer, explainer_type,
                                                               softmax_final, extra_inputs_list)
        self.activations = None
        self.gradients = None
        cams = np.stack(cam_list)
        cam_list, bar_ranges = self.__adjust_maps(cams, data_shape_list, self._is_2d_layer(target_layer))
        return cam_list, target_probs, bar_ranges

    def __adjust_maps(self, cams: np.ndarray, data_shape_list: List[Tuple[int, int]], is_2d_layer: bool) \
            -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Adjusts the output CAMs in both shape and value range (0-255), and returns the original importance score range.

        :param cams: (mandatory) A np.ndarray representing a batch of CAMs, one per item in the input data batch.
        :param data_shape_list: (mandatory) A list of integer tuples storing the original input sizes, used to set the
            CAM shape after resizing during preprocessing. The expected format is number of rows x number of columns.
        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.

        :return:
            - cam_list: A list of np.ndarray containing CAMs for each item in the input data list, corresponding to the
                given setting (defined by algorithm, target layer, and target class).
            - bar_ranges: A tuple containing two np.ndarrays, corresponding to the minimum and maximum importance scores
                per CAM for each item in the input data list, based on a given setting (defined by algorithm, target
                layer, and target class).
        """

        cams, bar_ranges = self.__normalize_cams(cams, is_2d_layer)

        cam_list = []
        for i in range(len(data_shape_list)):
            cam = cams[i]
            if is_2d_layer:
                dim_reshape = (data_shape_list[i][1], data_shape_list[i][0])
                if self.input_transposed:
                    dim_reshape = dim_reshape[::-1]
            else:
                dim_reshape = (1, data_shape_list[i][self.time_axs])
                if self.time_axs:
                    cam = np.transpose(cam)
            if self.padding_dim is not None:
                original_dim = dim_reshape[1]
                dim_reshape = (dim_reshape[0], self.padding_dim)
            cam = cv2.resize(cam, dim_reshape)

            if is_2d_layer and self.input_transposed:
                cam = np.transpose(cam)
            if self.padding_dim is not None:
                cam = cam[:original_dim, :]
            cam_list.append(cam)

        return cam_list, bar_ranges

    def __display_output(self, data_labels: List[int], target_class: int, explainer_type: str, target_layer: str,
                         cam_list: List[np.ndarray], predicted_probs: np.ndarray, results_dir_path: str,
                         data_names: List[str], data_sampling_freq: float = None, dt: float = 10,
                         aspect_factor: float = 100, bar_ranges: Tuple[np.ndarray, np.ndarray] = None,
                         channel_names: List[str | float] = None, time_names: List[str | float] = None,
                         axes_names: Tuple[str | None, str | None] = None) -> None:
        """
        Create plots displaying the obtained CAMs, set their axes, and show them as multiple figures or as ".png" files.

        :param data_labels: (mandatory) A list of integers representing the true labels of the data to be explained.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules, TensorFlow/Keras layers, or it should be a class dictionary key,
            used to retrieve the layer from the class attributes.
        :param cam_list: (mandatory) A list of np.ndarray containing CAMs for each item in the input data list,
            corresponding to the given setting (defined by algorithm, target layer, and target class).
        :param predicted_probs: (mandatory) A np.ndarray, representing the inferred class probabilities for each item in
            the input list.
        :param results_dir_path: (mandatory) A string representing the relative path to the directory
            for storing results. If None, the output will be displayed in a figure.
        :param data_names: (optional, default is None) A list of strings where each string represents the name of an
            input item.
        :param data_sampling_freq: (optional, default is None) A numerical value representing the sampling frequency of
            signal inputs in samples per second.
        :param dt: (optional, default is 10) A numerical value representing the granularity of the time axis in seconds
            in the output display.
        :param aspect_factor: (optional, default is 100) A numerical value to set the aspect ratio of the output signal
            one-dimensional CAM.
        :param bar_ranges: A tuple containing two np.ndarrays, corresponding to the minimum and maximum importance scores
            per CAM for each item in the input data list, based on a given setting (defined by algorithm, target
            layer, and target class).
        :param channel_names: (optional, default is None) A list of strings where each string represents the name of a
            signal channel for tick settings.
        :param time_names: (optional, default is None) A list of strings representing tick names for the time axis.
        :param axes_names: (optional, default is None) A tuple of strings representing names for X and Y axes,
            respectively.
        """

        if not os.path.exists(results_dir_path):
            os.makedirs(results_dir_path)

        is_2d_layer = self._is_2d_layer(self._get_layers_pool(extend_search=self.extend_search)[target_layer])

        n_cams = len(cam_list)
        for i in range(n_cams):
            map = cam_list[i]
            data_name = data_names[i]

            # Display CAM
            plt.figure()
            norm = self.__get_norm(map)

            if map.shape[1] == 1:
                aspect = int(map.shape[0] / aspect_factor)
                map = np.transpose(map)
            else:
                if is_2d_layer:
                    aspect = "auto"
                else:
                    aspect = 1
                if not self.time_axs:
                    map = np.transpose(map)
            plt.matshow(map, cmap=plt.get_cmap("jet"), norm=norm, aspect=aspect)

            # Add color bar
            self.__set_colorbar(bar_ranges, i)

            # Set title
            '''if map.shape[0] > 1 and is_2d_layer:
                title_h = 1
                plt.subplots_adjust(top=0.85, bottom=0.2)
            else:
                title_h = 0.98'''
            plt.title("CAM for class '" + str(self.class_names[target_class]) + "' (confidence = " +
                      str(np.round(predicted_probs[i] * 100, 2)) + "%) - true label " +
                      str(self.class_names[data_labels[i]]))#, y=title_h)

            # Set axis
            self.__set_axes(map, data_sampling_freq, dt, channel_names, time_names=time_names, axes_names=axes_names)

            # Store or show CAM
            self.__display_plot(results_dir_path, explainer_type, target_layer, target_class, data_name)

    def __get_data_for_plots(self, data_list: List[np.ndarray], i: int, target_item_ids: List[int],
                             cams_dict: Dict[str, List[np.ndarray]], explainer_type: str, target_layer: str,
                             target_class: int) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """
        Prepares input data and CAMs to be plotted, identifying the string key to retrieve CAMs, probabilities and
        ranges from the corresponding dictionaries.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param i: (mandatory) An integer representing the index of an item among the selected ones.
        :param target_item_ids: (optional, default is None) A list of integers representing the target item indices
            among the items in the input data list.
        :param cams_dict: (mandatory) A dictionary storing a list of CAMs. Each list contains CAMs for each item in the
            input data list, corresponding to a given setting (defined by algorithm, target layer, and target class).
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules, TensorFlow/Keras layers, or it should be a class dictionary key,
            used to retrieve the layer from the class attributes.
        :param target_class: (mandatory) An integer representing the target class for the explanation.

        :return:
            - cam: The CAM for the given setting (defined by algorithm, target layer, and target class), corresponding
                to the i-th item in the selected ones.
            - item: The i-th item in the selected ones.
            - batch_idx: The index corresponding to the i-th selected item within the original input data list.
            - item_key: A string representing the considered setting (defined by algorithm, target layer, and target
              class).
        """
        batch_idx = target_item_ids[i]
        item = data_list[batch_idx]
        item_key = explainer_type + "_" + target_layer + "_class" + str(target_class)
        cam = cams_dict[item_key][batch_idx]

        item_dims = item.shape
        if len(item_dims) == 3:
            if item_dims[0] == 1:
                item = item[0]
            elif item_dims[2] == 1:
                item = item[:, :, 0]
        elif len(item_dims) == 1:
            item = item[np.newaxis, :]

        if cam.shape[1] == 1 or cam.shape[1] > 1 and self.time_axs == 0:
            item = np.transpose(item)
            cam = np.transpose(cam)

        return cam, item, batch_idx, item_key

    def __set_axes(self, cam: np.ndarray, data_sampling_freq: float, dt: float, channel_names: List[str | float],
                   time_names: List[str | float], axes_names: Tuple[str | None, str | None] = None,
                   only_x: bool = False) -> None:
        """
        Sets the axes in the plot, including both tick marks and labels.

        :param cam: (mandatory) The CAM for the given setting (defined by algorithm, target layer, and target class),
            corresponding to the i-th item in the selected ones.
        :param data_sampling_freq: (mandatory) A numerical value representing the sampling frequency of signal inputs,
            in samples per second.
        :param dt: (mandatory) A numerical value representing the granularity of the time axis in seconds in the output
            display.
        :param channel_names: (mandatory) A list of strings where each string represents the name of a signal channel
            for tick settings.
        :param time_names: (mandatory) A list of strings representing tick names for the time axis.
        :param axes_names: (optional, default is None) A tuple of strings representing names for X and Y axes,
            respectively.
        :param only_x: (optional, default is False) A boolean flag indicating whether only the X axis should be set.
        """

        # Set X-axis
        if data_sampling_freq is not None:
            time_steps, points = self.__get_time_axis(cam, data_sampling_freq, dt)
            plt.xticks(ticks=points, labels=time_steps, fontsize=8)
            plt.xlabel("Time (s)")
        elif data_sampling_freq is None and cam.shape[0] == 1:
            plt.xlabel("Sample index")
        else:
            if time_names is None:
                plt.xticks([], [])
            else:
                desired_times = len(time_names)
                map_times = cam.shape[1]
                desired_ticks = [c / desired_times * map_times for c in range(desired_times)]
                plt.xticks(desired_ticks, time_names, fontsize=8)

        # Set Y-axis
        if not only_x:
            if cam.shape[0] > 1 and channel_names is not None:
                desired_channels = len(channel_names)
                map_channels = cam.shape[0]
                desired_ticks = [c / desired_channels * map_channels for c in range(desired_channels)]
                plt.yticks(desired_ticks, channel_names, rotation=0, fontsize=7)
                plt.ylabel("Signal channels")
            else:
                plt.yticks([], [])

        if axes_names is not None:
            if axes_names[0] is not None:
                plt.xlabel(axes_names[0])
            if axes_names[1] is not None:
                plt.ylabel(axes_names[1])

    def __get_cam_title(self, item_name: str, target_class: int, data_labels: List[int], batch_idx: int, item_key: str,
                        predicted_probs: Dict[str, np.ndarray]) -> str:
        """
        Builds the CAM title for a given item and target class.

        :param item_name: (mandatory) A string representing the name of an input item.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param data_labels: (mandatory) A list of integers representing the true labels of the data to be explained.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.
        :param item_key: (mandatory) A string representing the considered setting (defined by algorithm, target layer,
            and target class).
        :param predicted_probs: (mandatory) A np.ndarray, representing the inferred class probabilities for each item in
            the input list.

        :return:
            - title: A string representing the title of the CAM for a given item and target class.
        """

        title = ("'" + item_name + "': CAM for class '" + self.class_names[target_class] + "' (confidence = " +
                 str(np.round(predicted_probs[item_key][batch_idx] * 100, 2)) + "%) - true class " +
                 self.class_names[data_labels[batch_idx]])
        return title

    def __display_plot(self, results_dir_path: str, explainer_type: str, target_layer: str, target_class: int,
                       item_name: str = None, is_channel: bool = False) -> None:
        """
        Show one CAM plot as a figure or as a ".png" file.

        :param results_dir_path: (mandatory)  A string representing the relative path to the directory for storing
            results. If None, the output will be displayed in a figure.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules, TensorFlow/Keras layers, or it should be a class dictionary key,
            used to retrieve the layer from the class attributes.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param item_name: (optional, default is False) A string representing the name of an input item.
        :param is_channel: (optional, default is False) A boolean flag indicating whether the figure represents graphs
            of multiple input channels, to discriminate it from other display modalities.
        """

        if is_channel:
            name_addon = "channel_graphs_"
            descr_addon = "single-channel "
        elif item_name is None:
            name_addon = "results_"
            descr_addon = "overlapped "
        else:
            name_addon = ""
            descr_addon = ""

        if results_dir_path is not None:
            filepath = results_dir_path
            if item_name is not None:
                data_name = str(item_name).replace(" ", "_").lower()
                filepath = os.path.join(filepath, data_name)
                if data_name not in os.listdir(results_dir_path):
                    os.mkdir(filepath)
            filename = (name_addon + explainer_type + "_" + re.sub(r"\W", "_", target_layer) + "_class" +
                        str(target_class) + ".png")

            # Communicate outcome
            descr_addon1 = "for item '" + item_name + "' " if item_name is not None else ""
            self.__print_justify("Storing " + descr_addon + "output display " + descr_addon1 + "(class " +
                                 self.class_names[target_class] + ", layer " + target_layer + ", algorithm " + explainer_type +
                                 ") as '" + filename + "'...")

            plt.savefig(os.path.join(filepath, filename), format="png", bbox_inches="tight", pad_inches=0,
                        dpi=500)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _is_2d_layer(target_layer: torch.nn.Module | tf.keras.layers.Layer) -> bool:
        """
        Evaluates whether the target layer is a 2D-convolutional layer.

        :param target_layer: (mandatory) A PyTorch module or a TensorFlow/Keras layer.

        :return:
            - is_2d_layer: A boolean indicating whether the target layers 2D-convolutional layer.
        """

        raise ValueError(target_layer + " must be a 1D or 2D convolutional layer.")

    @staticmethod
    def __normalize_cams(cams: np.ndarray, is_2d_layer: bool) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Adjusts the CAMs in value range (0-255), and returns the original importance score range.

        :param cams: (mandatory) A np.ndarray representing a batch of raw CAMs, one per item in the input data batch.
        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.

        :return:
            - cams: A np.ndarray representing a batch of CAMs (normalised in the range 0-255), one per item in the input
            data batch.
            - bar_ranges: A tuple containing two np.ndarrays, corresponding to the minimum and maximum importance scores
            per CAM for each item in the input data list, based on a given setting (defined by algorithm, target
            layer, and target class).
        """

        if is_2d_layer:
            axis = (1, 2)
        else:
            axis = 1
        maxima = np.max(cams, axis=axis, keepdims=True)
        minima = np.min(cams, axis=axis, keepdims=True)

        is_uniform = maxima == minima
        cams = np.where(is_uniform,
                        np.ones_like(cams) * np.where(maxima == 1, 1, 0),
                        np.divide(cams - minima, maxima - minima, where=(maxima - minima) != 0))

        cams = np.uint8(255 * cams)
        bar_ranges = (minima, maxima)
        return cams, bar_ranges

    @staticmethod
    def __get_time_axis(cam: np.ndarray, data_sampling_freq: float, dt: float = 10) -> Tuple[List[str], np.ndarray]:
        """
        Returns the X axis ticks for a given CAM.

        :param cam: (mandatory) A np.ndarray representing a CAM.
        :param data_sampling_freq: (mandatory) A numerical value representing the sampling frequency of signal inputs,
            in samples per second.
        :param dt: (optional, default is 10) A numerical value representing the granularity of the time axis in seconds
            in the output display.

        :return:
            - time_steps: A list containing the X axis ticks, in seconds.
            - points: A list containing the X axis ticks, in number of samples.
        """

        max_p = cam.shape[1]
        points = np.arange(0, max_p + 1, np.ceil(dt * data_sampling_freq))
        time_steps = [str(p / data_sampling_freq) for p in points]

        return time_steps, points

    @staticmethod
    def __set_colorbar(bar_ranges: Tuple[np.ndarray, np.ndarray] = None, batch_idx: int = None) -> None:
        """
        Sets the colorbar describing a CAM, representing extreme colors as minimum and maximum importance score values.

        :param bar_ranges: (optional, default is None) A tuple containing two np.ndarrays, corresponding to the minimum
            and maximum importance scores per CAM for each item in the input data list, based on a given setting
            (defined by algorithm, target layer, and target class).
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.
        """

        bar_range = [bar_ranges[0][batch_idx], bar_ranges[1][batch_idx]]
        cbar = plt.colorbar()
        if bar_range is not None:
            minimum = float(bar_range[0])
            maximum = float(bar_range[1])
            min_str = str(minimum) if minimum == 0 else "{:.2e}".format(minimum)
            max_str = "" if maximum == minimum else "{:.2e}".format(maximum)
            cbar.ax.get_yaxis().set_ticks([cbar.vmin, cbar.vmax], labels=[min_str, max_str])

    @staticmethod
    def __check_input_types(target_classes: int | List[int], explainer_types: str | List[str],
                            target_layers: str | List[str]) -> Tuple[List[int], List[str], List[str]]:
        """
        Checks whether the setting specifics (target classes, explainer algorithms, and target layers) are provided
        as lists of values. If not, they are transformed into a list.

        :param target_classes: (mandatory) An integer or a list of integers representing the target classes for the
            explanation.
        :param explainer_types: (mandatory) A string or a list of strings representing the desired algorithms for the
            explanation. These strings should identify one of the CAM algorithms allowed, as listed by the class
            constructor.
        :param target_layers: (mandatory) A string or a list of strings representing the target layers for the
            explanations. These strings should identify either PyTorch named modules, TensorFlow/Keras layers, or they
            should be class dictionary keys, used to retrieve each layer from the class attributes.

        :return:
            - target_classes: A list of integers representing the target classes for the explanation.
            - explainer_types: A list of strings representing the desired algorithms for the explanation. These strings
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
            - target_layers: A list of strings representing the target layers for the explanations. These strings should
            identify either PyTorch named modules, TensorFlow/Keras layers, or they should be class dictionary keys,
            used to retrieve each layer from the class attributes.
        """

        if not isinstance(target_classes, list):
            target_classes = [target_classes]
        if not isinstance(explainer_types, list):
            explainer_types = [explainer_types]
        if not isinstance(target_layers, list):
            target_layers = [target_layers]

        return target_classes, explainer_types, target_layers

    @staticmethod
    def __set_grid(n_items: int, grid_instructions: Tuple[int, int]) -> Tuple[int, int]:
        """
        Computes number of columns (width) and number of rows (height) for the tabular layout of the figure subplots.

        :param n_items: (mandatory) The number of target item among the items in the input data list.
        :param grid_instructions: (optional, default is None) A tuple of integers defining the desired tabular layout
            for figure subplots. The expected format is number of columns (width) x number of rows (height).

        :return:
            - w: The number of columns (width) of the desired tabular layout for figure subplots.
            - h: The number of rows (height) of the desired tabular layout for figure subplots.
        """

        if grid_instructions is not None:
            w, h = grid_instructions
        else:
            w, h = n_items, 1

        return w, h

    @staticmethod
    def __get_norm(cam: np.ndarray) -> m_colors.Normalize | None:
        """
       Determines the eventual normalization for the given CAM. The objective is to ensure that meaningless CAMs (with
       all zero values) receive a blue coloration.

        :param cam: (mandatory) A np.ndarray representing a CAM.

        :return:
            - norm: A matplotlib.colors.Normalize object if the CAM has all zero values, or None.
        """

        if len(np.unique(cam)) == 1 and np.unique(cam) == 0:
            norm = m_colors.Normalize(vmin=0, vmax=255)
        else:
            norm = None

        return norm

    @staticmethod
    def __print_justify(text: str, n_characters: int = 100) -> None:
        """
        Prints a message in a fully justified format within a specified line width.

        :param text: (mandatory) Text string to be displayed.
        :param n_characters: (optional, default is 100) The number of characters allowed per line.
        """
        text = "\n".join(text[i:i + n_characters] for i in range(0, len(text), n_characters))
        print(text)
