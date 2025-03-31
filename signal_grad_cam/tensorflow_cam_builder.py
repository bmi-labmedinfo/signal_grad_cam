# Import dependencies
import os
os.environ["PYTHONHASHSEED"] = "11"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import keras
import tensorflow as tf
from typing import Callable, List, Tuple, Dict, Any

from signal_grad_cam import CamBuilder


# Class
class TfCamBuilder(CamBuilder):
    """
    Represents a TensorFlow/Keras Class Activation Map (CAM) builder, supporting multiple methods such as Grad-CAM and
    HiResCAM.
    """

    def __init__(self, model: tf.keras.Model | Any, transform_fn: Callable = None, class_names: List[str] = None,
                 time_axs: int = 1, input_transposed: bool = False, ignore_channel_dim: bool = False,
                 model_output_index: int = None, extend_search: bool = False, padding_dim: int = None, seed: int = 11):
        """
        Initializes the TfCamBuilder class. The constructor also displays, if present and retrievable, the 1D- and
        2D-convolutional layers in the network, as well as the final Sigmoid/Softmax activation. Additionally, the CAM
        algorithms available for generating the explanations are shown.

        :param model: (mandatory) A tf.keras.Model or any object (with TensorFlow/Keras layers among its attributes)
            representing a convolutional neural network model to be explained. Unconventional models should always be
            set to inference mode before being provided as inputs.
        :param transform_fn: (optional, default is None) A callable function to preprocess np.ndarray data before model
            evaluation. This function is also expected to convert data into either PyTorch or TensorFlow tensors.
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

        # Initialize attributes
        super(TfCamBuilder, self).__init__(model=model, transform_fn=transform_fn, class_names=class_names,
                                           time_axs=time_axs, input_transposed=input_transposed,
                                           ignore_channel_dim=ignore_channel_dim, model_output_index=model_output_index,
                                           extend_search=extend_search, padding_dim=padding_dim, seed=seed)

        # Set seeds
        tf.random.set_seed(seed)

        # Check for input/output attributes
        if not hasattr(model, "inputs"):
            print("Your TensorFlow/Keras model has no attribute 'inputs'. Ensure it is built or loaded correctly, or\n"
                  "provide a different one. If the model contains a 'Sequential' attribute, that Sequential object may\n"
                  "be a suitable candidate for an input model.")
        elif not hasattr(model, "output"):
            print("Your TensorFlow/Keras model has no attribute 'output'. Ensure it is built or loaded correctly, or\n"
                  "provide a different one. If the model contains a 'Sequential' attribute, that Sequential object may\n"
                  "be a suitable candidate for an input model.")

    def _get_layers_pool(self, show: bool = False, extend_search: bool = False) \
            -> Dict[str, tf.keras.layers.Layer | Any]:
        """
        Retrieves a dictionary containing all the available TensorFlow/Keras layers (or instance attributes), with the
        layer (or attribute) names used as keys.

        :param show: (optional, default is False) A boolean flag indicating whether to display the retrieved layers
            along with their names.
        :param extend_search: (optional, default is False) A boolean flag indicating whether to deepend the search for
            candidate layers. It should be set true if no convolutional layer was found.

        :return:
            - layers_pool: A dictionary storing the model's TensorFlow/Keras layers (or instance attributes), with layer
            (or attribute) names as keys.
        """

        if hasattr(self.model, "layers"):
            layers_pool = {layer.name: layer for layer in self.model.layers}
            if show:
                for name, layer in layers_pool.items():
                    self._show_layer(name, layer)
            layers_pool.update(self._get_sub_layers_pool(layers_pool, show=show))
        else:
            layers_pool = super()._get_layers_pool(show=show)
            layers_pool.update(self._get_sub_layers_pool(layers_pool, show=show))

        if extend_search:
            layers_pool.update(super()._get_layers_pool(show=show))

        return layers_pool

    def _get_sub_layers_pool(self, layers_pool: Dict[str, tf.keras.layers.Layer | Any], show: bool = False) \
            -> Dict[str, tf.keras.layers.Layer | Any]:
        """
        Retrieves a dictionary containing all the available TensorFlow/Keras layers (or instance attributes), with the
        layer (or attribute) names used as keys.

        :param show: (optional, default is False) A boolean flag indicating whether to display the retrieved layers
            along with their names.
        :param layers_pool: (mandatory) A dictionary storing the model's TensorFlow/Keras layers (or instance
            attributes), with layer (or attribute) names as keys.

        :return:
            - sub_layers_pool: A dictionary storing the model's TensorFlow/Keras layers (or instance attributes), with
            layer (or attribute) names as keys, enhanced with sub-layers formerly encapsulated in keras.Sequential
            objects.
        """

        sub_layers_pool = {}
        for name, layer in layers_pool.items():
            if isinstance(layer, keras.Sequential):
                sub_layers_pool.update({name + "." + sub_layer.name: sub_layer for sub_layer in layer.layers})
        if show:
            for name, layer in sub_layers_pool.items():
                self._show_layer(name, layer)

        return sub_layers_pool

    def _show_layer(self, name: str, layer: tf.keras.layers.Layer | Any, potential: bool = False) -> None:
        """
        Displays a single available layer (or instance attribute) in the model, along with its corresponding name.

        :param name: (mandatory) A string representing the name of the layer or attribute.
        :param layer: (mandatory) A TensorFlow/Keras layer, or an instance attribute in the model.
        :param potential: (optional, default is False) A flag indicating whether the object displayed is potentially
            a layer (i.e., a generic instance attribute, not guaranteed to be a layer).
        """

        if (isinstance(layer, keras.layers.Conv1D) or isinstance(layer, keras.layers.Conv2D) or
                isinstance(layer, keras.layers.Softmax) or isinstance(layer, keras.Sequential)):
            super()._show_layer(name, layer, potential=potential)

    def _create_raw_batched_cams(self, data_list: List[np.array], target_class: int,
                                 target_layer: tf.keras.layers.Layer, explainer_type: str, softmax_final: bool) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Retrieves raw CAMs from an input data list based on the specified settings (defined by algorithm, target layer,
        and target class). Additionally, it returns the class probabilities predicted by the model.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either TensorFlow/Keras layers or it should be a class dictionary key, used to retrieve the layer
            from the class attributes.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param softmax_final: (mandatory) A boolean indicating whether the network terminates with a Sigmoid/Softmax
            activation function.

        :return:
            - cam_list: A list of np.ndarray containing CAMs for each item in the input data list, corresponding to the
                given setting (defined by algorithm, target layer, and target class).
            - target_probs: A np.ndarray, representing the inferred class probabilities for each item in the input list.
        """

        # Data batching
        if not isinstance(data_list[0], tf.Tensor):
            data_list = [tf.convert_to_tensor(x) for x in data_list]
        if self.padding_dim is not None:
            padded_data_list = []
            for item in data_list:
                pad_size = self.padding_dim - tf.shape(item)[self.time_axs]
                if not self.time_axs:
                    zeros = tf.zeros((pad_size, tf.shape(item)[1]), dtype=item.dtype)
                else:
                    zeros = tf.zeros((tf.shape(item)[0], pad_size), dtype=item.dtype)
                padded_data_list.append(tf.concat([item, zeros], axis=self.time_axs))
            data_list = padded_data_list

        is_2d_layer = self._is_2d_layer(target_layer)
        if not self.ignore_channel_dim and (is_2d_layer and len(data_list[0].shape) == 2 or not is_2d_layer
                                            and len(data_list[0].shape) == 1):
            data_list = [tf.expand_dims(x, axis=0) for x in data_list]
        data_batch = tf.stack(data_list, axis=0)

        grad_model = keras.models.Model(self.model.inputs[0], [target_layer.output, self.model.output])
        with tf.GradientTape() as tape:
            self.activations, outputs = grad_model(data_batch)

            if softmax_final:
                # Approximate Softmax inversion formula logit = log(prob) + constant, as the constant is negligible
                # during derivation
                target_scores = tf.math.log(outputs)
                target_probs = outputs
            else:
                target_scores = outputs
                target_probs = tf.nn.softmax(target_scores, axis=1)

            target_scores = target_scores[:, target_class]
            target_probs = target_probs[:, target_class]
            self.gradients = tape.gradient(target_scores, self.activations)

        cam_list = []
        is_2d_layer = self._is_2d_layer(target_layer)
        for i in range(len(data_list)):
            if explainer_type == "HiResCAM":
                cam = self._get_hirecam_map(is_2d_layer=is_2d_layer, batch_idx=i)
            else:
                cam = self._get_gradcam_map(is_2d_layer=is_2d_layer, batch_idx=i)
            cam_list.append(cam.numpy())

        return cam_list, target_probs

    def _get_gradcam_map(self, is_2d_layer: bool, batch_idx: int) -> tf.Tensor:
        """
        Compute the CAM using the vanilla Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A TensorFlow/Keras tensor representing the Class Activation Map (CAM) for the batch_idx-th input,
            built with the Grad-CAM algorithm.
        """

        if is_2d_layer:
            dim_mean = (0, 1)
        else:
            dim_mean = 0
        weights = tf.reduce_mean(self.gradients[batch_idx], axis=dim_mean)
        activations = self.activations[batch_idx].numpy()

        for i in range(activations.shape[-1]):
            if is_2d_layer:
                activations[:, :, i] *= weights[i]
            else:
                activations[:, i] *= weights[i]
                activations[:, i] *= weights[i]

        cam = tf.reduce_sum(tf.convert_to_tensor(activations), axis=-1)
        cam = tf.nn.relu(cam)
        return cam

    def _get_hirecam_map(self, is_2d_layer: bool, batch_idx: int) -> tf.Tensor:
        """
        Compute the CAM using the High-Resolution Class Activation Mapping (HiResCAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A TensorFlow/Keras tensor representing the Class Activation Map (CAM) for the batch_idx-th input,
            built with the HiResCAM algorithm.
        """

        activations = self.activations[batch_idx].numpy()
        gradients = self.gradients[batch_idx].numpy()

        for i in range(activations.shape[-1]):
            if is_2d_layer:
                activations[:, :, i] *= gradients[:, :, i]
            else:
                activations[:, i] *= gradients[:, i]

        cam = tf.reduce_sum(tf.convert_to_tensor(activations), axis=-1)
        cam = tf.nn.relu(cam)
        return cam

    @staticmethod
    def _is_2d_layer(target_layer: tf.keras.layers.Layer) -> bool:
        """
        Evaluates whether the target layer is a 2D-convolutional layer.

        :param target_layer: (mandatory) A TensorFlow/Keras layer.

        :return:
            - is_2d_layer: A boolean indicating whether the target layers 2D-convolutional layer.
        """

        if isinstance(target_layer, keras.layers.Conv1D):
            is_2d_layer = False
        elif isinstance(target_layer, keras.layers.Conv2D):
            is_2d_layer = True
        else:
            is_2d_layer = CamBuilder._is_2d_layer(target_layer)
        return is_2d_layer
