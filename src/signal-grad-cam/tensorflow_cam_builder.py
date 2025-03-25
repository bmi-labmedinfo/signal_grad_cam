# Import dependencies
import os
os.environ["PYTHONHASHSEED"] = "11"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import keras
import tensorflow as tf
from typing import Callable, List, Tuple, Dict, Any

from easy_grad_cam import CamBuilder


# Class
class TfCamBuilder(CamBuilder):
    """
    Represents a TensorFlow/Keras Class Activation Map (CAM) builder, supporting multiple methods such as Grad-CAM and
    HiResCAM.
    """

    def __init__(self, model: tf.keras.Model | Any, transform_fn: Callable = None, class_names: List[str] = None,
                 time_axs: int = 1, input_transposed: bool = False, model_output_index: int = None, seed: int = 11):
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
        :param model_output_index: (optional, default is None) An integer index specifying which of the model's outputs
            represents output scores (or probabilities). If there is only one output, this argument can be ignored.
        :param seed: (optional, default is 11) An integer seed for random number generators, used to ensure
            reproducibility during model evaluation.
        """

        # Initialize attributes
        super(TfCamBuilder, self).__init__(model, transform_fn, class_names, time_axs, input_transposed,
                                           model_output_index, seed=seed)

        # Set seeds
        tf.random.set_seed(seed)

    def _get_layers_pool(self, show: bool = False) -> Dict[str, tf.keras.layers.Layer | Any]:
        """
        Retrieves a dictionary containing all the available TensorFlow/Keras layers (or instance attributes), with the
        layer (or attribute) names used as keys.

        :param show: (optional, default is False) A boolean flag indicating whether to display the retrieved layers
            along with their names.

        :return:
            - layers_pool: A dictionary storing the model's TensorFlow/Keras layers (or instance attributes), with layer
            (or attribute) names as keys.
        """

        if hasattr(self.model, "layers"):
            layers_pool = {layer.name: layer for layer in self.model.layers}
            if show:
                for name, layer in layers_pool.items():
                    self._show_layer(name, layer)
        else:
            layers_pool = super()._get_layers_pool(show)

        return layers_pool

    def _show_layer(self, name: str, layer: tf.keras.layers.Layer | Any, potential: bool = False) -> None:
        """
        Displays a single available layer (or instance attribute) in the model, along with its corresponding name.

        :param name: (mandatory) A string representing the name of the layer or attribute.
        :param layer: (mandatory) A TensorFlow/Keras layer, or an instance attribute in the model.
        :param potential: (optional, default is False) A flag indicating whether the object displayed is potentially
            a layer (i.e., a generic instance attribute, not guaranteed to be a layer).
        """

        if (isinstance(layer, keras.layers.Conv1D) or isinstance(layer, keras.layers.Conv2D) or
                isinstance(layer, keras.layers.Softmax)):
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
        if not isinstance(data_list[0], tf.Tensor):
            data_list = [tf.convert_to_tensor(x) for x in data_list]

        is_2d_layer = self._is_2d_layer(target_layer)
        if is_2d_layer and len(data_list[0].shape) == 2 or not is_2d_layer and len(data_list[0].shape) == 1:
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
