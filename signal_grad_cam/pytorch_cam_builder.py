# Import dependencies
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Dict, Any, Optional

from signal_grad_cam import CamBuilder


# Class
class TorchCamBuilder(CamBuilder):
    """
    Represents a PyTorch Class Activation Map (CAM) builder, supporting multiple methods such as Grad-CAM and HiResCAM.
    """

    def __init__(self, model: nn.Module | Any, transform_fn: Callable[[np.ndarray, *tuple[Any, ...]], torch.Tensor]
                 = None, class_names: List[str] = None, time_axs: int = 1, input_transposed: bool = False,
                 ignore_channel_dim: bool = False, model_output_index: int = None, extend_search: bool = False,
                 use_gpu: bool = False, padding_dim: int = None, seed: int = 11):
        """
        Initializes the TorchCamBuilder class. The constructor also displays, if present and retrievable, the 1D- and
        2D-convolutional layers in the network, as well as the final Sigmoid/Softmax activation. Additionally, the CAM
        algorithms available for generating the explanations are shown.

        :param model: (mandatory) A torch.nn.Module or any object (with PyTorch layers among its attributes)
            representing a convolutional neural network model to be explained. Unconventional models should always be
            set to inference mode before being provided as inputs.
        :param transform_fn: (optional, default is None) A callable function to preprocess np.ndarray data before model
            evaluation. This function is also expected to convert data into PyTorch tensors.The function may optionally
            take as a second input a list of objects required by the preprocessing method.
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
        :param use_gpu: (optional, default is False) A boolean flag indicating whether to use GPU for data processing,
            if GPU is available.
        :param padding_dim: (optional, default is None) An integer specifying the maximum length along the time axis to
            which each item will be padded for batching.
        :param seed: (optional, default is 11) An integer seed for random number generators, used to ensure
            reproducibility during model evaluation.
        """

        # Initialize attributes
        super(TorchCamBuilder, self).__init__(model=model, transform_fn=transform_fn, class_names=class_names,
                                              time_axs=time_axs, input_transposed=input_transposed,
                                              ignore_channel_dim=ignore_channel_dim,
                                              model_output_index=model_output_index, extend_search=extend_search,
                                              padding_dim=padding_dim, seed=seed)

        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Check for the evaluation mode method
        if hasattr(model, "eval"):
            self.model.eval()
        else:
            print("Your PyTorch model has no 'eval' method. Please verify that the networks has been set to "
                  "evaluation mode before the TorchCamBuilder initialization.")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        # Assign the default transform function
        if transform_fn is None:
            self.transform_fn = self.__default_transform_fn

    def _get_layers_pool(self, show: bool = False, extend_search: bool = False) \
            -> Dict[str, torch.nn.Module | Any]:
        """
        Retrieves a dictionary containing all the available PyTorch layers (or instance attributes), with the layer (or
        attribute) names used as keys.

        :param show: (optional, default is False) A boolean flag indicating whether to display the retrieved layers
            along with their names.
        :param extend_search: (optional, default is False) A boolean flag indicating whether to deepend the search for
            candidate layers. It should be set true if no convolutional layer was found.

        :return:
            - layers_pool: A dictionary storing the model's PyTorch layers (or instance attributes),
            with layer (or attribute) names as keys.
        """

        if hasattr(self.model, "named_modules"):
            layers_pool = dict(self.model.named_modules())
            if show:
                for name, layer in layers_pool.items():
                    self._show_layer(name, layer)
        else:
            layers_pool = super()._get_layers_pool(show=show)

        if extend_search:
            layers_pool.update(super()._get_layers_pool(show=show))

        return layers_pool

    def _show_layer(self, name: str, layer: nn.Module | Any, potential: bool = False):
        """
        Displays a single available layer (or instance attribute) in the model, along with its corresponding name.

        :param name: (mandatory) A string representing the name of the layer or attribute.
        :param layer: (mandatory) A PyTorch layer or an instance attribute in the model.
        :param potential: (optional, default is False) A flag indicating whether the object displayed is potentially
            a layer (i.e., a generic instance attribute, not guaranteed to be a layer).
        """

        if (isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d) or
                isinstance(layer, nn.Softmax) or isinstance(layer, nn.Sigmoid)):
            super()._show_layer(name, layer, potential=potential)

    def _create_raw_batched_cams(self, data_list: List[np.ndarray | torch.Tensor], target_class: int,
                                 target_layer: nn.Module, explainer_type: str, softmax_final: bool,
                                 extra_inputs_list: List[Any] = None)  -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Retrieves raw CAMs from an input data list based on the specified settings (defined by algorithm, target layer,
        and target class). Additionally, it returns the class probabilities predicted by the model.

        :param data_list: (mandatory) A list of np.ndarrays to be explained, representing either a signal or an image.
        :param target_class: (mandatory) An integer representing the target class for the explanation.
        :param target_layer: (mandatory) A string representing the target layer for the explanation. This string should
            identify either PyTorch named modules or it should be a class dictionary key, used to retrieve the layer
            from the class attributes.
        :param explainer_type: (mandatory) A string representing the desired algorithm for the explanation. This string
            should identify one of the CAM algorithms allowed, as listed by the class constructor.
        :param softmax_final: (mandatory) A boolean indicating whether the network terminates with a Sigmoid/Softmax
            activation function.
        :param extra_inputs_list: (optional, defaults is None) A list of additional input objects required by the
            model's forward method.

        :return:
            - cam_list: A list of np.ndarray containing CAMs for each item in the input data list, corresponding to the
                given setting (defined by algorithm, target layer, and target class).
            - target_probs: A np.ndarray, representing the inferred class probabilities for each item in the input list.
        """

        # Register hooks
        _ = target_layer.register_forward_hook(self.__get_activation_forward_hook, prepend=False)
        _ = target_layer.register_forward_hook(self.__get_gradient_forward_hook, prepend=False)

        # Data batching
        if not isinstance(data_list[0], torch.Tensor):
            data_list = [torch.Tensor(x) for x in data_list]
        if self.padding_dim is not None:
            padded_data_list = []
            for item in data_list:
                pad_size = self.padding_dim - item.shape[self.time_axs]
                if not self.time_axs:
                    zeros = torch.zeros((pad_size, item.shape[1]), dtype=item.dtype,
                                        device=item.device)
                else:
                    zeros = torch.zeros((item.shape[0], pad_size), dtype=item.dtype,
                                        device=item.device)
                padded_data_list.append(torch.cat((item, zeros), dim=self.time_axs))
            data_list = padded_data_list

        is_2d_layer = self._is_2d_layer(target_layer)
        if not self.ignore_channel_dim and (is_2d_layer and len(data_list[0].shape) == 2 or not is_2d_layer
                                            and len(data_list[0].shape) == 1):
            data_list = [x.unsqueeze(0) for x in data_list]
        data_batch = torch.stack(data_list)

        # Set device
        self.model = self.model.to(self.device)
        data_batch = data_batch.to(self.device)

        extra_inputs_list = extra_inputs_list or []
        outputs = self.model(data_batch, *extra_inputs_list)
        if isinstance(outputs, tuple):
            outputs = outputs[self.model_output_index]

        if softmax_final:
            # Approximate Softmax inversion formula logit = log(prob) + constant, as the constant is negligible
            # during derivation
            target_scores = torch.log(outputs)
            target_probs = outputs

            # Adjust results for binary network
            if len(outputs.shape) == 1:
                target_scores = torch.stack([-target_scores, target_scores], dim=1)
                target_probs = torch.stack([1 - target_probs, target_probs], dim=1)
            elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                target_scores = torch.cat([-target_scores, target_scores], dim=1)
                target_probs = torch.cat([1 - target_probs, target_probs], dim=1)
        else:
            if len(outputs.shape) == 1:
                outputs = torch.stack([-outputs, outputs], dim=1)
            elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                outputs = torch.cat([-outputs, outputs], dim=1)
            target_scores = outputs

            if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                target_probs = torch.softmax(target_scores, dim=1)
            else:
                tmp = torch.sigmoid(target_scores[:, 1])
                target_probs = torch.stack([1 - tmp, tmp], dim=1)

        target_probs = target_probs[:, target_class].cpu().detach().numpy()

        cam_list = []
        for i in range(len(data_list)):
            self.model.zero_grad()
            target_score = target_scores[i, target_class]
            target_score.backward(retain_graph=True)

            if explainer_type == "HiResCAM":
                cam = self._get_hirescam_map(is_2d_layer=is_2d_layer, batch_idx=i)
            else:
                cam = self._get_gradcam_map(is_2d_layer=is_2d_layer, batch_idx=i)
            cam_list.append(cam.cpu().detach().numpy())

        return cam_list, target_probs

    def _get_gradcam_map(self, is_2d_layer: bool, batch_idx: int) -> torch.Tensor:
        """
        Compute the CAM using the vanilla Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A PyTorch tensor representing the Class Activation Map (CAM) for the batch_idx-th input, built
            with the Grad-CAM algorithm.
        """

        if is_2d_layer:
            dim_mean = (1, 2)
        else:
            dim_mean = 1
        weights = torch.mean(self.gradients[batch_idx], dim=dim_mean)
        activations = self.activations[batch_idx].clone()

        for i in range(self.activations.shape[1]):
            if is_2d_layer:
                activations[i, :, :] *= weights[i]
            else:
                activations[i, :] *= weights[i]

        cam = torch.sum(activations, dim=0)
        cam = torch.relu(cam)
        return cam

    def _get_hirescam_map(self, is_2d_layer: bool, batch_idx: int) -> torch.Tensor:
        """
        Compute the CAM using the High-Resolution Class Activation Mapping (HiResCAM) algorithm.

        :param is_2d_layer: (mandatory) A boolean indicating whether the target layers 2D-convolutional layer.
        :param batch_idx: (mandatory) The index corresponding to the i-th selected item within the original input data
            list.

        :return: cam: A PyTorch tensor representing the Class Activation Map (CAM) for the batch_idx-th input, built
            with the HiResCAM algorithm.
        """

        activations = self.activations[batch_idx].clone()
        gradients = self.gradients[batch_idx]

        for i in range(self.activations.shape[1]):
            if is_2d_layer:
                activations[i, :, :] *= gradients[i, :, :]
            else:
                activations[i, :] *= gradients[i, :]

        cam = torch.sum(activations, dim=0)
        cam = torch.relu(cam)
        return cam

    def __get_activation_forward_hook(self, layer: nn.Module, inputs: Tuple[torch.Tensor, ...], outputs: torch.Tensor) \
            -> None:
        """
        Defines the forward hook function for capturing intermediate activations during model inference.

        :param layer: (mandatory) The target PyTorch layer where the hook is attached.
        :param inputs: (mandatory) A tuple containing the input tensors received by the layer.
        :param outputs: (mandatory) The output tensor produced by the layer after applying its operations.
        """

        self.activations = outputs

    def __get_gradient_forward_hook(self, layer: nn.Module, inputs: Tuple[torch.Tensor, ...], outputs: torch.Tensor) \
            -> None:
        """
        Defines the forward hook function for capturing intermediate gradients during model inference.

        :param layer: (mandatory) The target PyTorch layer where the hook is attached.
        :param inputs: (mandatory) A tuple containing the input tensors received by the layer.
        :param outputs: (mandatory) The output tensor produced by the layer after applying its operations.
        """

        outputs.register_hook(self.__store_grad)

    def __store_grad(self, gradients: torch.Tensor) -> None:
        """
        Captures intermediate gradients during backpropagation.

        :param gradients: (mandatory) A tensor containing the gradients of the layer's outputs, computed during
        backpropagation.
        """

        self.gradients = gradients

    @staticmethod
    def _is_2d_layer(target_layer: nn.Module) -> bool:
        """
        Evaluates whether the target layer is a 2D-convolutional layer.

        :param target_layer: (mandatory) A PyTorch module.

        :return:
            - is_2d_layer: A boolean indicating whether the target layers 2D-convolutional layer.
        """

        if isinstance(target_layer, nn.Conv1d):
            is_2d_layer = False
        elif isinstance(target_layer, nn.Conv2d):
            is_2d_layer = True
        else:
            is_2d_layer = CamBuilder._is_2d_layer(target_layer)
        return is_2d_layer

    @staticmethod
    def __default_transform_fn(np_input: np.ndarray) -> torch.Tensor:
        """
        Converts a NumPy array to a PyTorch tensor with float type.

        :param np_input: (mandatory) A NumPy array representing the input data.

        :return: A PyTorch tensor converted from the input NumPy array, with float data type.
        """

        torch_input = torch.from_numpy(np_input).float()
        return torch_input
