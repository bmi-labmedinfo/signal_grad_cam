'''

The PyTorch model used in this script was obtained from TorchVision: pytorch.org/vision/stable/models.html

The TensorFlow/Keras model used in this script was obtained from Keras Applications: keras.io/api/applications/

The image data used in this script were available online:
    - "dog_cat.jpg": github.com/jacobgil/pytorch-grad-cam
    - "english_setter.jpg": wikipedia.org/wiki/English_Setter
    - "goldfishes.jpg": wikipedia.org/wiki/Common_goldfish

'''

# Import dependencies
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import requests
import keras.api.applications.resnet50 as tf_resnet

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from signal_grad_cam import TorchCamBuilder, TfCamBuilder


# Load Imagenet class labels
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(url)
imagenet_classes = response.json()
imagenet_classes = [val[1] for val in imagenet_classes.values()]

# Set results directory
working_dir = "./"
data_dir = working_dir + "data/datasets/image_data/"

# Load and preprocess data
data_names = ["english_setter", "goldfishes", "dog_cat"]
data_list = [cv2.cvtColor(cv2.imread(data_dir + data_name + ".jpg"), cv2.COLOR_BGR2RGB) for data_name in data_names]
data_labels = [212, 1, 207]

# Set SignalGrad-CAM variables
target_classes = [1, 207, 212, 281]
explainer_types = ["Grad-CAM", "HiResCAM"]


# Test functions
def pytorch_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_torch_image_model/"
    if "results_for_torch_image_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Define preprocessing function
    transform_fn = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    # Draw CAMs
    cam_builder = TorchCamBuilder(model, transform_fn=transform_fn, class_names=imagenet_classes)
    target_layers_names = "layer4.2.conv3"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=data_names, results_dir_path=results_dir)

    # Compare HiResCAM functioning in different input images
    comparison_class = 281
    comparison_algorithm = "HiResCAM"
    cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels,
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=comparison_class,
                                          target_layers=target_layers_names, data_names=data_names,
                                          grid_instructions=(3, 1), bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir)


def tensorflow_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_tf_image_model/"
    if "results_for_tf_image_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Define preprocessing function
    def transform_fn(image):
        image = cv2.resize(image, (224, 224))
        image = tf_resnet.preprocess_input(image)
        return image

    # Load the model
    model = tf_resnet.ResNet50(weights="imagenet")

    # Draw CAMs
    cam_builder = TfCamBuilder(model, transform_fn=transform_fn, class_names=imagenet_classes)
    target_layers_names = "conv5_block3_3_conv"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            data_names=data_names, results_dir_path=results_dir)

    # Compare Grad-CAM functioning in different input images
    comparison_class = 281
    comparison_algorithm = "Grad-CAM"
    cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels,
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=comparison_class,
                                          target_layers=target_layers_names, data_names=data_names,
                                          grid_instructions=(3, 1), bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir)


# Main
pytorch_model_testing()
print("\n===========================================================================================================\n")
tensorflow_model_testing()
