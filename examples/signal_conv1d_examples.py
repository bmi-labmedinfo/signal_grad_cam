'''
The PyTorch model used in this script was obtained from GitHub: github.com/hsd1503/resnet1d
Licensed under the Apache License, Version 2.0, January 2004. See apache.org/licenses/LICENSE-2.0 for details.
@inproceedings{hong2020holmes,
  title={HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units},
  author={Hong, Shenda and Xu, Yanbo and Khare, Alind and Priambada, Satria and Maher, Kevin and Aljiffry, Alaa and Sun,
          Jimeng and Tumanov, Alexey},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  pages={1614--1624},
  year={2020}
}
and the data come from the PhysioNet/CinC Challenge 2017: physionet.org/content/challenge-2017/1.0.0/

Due to the large file size, the PyTorch model and data are not included in this GitHub repository. However, they can be
retrieved directly from the source.
    - The model file, "model.pth", is available at github.com/hsd1503/resnet1d/tree/master/trained_model
      After downloading, rename the file and place it in the following directory within this project:
      ./data/models/ecg_model_1D_torch.pth
    - The data file, "challenge2017.pkl", is available at drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
      After downloading, rename the file and place it in the following directory within this project:
      ./data/datasets/signal_data_1D/ecg.pkl

The TensorFlow/Keras model used in this script was obtained from GitHub: github.com/ZFTurbo/classification_models_1D
and the data come from AudioSet: research.google.com/audioset/
Due to the large file size, the data files for the Tensorflow/Keras model evaluation are not included in this GitHub
repository. However, they can be retrieved directly from the source.
    - The first audio file, "-0RWZT-miFs.wav", is available kaggle.com/datasets/zfturbo/audioset-valid
      After downloading, rename the file and place it in the following directory within this project:
      ./data/datasets/signal_data_1D/people_car_keys.wav
    - The second audio file, "-0vPFx-wRRI.wav", is available at kaggle.com/datasets/zfturbo/audioset-valid
      After downloading, rename the file and place it in the following directory within this project:
      ./data/datasets/signal_data_1D/sing_fingersnap.wav
'''

# Import dependencies
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pickle
import librosa
import pandas as pd
import torch

from examples.net1d import MyDataset
from classification_models_1D.tfkeras import Classifiers

from signal_grad_cam import TorchCamBuilder, TfCamBuilder


# Set working directories
working_dir = "./"
data_dir = working_dir + "data/"


# Test functions
def pytorch_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_torch_signal1D_model/"
    if "results_for_torch_signal1D_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Read data
    with open(data_dir + "datasets/signal_data_1D/ecg.pkl", "rb") as f:
        data = pickle.load(f)
    ecg_classes = ["normal sinus rhythm", "atrial fibrillation", "other", "too noisy"]
    fc = 300

    # Preprocess data
    x = data["data"]
    y = []
    for l in data["label"]:
        if l == "N":
            y.append(0)
        elif l == "A":
            y.append(1)
        elif l == "O":
            y.append(2)
        elif l == "~":
            y.append(3)
    y = np.array(y)
    data = MyDataset(x, y)

    # Define attributes
    data_idx = [0, 3, 16, 21]
    data_list = [data.__getitem__(idx)[0] for idx in data_idx]
    data_labels = [0, 1, 2, 3]
    data_names = ["Signal " + str(idx) for idx in data_idx]

    # Set SignalGrad-CAM variables
    target_classes = [0, 1, 2, 3]
    explainer_types = ["Grad-CAM", "HiResCAM"]

    # Define preprocessing function
    def transform_fn(signal):
        signal = (signal - torch.mean(signal)) / torch.std(signal)
        return signal

    # Load the model
    model = torch.load(data_dir + "models/ecg_model_1D_torch.pth", weights_only=False)

    # Draw CAMs
    cam_builder = TorchCamBuilder(model, transform_fn=transform_fn, class_names=ecg_classes, time_axs=0)
    target_layers_names = "stage_list.6.block_list.3.conv3.conv"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            data_sampling_freq=fc)

    # Explain different input channels
    comparison_algorithm = "Grad-CAM"
    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_classes,
                                              target_layers=target_layers_names, data_names=data_names, fig_size=(8, 6),
                                              grid_instructions=(1, 1), bar_ranges_dict=bar_ranges,
                                              results_dir_path=results_dir, data_sampling_freq=fc, dt=10,
                                              line_width=0.5, marker_width=30, axes_names=(None, "Voltage (mV)"))


def tensorflow_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_tf_signal1D_model/"
    if "results_for_tf_signal1D_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Read data
    data_names = ["people_car_keys", "sing_fingersnap"]
    data_list = [librosa.load(data_dir + "datasets/signal_data_1D/" + data_name + ".wav", sr=None, mono=False)[0] for
                 data_name in data_names]
    audio_classes = pd.read_csv(data_dir + "datasets/signal_data_1D/class_labels_indices.csv", delimiter=",")
    audio_classes = list(audio_classes.display_name)
    fc = 44100
    data_labels = [379, 62]

    # Set SignalGrad-CAM variables
    target_classes = [379, 62, 0]
    explainer_types = ["Grad-CAM", "HiResCAM"]

    # Load the model
    ResNet18, preprocess_input = Classifiers.get("resnet18_pool8")
    model = ResNet18(input_shape=(441000, 2), weights="audioset")

    # Define preprocessing function
    def transform_fn(signal):
        signal = preprocess_input(signal)
        signal = np.transpose(signal)
        return signal

    # Draw CAMs
    cam_builder = TfCamBuilder(model, transform_fn=transform_fn, class_names=audio_classes, time_axs=1)
    target_layers_names = ["stage7_unit2_conv1"]
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            data_sampling_freq=fc, dt=1, aspect_factor=20)

    # Explain different input channels
    comparison_algorithm = "Grad-CAM"
    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_classes,
                                              target_layers=target_layers_names, data_names=data_names,
                                              fig_size=(8, 10), grid_instructions=(2, 1), bar_ranges_dict=bar_ranges,
                                              results_dir_path=results_dir, data_sampling_freq=fc, dt=10,
                                              line_width=0.05, marker_width=30,
                                              axes_names=(None, "Digital Audio Amplitude"))


# Main
#pytorch_model_testing()
print("\n===========================================================================================================\n")
tensorflow_model_testing()
