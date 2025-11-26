'''

The PyTorch model used in this script was obtained from GitHub:
    github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch
and the data come from Kaggle: kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

The TensorFlow/Keras model used in this script was obtained from GitHub: github.com/HaneenElyamani/ECG-classification
and the data come from PhysioNet: physionet.org/content/ptb-xl/1.0.1/

'''


# Import dependencies
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import wfdb
import torch
import tensorflow as tf

from tensorflow.keras.models import load_model
from data.models.ParallelModel import ParallelModel
#from data.models.RavdessDataLoad import ravdess_get_test_data, ravdess_get_test_label, ravdess_get_test_raw_data

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
    results_dir += "results_for_torch_signal2D_model/"
    if "results_for_torch_signal2D_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Read data
    data_idx = [1, 100, 200]
    data_list = [ravdess_get_test_raw_data(i) for i in data_idx]
    data_preprocessed = [ravdess_get_test_data(i) for i in data_idx]
    data_labels = [ravdess_get_test_label(i) for i in data_idx]
    data_shape_list = [datum.shape[1:] for datum in data_list]

    emotion_classes = ["surprised", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]
    data_names = [emotion_classes[y] for y in data_labels]
    channel_names = list(range(0, 129, 5))
    time_names = list(range(0, 3))
    axes_names = ["Time (s)", "Frequency (Hz)"]

    # Set SignalGrad-CAM variables
    target_classes = [0, 3, 5]
    explainer_types = ["Grad-CAM", "HiResCAM"]

    # Load the model
    model = ParallelModel(len(emotion_classes))
    model.load_state_dict(torch.load(data_dir + "models/emotion_model_2D_torch.pt",
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval()

    # Draw CAMs
    cam_builder = TorchCamBuilder(model, transform_fn=None, class_names=emotion_classes, time_axs=1,
                                  model_output_index=1)
    target_layers_names = "conv2Dblock.15"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_preprocessed, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            data_shape_list=data_shape_list,
                                                            channel_names=channel_names, time_names=time_names,
                                                            axes_names=axes_names)

    # Compare Grad-CAM functioning in different input signals
    comparison_class = 0
    comparison_algorithm = "Grad-CAM"
    cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels,
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=comparison_class,
                                          target_layers=target_layers_names, data_names=data_names, fig_size=(10, 7),
                                          grid_instructions=(3, 1), bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir, axes_names=axes_names)

    # Explain different input channels
    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_classes,
                                              target_layers=target_layers_names, data_names=data_names,
                                              fig_size=(12, 10), desired_channels=[0, 50, 125],
                                              bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                              axes_names=axes_names, line_width=0.5, marker_width=50)

    # Contrastive explainability: Why "happy", rather than "angry"?
    fact_class = 3
    contrastive_foil_class = 5
    cams, predicted_probs, bar_ranges = cam_builder.get_cam([data_preprocessed[1]], data_labels=[data_labels[1]],
                                                            target_classes=fact_class,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            data_names=[data_names[1]], results_dir_path=results_dir,
                                                            data_shape_list=[data_shape_list[1]],
                                                            channel_names=channel_names, time_names=time_names,
                                                            axes_names=axes_names,
                                                            contrastive_foil_classes=contrastive_foil_class)
    comparison_algorithm = "HiResCAM"
    cam_builder.overlapped_output_display(data_list=[data_list[1]], data_labels=[data_labels[1]],
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=fact_class,
                                          target_layers=target_layers_names, data_names=[data_names[1]],
                                          fig_size=(10, 7), grid_instructions=(3, 1), bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir, axes_names=axes_names,
                                          contrastive_foil_classes=contrastive_foil_class)
    cam_builder.single_channel_output_display(data_list=[data_list[1]], data_labels=[data_labels[1]],
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=fact_class,
                                              target_layers=target_layers_names, data_names=[data_names[1]],
                                              fig_size=(12, 10), desired_channels=[0, 50, 125],
                                              bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                              axes_names=axes_names, line_width=0.5, marker_width=50,
                                              contrastive_foil_classes=contrastive_foil_class)


def tensorflow_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_tf_signal2D_model/"
    if "results_for_tf_signal2D_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Read data
    data_names = ["norm", "left_ventr_hyp", "inf_myiocard_infarc"]
    data_list = [wfdb.rdsamp(data_dir + "datasets/signal_data_2D/" + f)[0] for f in data_names]
    ecg_classes = ["conduction disturbances", "hypertrophy", "myocardial infarction", "normal", "ST-T changes"]
    fc = 100
    ecg_leads = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    data_labels = [3, 1, 2]

    # Set SignalGrad-CAM variables
    target_classes = [1, 2, 3]
    explainer_types = ["Grad-CAM", "HiResCAM"]
    # n_items = len(data_names)

    # Load the model
    model = load_model(data_dir + "models/ecg_model_2D_tf.h5", compile=False)

    # Define preprocessing function and make inference
    def transform_fn(signal):
        signal = np.transpose(signal)
        signal = tf.convert_to_tensor(signal)
        signal = tf.expand_dims(signal, -1)
        return signal

    # Draw CAMs
    cam_builder = TfCamBuilder(model, transform_fn=transform_fn, class_names=ecg_classes, time_axs=0,
                               input_transposed=True)
    target_layers_names = "conv2d_34"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            data_sampling_freq=fc, dt=1, channel_names=ecg_leads)

    # Explain different input channels
    comparison_algorithm = "Grad-CAM"
    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_classes,
                                              target_layers=target_layers_names, data_names=data_names,
                                              fig_size=(15, 18), grid_instructions=(4, 3), bar_ranges_dict=bar_ranges,
                                              results_dir_path=results_dir, data_sampling_freq=fc, dt=1,
                                              channel_names=ecg_leads, line_width=0.3, marker_width=15)

    # Contrastive explainability: Why "myocardial infarction", rather than "normal"?
    fact_classes = [1, 2, 3]
    contrastive_foil_classes = [1, 2, 3]
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=fact_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            data_sampling_freq=fc, dt=1, channel_names=ecg_leads,
                                                            contrastive_foil_classes=contrastive_foil_classes)
    comparison_algorithms = ["Grad-CAM", "HiResCAM"]
    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithms, target_classes=fact_classes,
                                              target_layers=target_layers_names, data_names=data_names,
                                              fig_size=(15, 18), grid_instructions=(4, 3), bar_ranges_dict=bar_ranges,
                                              results_dir_path=results_dir, data_sampling_freq=fc, dt=1,
                                              channel_names=ecg_leads, line_width=0.3, marker_width=15,
                                              contrastive_foil_classes=contrastive_foil_classes)


# Main
#pytorch_model_testing()
print("\n===========================================================================================================\n")
tensorflow_model_testing()
