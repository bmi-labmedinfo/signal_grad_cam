'''

The PyTorch model used in this script was obtained from TorchVision:
pytorch.org/vision/stable/models.html#video-classification and the data come from Kinetics-400:
https://www.kaggle.com/datasets/ipythonx/k4testset

Due to the large file size, the data files for the PyTorch model evaluation are not included in this GitHub repository.
However, they can be retrieved directly from the source.
    - The first video file, "--07WQ2iBlw.mp4", is available kaggle.com/datasets/ipythonx/k4testset
      After downloading, rename the file ("javelin_throw.mp4") and place it in the following directory within this
      project: ./data/datasets/video_data/
    - The second video file, "--7VUM9MKg4.mp4", is available at kaggle.com/datasets/ipythonx/k4testset
      After downloading, rename the file ("playing_saxophone.wav") and place it in the following directory within this
      project: ./data/datasets/video_data/

The code for loading and fine-tuning the TensorFlow/Keras model used in this script was obtained from GitHub:
https://github.com/ZFTurbo/classification_models_3D and the data come from OrganMNIST3D: medmnist.com

'''

# Import dependencies
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
import tensorflow as tf
import medmnist

from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from classification_models_3D.kkeras import Classifiers
from medmnist import INFO

from signal_grad_cam import TorchCamBuilder, TfCamBuilder

# Set working directories
working_dir = "./"

# Test functions
def pytorch_model_testing():
    # Set data directory
    data_dir = working_dir + "data/datasets/video_data/"

    # Read data
    data_names = ["javelin_throw", "playing_saxophone"]
    data_list = [read_video(data_dir + data_name + ".mp4", output_format="TCHW", pts_unit="sec")[0].numpy()
                 for data_name in data_names]
    fps_list = [read_video(data_dir + data_name + ".mp4", output_format="TCHW", pts_unit="sec")[2]["video_fps"]
                for data_name in data_names]
    data_labels = [166, 244]
    channel_names = ["red", "green", "blue"]

    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_torch_video_model/"
    if "results_for_torch_video_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Define preprocessing function
    weights = R3D_18_Weights.DEFAULT
    def transform_fn(video):
        preprocess = weights.transforms()
        video = preprocess(torch.tensor(video))
        return video

    # Load the model
    model = r3d_18(weights=weights)
    model.eval()

    # Draw CAMs
    cam_builder = TorchCamBuilder(model, transform_fn=transform_fn, class_names=weights.meta["categories"], time_axs=0)
    target_classes = [166, 244, 357]
    explainer_types = ["Grad-CAM", "HiResCAM"]
    target_layers_names = "layer4.1.conv2.0"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=data_names, results_dir_path=results_dir,
                                                            video_fps_list=fps_list)

    # Overlap CAM on the original video
    target_class = 166
    comparison_algorithm = "Grad-CAM"
    cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels,
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=target_class,
                                          target_layers=target_layers_names, data_names=data_names,
                                          bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                          video_fps_list=fps_list)

    cam_builder.single_channel_output_display(data_list=data_list, data_labels=data_labels,
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_classes,
                                              target_layers=target_layers_names, data_names=data_names,
                                              bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                              channel_names=channel_names, video_fps_list=fps_list, video_channel_idx=1)

    # Contrastive Explanations: Why "javelin_throw", rather than "throwing ball"?
    contrastive_foil_class = 357
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list=[data_list[0]], data_labels=[data_labels[0]],
                                                            target_classes=target_class,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            data_names=[data_names[0]], results_dir_path=results_dir,
                                                            contrastive_foil_classes=contrastive_foil_class,
                                                            video_fps_list=[fps_list[0]])

    cam_builder.overlapped_output_display(data_list=[data_list[0]], data_labels=[data_labels[0]],
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=[target_class],
                                          target_layers=target_layers_names, data_names=[data_names[0]],
                                          bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                          contrastive_foil_classes=contrastive_foil_class, video_fps_list=[fps_list[0]])

    cam_builder.single_channel_output_display(data_list=[data_list[0]], data_labels=[data_labels[0]],
                                              predicted_probs_dict=predicted_probs, cams_dict=cams,
                                              explainer_types=comparison_algorithm, target_classes=target_class,
                                              target_layers=target_layers_names, data_names=[data_names[0]],
                                              bar_ranges_dict=bar_ranges, results_dir_path=results_dir,
                                              channel_names=channel_names, video_fps_list=[fps_list[0]],
                                              video_channel_idx=1, contrastive_foil_classes=contrastive_foil_class)

def tensorflow_model_testing():
    # Load data
    info = INFO["organmnist3d"]
    DataClass = getattr(medmnist, info["python_class"])
    train_ds = DataClass(split="train", download=True)
    class_names = list(info["label"].values())

    data_idx = [0, 2]
    data_list = [train_ds.imgs[i] for i in data_idx]
    data_labels = [train_ds.labels[i][0] for i in data_idx]

    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_tf_video_model/"
    if "results_for_tf_video_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Load pretrained model and attach an un-trained classification layer
    ResNet18, preprocess_input = Classifiers.get("resnet18")
    model = ResNet18(input_shape=(28, 28, 28, 3), weights="imagenet", include_top=False)
    output = model.output
    output = tf.keras.layers.GlobalAveragePooling3D()(output)
    output = tf.keras.layers.Dense(len(class_names))(output)
    output = tf.keras.layers.Softmax(name="output")(output)
    model = tf.keras.Model(model.input, output)

    # Define preprocessing function
    def transform_fn(video):
        video = video.astype("float32") / 255.0
        video = np.tile(video[..., np.newaxis], (1, 1, 1, 3))
        video = preprocess_input(video)
        return video

    # Slightly fine-tune the model on the OrganMNIST3D training set for demonstration purposes
    tf.random.set_seed(0)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
    X_train = train_ds.imgs
    y_train = train_ds.labels
    _ = model.fit(transform_fn(X_train), y_train, epochs=10, batch_size=32)

    # Draw CAMs
    cam_builder = TfCamBuilder(model, transform_fn=transform_fn, class_names=class_names,
                               extend_search=True, time_axs=0)
    target_classes = [3, 4]
    explainer_types = ["Grad-CAM", "HiResCAM"]
    target_layers_names = "stage2_unit2_conv2"
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            results_dir_path=results_dir, show_single_video_frames=True)

    # Overlap CAM on the original video
    target_class = 3
    comparison_algorithm = "Grad-CAM"
    cam_builder.overlapped_output_display(data_list=data_list, data_labels=data_labels,
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=target_class,
                                          target_layers=target_layers_names, bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir, show_single_video_frames=True)

    # Contrastive Explanations: Why "femur-right", rather than "femur-left"?
    contrastive_foil_class = 4
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list=[data_list[0]], data_labels=[data_labels[0]],
                                                            target_classes=target_class, explainer_types=comparison_algorithm,
                                                            target_layers=target_layers_names, softmax_final=True,
                                                            results_dir_path=results_dir,
                                                            contrastive_foil_classes=contrastive_foil_class,
                                                            show_single_video_frames=True)

    cam_builder.overlapped_output_display(data_list=[data_list[0]], data_labels=[data_labels[0]],
                                          predicted_probs_dict=predicted_probs, cams_dict=cams,
                                          explainer_types=comparison_algorithm, target_classes=[target_class],
                                          target_layers=target_layers_names, bar_ranges_dict=bar_ranges,
                                          results_dir_path=results_dir, contrastive_foil_classes=contrastive_foil_class,
                                          show_single_video_frames=True)


# Main
pytorch_model_testing()
print("\n===========================================================================================================\n")
tensorflow_model_testing()
