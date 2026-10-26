"""

The PyTorch model used in this script was obtained from HuggingFace Hub and was used as a backbone for a toy binary
classifier and for next nucleotide (token) prediction:
https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf
while the DNA datasets were generated synthetically for demonstration purposes.

"""

# Import dependencies
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from signal_grad_cam import TorchCamBuilder

# Set working directories
working_dir = "./"

# Define global variables
checkpoint = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
bases = ["A", "C", "G", "T"]


# Test functions
def pytorch_model_testing_next_token_prediction():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_torch_text_model_next_token_prediction/"
    if "results_for_torch_text_model_next_token_prediction" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    model.eval()

    # Define textual input
    text = ["ACGTNACGTACGTNACGTACGTNACGTACGTNACGTACGTAACGTACGTNACGTACGTNACGTACGTNACGTACGTNACGTACGTNACGTACGTNACGTACGTNAC"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    data_list = [input_ids]
    classes = ["[CLS]", "[SEP]", "[BOS]", "[MASK]", "[PAD]", "[RESERVED]", "[UNK]"] + bases + ["N"]
    data_labels = [7]

    # Draw CAMs
    cam_builder = TorchCamBuilder(model, class_names=classes, is_transformer_network=True)
    target_classes = [7]
    explainer_types = ["Grad-CAM"]
    target_layers_names = ["hyena.backbone.layers.1.mixer.short_filter"]
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(data_list, data_labels=data_labels,
                                                            target_classes=target_classes,
                                                            explainer_types=explainer_types,
                                                            target_layers=target_layers_names, softmax_final=False,
                                                            results_dir_path=results_dir,
                                                            extra_inputs_dict={"output_hidden_states": True},
                                                            aspect_factor=50)

    # Overlap CAM on the original video
    target_class = 7
    comparison_algorithm = "Grad-CAM"
    token_data_list = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    cam_builder.text_output_display(token_data_list=token_data_list, data_labels=data_labels,
                                    predicted_probs_dict=predicted_probs, cams_dict=cams,
                                    explainer_types=comparison_algorithm, target_classes=target_class,
                                    target_layers=target_layers_names, bar_ranges_dict=bar_ranges,
                                    results_dir_path=results_dir)


def pytorch_model_testing():
    # Define and create results directory
    results_dir = working_dir + "outputs/"
    if "outputs" not in os.listdir(working_dir):
        os.mkdir(results_dir)
    results_dir += "results_for_torch_text_model/"
    if "results_for_torch_text_model" not in os.listdir(working_dir + "outputs/"):
        os.mkdir(results_dir)

    # Load pretrained model (Hyena) and attach an un-trained classification layer
    class HyenaClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
            hidden_size = self.backbone.config.d_model
            self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                     nn.Linear(hidden_size // 2, num_classes))

        def forward(self, input_ids):
            if input_ids.dim() == 3 and input_ids.size(1) == 1:
                input_ids = input_ids.squeeze(1)
            output = self.backbone(input_ids)
            seq_output = output.last_hidden_state
            pooled = seq_output.mean(dim=1)
            return self.mlp(pooled)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = HyenaClassifier(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate DNA synthetic training set
    class DNADataset(Dataset):
        def __init__(self, sequences, labels):
            tokens = tokenizer(sequences, padding="max_length", truncation=True, max_length=seq_len,
                               return_tensors="pt")
            self.input_ids = tokens["input_ids"]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.labels[idx]

    motif = "GCGCGCGCGCGCGCGC"
    seq_len = 300
    sequence_list, label_list = [], []
    for i in range(600):
        seq = list("".join(random.choices(bases, k=seq_len)))
        if i % 2 == 0:
            label_list.append(0)
        else:
            motif_position = random.randint(0, seq_len - len(motif))
            seq[motif_position:motif_position + len(motif)] = list(motif)
            label_list.append(1)
        sequence_list.append("".join(seq))
    train_loader = DataLoader(DNADataset(sequence_list, label_list), batch_size=16, shuffle=True)

    # Slightly fine-tune the model on the synthetic DNA dataset for demonstration purposes
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30
    model.train()
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        total_loss, correct, total = 0, 0, 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc*100:.1f}%")
    model.eval()
    model.to("cpu")

    # Generate DNA synthetic test dataset
    seq_class0 = "".join(random.choices(bases, k=seq_len))
    seq_class1_list = list(seq_class0)
    seq_class1_list[0:len(motif)] = list(motif)
    seq_class1 = "".join(seq_class1_list)
    eval_sequences = [seq_class0, seq_class1]
    eval_labels = [0, 1]
    eval_tokens = tokenizer(eval_sequences, padding="max_length", truncation=True, max_length=seq_len,
                            return_tensors="np")
    eval_data_list = [eval_tokens["input_ids"][i:i+1] for i in range(len(eval_sequences))]

    # Draw CAMs
    def preprocess_fn(data):
        return torch.tensor(data, dtype=torch.long, device=next(model.parameters()).device)
    cam_builder = TorchCamBuilder(model, transform_fn=preprocess_fn, is_transformer_network=True,
                                  class_names=["CG-pattern absent", "CG-pattern present"])

    target_classes = [1]
    explainer_types = ["Grad-CAM"]
    target_layers = ["backbone.backbone.layers.0.mixer.short_filter"]
    cam_results, predicted_probs, score_ranges = cam_builder.get_cam(data_list=eval_data_list,
                                                                     data_labels=eval_labels,
                                                                     target_classes=target_classes,
                                                                     explainer_types=explainer_types,
                                                                     target_layers=target_layers,
                                                                     softmax_final=False,
                                                                     axes_names=("Position (nt)", "Channel"),
                                                                     results_dir_path=results_dir)

    token_data_list = [[tokenizer.convert_ids_to_tokens(ids) for ids in eval_datum] for eval_datum in eval_data_list]
    cam_builder.text_output_display(token_data_list=token_data_list, data_labels=eval_labels,
                                    predicted_probs_dict=predicted_probs, cams_dict=cam_results,
                                    explainer_types=explainer_types, target_classes=target_classes,
                                    target_layers=target_layers, bar_ranges_dict=score_ranges,
                                    results_dir_path=results_dir)


# Main
pytorch_model_testing_next_token_prediction()
print("\n===========================================================================================================\n")
pytorch_model_testing()
