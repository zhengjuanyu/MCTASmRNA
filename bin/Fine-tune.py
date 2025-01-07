import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_file, max_len=50):
    def filter_lines(lines, labels_list):
        new_lines = []
        for line in lines:
            if line[:2] in labels_list.keys():
                new_lines.append(line)
        return new_lines

    def label_trans(array, array_dict):
        new_array = []
        for i in array:
            new_array.append(array_dict[i])
        return new_array

    def to_one_hot_coding(string):
        for i in string:
            if i == 'A' or i == 'a':
                yield [1, 0, 0, 0, 0]
            elif i == "T" or i == 't':
                yield [0, 1, 0, 0, 0]
            elif i == "C" or i == "c":
                yield [0, 0, 1, 0, 0]
            elif i == "G" or i == "g":
                yield [0, 0, 0, 1, 0]
            elif i == ",":
                yield [0, 0, 0, 0, 1]
            else:
                yield [0, 0, 0, 0, 0]

    labels_list = {'SE': 0, 'A3': 1, 'A5': 2, 'RI': 3}

    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
    except IOError:
        raise ("Can not find file{}".format(data_file))

    lines = filter_lines(lines, labels_list)
    tmp_labels = []
    tmp_data = []

    for line in lines:
        tmp = line.strip('\n').split(',')
        tmp_labels.append(tmp[0])
        if len(tmp[2]) >= max_len:
            tmp_start = tmp[2][:max_len]
            tmp_end = tmp[2][-max_len:]
        else:
            tmp_start = tmp[2] + "0" * (max_len - len(tmp[2]))
            tmp_end = "0" * (max_len - len(tmp[2])) + tmp[2]
        if len(tmp[1]) > max_len:
            tmp[1] = tmp[1][-max_len:]
        elif len(tmp[1]) < max_len:
            tmp[1] = "0" * (max_len - len(tmp[1])) + tmp[1]
        if len(tmp[3]) > max_len:
            tmp[3] = tmp[3][:max_len]
        elif len(tmp[3]) < max_len:
            tmp[3] = tmp[3] + "0" * (max_len - len(tmp[3]))

        tmp_data.append(','.join((tmp[1], tmp_start)) + ','.join((tmp_end, tmp[3])))

    tmp_labels = label_trans(tmp_labels, labels_list)
    tmp_labels = np.array(tmp_labels).astype('int64')
    data0 = []
    for i in tmp_data:
        tmp = []
        for j in to_one_hot_coding(i):
            tmp.append(j)
        data0.append(tmp)
    tmp_data = np.array(data0).astype("float32")
    return torch.tensor(tmp_data), torch.tensor(tmp_labels)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiScaleConv, self).__init__()
        self.convs = nn.ModuleList()
        self.paddings = []
        for k in kernel_sizes:
            padding = (k - 1) // 2  # symmetric padding for odd kernels
            if k % 2 == 0:
                self.paddings.append((padding, padding + 1))
            else:
                self.paddings.append((padding, padding))
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k),
                nn.ReLU(),
                ECAAttention(out_channels)
            ))

    def forward(self, x):
        features = []
        for conv, padding in zip(self.convs, self.paddings):
            padded_x = F.pad(x, padding)  # Apply manual padding
            conv_output = conv(padded_x)
             features.append(conv_output)
        features = torch.cat(features, dim=1)
        return features

class RoPEEncoding(nn.Module):
    def __init__(self, d_model):
        super(RoPEEncoding, self).__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, "d_model must be even for RoPE encoding"

    def forward(self, x):
        seq_len, batch_size, d_model = x.size()
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1).to(x.device)  # (seq_len, 1)

        # Compute the angular frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model)).to(
            x.device)  # (d_model//2)

        # Compute sine and cosine components
        sin_part = torch.sin(position * div_term)  # (seq_len, d_model//2)
        cos_part = torch.cos(position * div_term)  # (seq_len, d_model//2)

        # Expand dimensions to match the input
        sin_part = sin_part.unsqueeze(1)  # (seq_len, 1, d_model//2)
        cos_part = cos_part.unsqueeze(1)  # (seq_len, 1, d_model//2)

        # Apply RoPE by rotating the input embeddings
        x1 = x[..., 0::2] * cos_part - x[..., 1::2] * sin_part
        x2 = x[..., 0::2] * sin_part + x[..., 1::2] * cos_part

        return torch.cat([x1, x2], dim=-1)

class PerceptionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(PerceptionAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        # Perception module: a small feed-forward network to process attention output
        self.perception_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Optional: final linear layer to combine the perception output with the original attention output
        self.final_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)

        # Apply the perception module to the attention output
        perception_output = self.perception_layer(attn_output)

        # Combine the perception output with the original attention output
        combined_output = self.final_layer(perception_output + attn_output)

        return combined_output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.5):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = PerceptionAttention(embed_dim, num_heads)  # Use the new PerceptionAttention here
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Perception-based Multi-head attention with residual connection
        x_norm = self.layernorm1(x)
        attn_output = self.attention(x_norm)
        # attn_output = self.dropout1(attn_output)
        out1 = x + attn_output
        # out1 = self.layernorm1(x + attn_output)  # Residual connection and LayerNorm

        # Feed-forward network with residual connection
        out1_norm =  self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        # ffn_output = self.dropout2(ffn_output)
        # out2 = self.layernorm2(out1 + ffn_output)  # Residual connection and LayerNorm
        out2 = out1 + ffn_output

        return out2


class ECAAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CNN_transformer_Model(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_heads=8, ff_dim=512, num_layers=2, max_len=202):
        super(CNN_transformer_Model, self).__init__()
        self.multi_scale_conv1 = MultiScaleConv(in_channels=input_dim, out_channels=32, kernel_sizes=[3,6,9])
        self.dropout1 = nn.Dropout(0.5)
        self.conv = nn.Conv1d(32*3, embed_dim, kernel_size=1)
        self.relu = nn.ReLU()

        self.pos_encoder = RoPEEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.fc_feature = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, extract_features=False):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        x = self.multi_scale_conv1(x)
        x = self.dropout1(x)
        x = self.conv(x)
        x = self.relu(x)

        x = x.permute(2, 0, 1)  # (seq_length, batch_size, embed_dim)
        x = self.pos_encoder(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.permute(1, 0, 2)  # (batch_size, seq_length, embed_dim)
        x = x.mean(dim=1)

        if extract_features:
            return self.fc_feature(x)  
        else:
            return self.fc(x)  


def train_model(model, train_data, train_labels, val_data, val_labels, model_weights_path, center_loss_weight=0.01, epochs=200, batch_size=128):
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_classes = len(torch.unique(train_labels))
    focal_loss = FocalLossDynamicWeight(gamma=1.0)
    center_loss_criterion = CenterLossDynamicWeight(num_classes=num_classes, feat_dim=model.fc_feature.out_features, device=device, base_weight=center_loss_weight)
    criterion = CombinedLoss(focal_loss=focal_loss, center_loss=center_loss_criterion, ce_weight=0.75, focal_weight=0.1,
                             center_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    early_stopping_patience = 20
    best_val_loss = float('inf')
    patience_counter = 0

    loss_values = []
    accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_center_loss = 0.0
        correct = 0
        total = 0

        all_features = []
        all_labels = []

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            features = model(inputs, extract_features=True)

            loss = criterion(inputs=outputs, targets=labels, features=features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_features.append(features.detach())
            all_labels.append(labels.detach())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        confusion_matrix = calculate_confusion_matrix(model, train_loader, num_classes)
        focal_loss.update_weights(confusion_matrix)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        center_loss_criterion.update_weights(all_features, all_labels)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                features = model(inputs, extract_features=True)
                loss = criterion(inputs=outputs, targets=labels, features=features)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        loss_values.append(train_loss)
        accuracy_values.append(train_accuracy)
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)

        print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch + 1))
                break

    return loss_values, accuracy_values, val_loss_values, val_accuracy_values

def main():
    if torch.cuda.is_available():
        device_cuda = torch.device('cuda')
        print(f'Using device: {torch.cuda.get_device_name(device_cuda)}')
    else:
        print('Using CPU')
    train_data_file = '/path/data/Celegans_training_100.csv'
    val_data_file = '/path/data/Celegans_validation_100.csv'
    output_dir = '/path/model/Ara_based/Celegans'
    human_pth= '/path/bin/best_parameters/arabidopsis.pth'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_weights_path = os.path.join(output_dir, 'Celegans.pth')

    train_data, train_labels = load_data(train_data_file)
    val_data, val_labels = load_data(val_data_file)

    input_dim = train_data.shape[2]
    num_classes = len(np.unique(train_labels))
    model = CNN_transformer_Model(input_dim=input_dim, num_classes=num_classes)
    checkpoint = torch.load(human_pth)
    model.load_state_dict(checkpoint)
    fine_tune_at = 4
    layers = list(model.children())
    for layer in layers[:fine_tune_at]:
        for param in layer.parameters():
            param.requires_grad = False  

    loss_values, accuracy_values, val_loss_values, val_accuracy_values = train_model( model,
                                                                                      train_data, train_labels, val_data, val_labels, model_weights_path, epochs=20, batch_size=128)

if __name__ == "__main__":
    main()



