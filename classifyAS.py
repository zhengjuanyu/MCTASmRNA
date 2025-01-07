import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import argparse
import torch.nn.functional as F

max_length = int(os.getenv('MAX_LENGTH', 80))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#preprocess
def load_data(data_file, max_len=max_length):
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
    reverse_labels_list = {v: k for k, v in labels_list.items()}

    with open(data_file, 'r') as f:
        lines = f.readlines()

    tmp_id = []
    tmp_data = []

    for line in lines:
        tmp = line.strip('\n').split(',')
        tmp_id.append(tmp[0])
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

    data0 = []
    for i in tmp_data:
        tmp = []
        for j in to_one_hot_coding(i):
            tmp.append(j)
        data0.append(tmp)

    tmp_data = np.array(data0).astype(np.float32)

    return torch.tensor(tmp_data, dtype=torch.float32), tmp_id, reverse_labels_list


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

    def forward(self, x):
        # Perception-based Multi-head attention with residual connection
        x_norm = self.layernorm1(x)
        attn_output = self.attention(x_norm)
        out1 = x + attn_output
        out1_norm =  self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        out2 = out1 + ffn_output

        return out2

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

class CNN_transformer_Model(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_heads=8, ff_dim=512, num_layers=2):
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


def load_position_file(position_file):
    with open(position_file, 'r') as f:
        position_list = [line.strip().split(',')[1] for line in f]
    return position_list

def test_model(model, test_data, tmp_id, reverse_labels_list, model_weights_path, output_file, position_list):
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)

    with open(output_file, 'w') as fw:
        for i in range(len(tmp_id)):
            fw.write(tmp_id[i])
            fw.write("\t")
            j = np.argmax(all_preds[i])
            fw.write(reverse_labels_list[j])
            fw.write("\t")
            fw.write(str(all_preds[i][j]))
            fw.write("\t")

            if i < len(position_list):
                position_info = position_list[i]
            else:
                position_info = "Position Info"
            fw.write(position_info)
            fw.write("\n")

def main():
    #
    parser = argparse.ArgumentParser("Arguments for use CNN_transformer model predict Alternative Splice event")
    parser.add_argument(dest='input',
                        help='input sequence of AS event, a csv file contains 4 columns, the 1st columns for seq id, 2nd for upstream seq, 3rd for alternative seq, 4th for downstream seq. 2nd and 4th should in length 50',
                        type=argparse.FileType('r'))
    parser.add_argument('-m', help='model for specie, choose from [arabidopsis, human, rice, Poplar,Celegans, Fly, Moryzae, Rstol]',
                        default="human", choices=['arabidopsis', 'human', 'rice','TAIR10','Poplar','Celegans', 'Fly', 'Moryzae', 'Rstol'])
    parser.add_argument('-o', help='output file name', default="result.txt")
    parser.add_argument('-pos', help='position file name', required=True)
    args = parser.parse_args()

    position_list = load_position_file(args.pos)

    test_data, tmp_id, reverse_labels_list = load_data(args.input.name)


    input_dim = test_data.shape[2]
    num_classes = 4  # SE, A3, A5, RI
    model = CNN_transformer_Model(input_dim=input_dim, num_classes=num_classes)


    model_weights_path = f"./best_parameters/{args.m}.pth"
    test_model(model, test_data, tmp_id, reverse_labels_list, model_weights_path, args.o, position_list)

if __name__ == "__main__":
    main()

