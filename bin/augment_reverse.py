import os
import pandas as pd
import random

def reverse_complement(s, complement=None):
    if complement is None:
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    result = [complement[i] for i in list(s.upper())]
    result = reversed(result)
    return ''.join(result)

def augment_data(sequences):
    augmented_sequences = []
    for seq in sequences:
        as_type, pre_splice_seq, as_seq, post_splice_seq = seq
        augmented_sequences.append(seq)
        if as_type in ['SE']:  
            pre_splice_seq, post_splice_seq = post_splice_seq, pre_splice_seq
            augmented_sequences.append([
                as_type,
                reverse_complement(pre_splice_seq),
                reverse_complement(as_seq),
                reverse_complement(post_splice_seq)
            ])
    return augmented_sequences

def process_file(input_path, output_dir):
    df = pd.read_csv(input_path)

    sequences = df[['as_type', 'pre_splice_seq', 'as_seq', 'post_splice_seq']].values.tolist()
    augmented_sequences = sequences 
    augmented_df = pd.DataFrame(augmented_sequences, columns=['as_type', 'pre_splice_seq', 'as_seq', 'post_splice_seq'])

    result_df = pd.concat([df, augmented_df]).drop_duplicates().reset_index(drop=True)

    result_df = result_df.sample(frac=1).reset_index(drop=True)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    result_df.to_csv(output_path, index=False)


def main():
    base_dir = '/path/input_data_dir/'
    input_files = [os.path.join(base_dir, f) for f in
                   ['training_100.csv']]
    output_dir = '/path/output_dir 


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in input_files:
        process_file(input_file, output_dir)

if __name__ == "__main__":
    main()

