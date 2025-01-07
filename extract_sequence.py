import csv
import os
import subprocess
from sklearn.model_selection import train_test_split
import random

def read_as_files(file_path):
    data = []
    with open(file_path, 'r') as file:
        header = file.readline().strip().replace('"', '').split()
        chr_index = header.index('chr')
        as_type_index = header.index('as_type')
        as_start_index = header.index('as_start')
        as_end_index = header.index('as_end')
        splice_start_index = header.index('splice_start')
        splice_end_index = header.index('splice_end')
        splice_start_exon_start_index = header.index('splice_start_exon_start')
        splice_start_exon_end_index = header.index('splice_start_exon_end')
        splice_end_exon_start_index = header.index('splice_end_exon_start')
        splice_end_exon_end_index = header.index('splice_end_exon_end')

        for line in file:
            parts = line.strip().replace('"', '').split()
            data.append({
                'chr': parts[chr_index],
                'as_type': parts[as_type_index],
                'as_start': int(parts[as_start_index]),
                'as_end': int(parts[as_end_index]),
                'splice_start': int(parts[splice_start_index]),
                'splice_end': int(parts[splice_end_index]),
                'splice_start_exon_start': int(parts[splice_start_exon_start_index]),
                'splice_start_exon_end': int(parts[splice_start_exon_end_index]),
                'splice_end_exon_start': int(parts[splice_end_exon_start_index]),
                'splice_end_exon_end': int(parts[splice_end_exon_end_index])
            })
    return data

def fetch_sequence(genome_file, chr, start, end):
    output_file = 'extracted_sequence.fasta'
    command = f"samtools faidx {genome_file} {chr}:{start}-{end} > {output_file}"
    subprocess.run(command, shell=True, check=True)

    with open(output_file, 'r') as file:
        sequence = ''.join(file.readlines()[1:]).replace('\n', '')

    os.remove(output_file)  
    return sequence

def extract_sequences(as_data, genome_file, length=100):
    extracted_sequences = []
    for entry in as_data:
        chr = entry['chr']
        as_type = entry['as_type']
        as_start = entry['as_start']
        as_end = entry['as_end']
        splice_start = entry['splice_start']
        splice_end = entry['splice_end']
        splice_start_exon_start = entry['splice_start_exon_start']
        splice_start_exon_end = entry['splice_start_exon_end']
        splice_end_exon_start = entry['splice_end_exon_start']
        splice_end_exon_end = entry['splice_end_exon_end']

        pre_splice_len = min(length, splice_start_exon_end - splice_start_exon_start + 1)
        pre_splice_start = splice_start_exon_end - pre_splice_len + 1
        # splice_start-1 = splice_start_exon_end
        pre_splice_seq = fetch_sequence(genome_file, chr, pre_splice_start, splice_start_exon_end)

        as_seq = fetch_sequence(genome_file, chr, as_start, as_end)

        post_splice_len = min(length, splice_end_exon_end - splice_end_exon_start + 1)
        post_splice_end = splice_end_exon_start + post_splice_len - 1
        # splice_end+1 = splice_end_exon_start
        post_splice_seq = fetch_sequence(genome_file, chr, splice_end_exon_start, post_splice_end)

        result = [as_type, pre_splice_seq, as_seq, post_splice_seq]
        extracted_sequences.append(result)
    return extracted_sequences


def save_sequences_to_csv(sequences, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['as_type', 'pre_splice_seq', 'as_seq', 'post_splice_seq'])
        writer.writerows(sequences)

def split_and_save_data(sequences, output_dir):

    grouped_data = {}
    for seq in sequences:
        as_type = seq[0]
        if as_type not in grouped_data:
            grouped_data[as_type] = []
        grouped_data[as_type].append(seq)


    all_sequences = []
    for group in grouped_data.values():
        all_sequences.extend(group)

    save_sequences_to_csv(all_sequences, os.path.join(output_dir, 'alldata_100.csv'))


    random.shuffle(all_sequences)

    # 8:1:1
    train, temp = train_test_split(all_sequences, test_size=0.2, random_state=42)  #
    val, test = train_test_split(temp, test_size=0.5, random_state=42)  # 

    # save
    save_sequences_to_csv(train, os.path.join(output_dir, 'training_100.csv'))
    save_sequences_to_csv(val, os.path.join(output_dir, 'validation_100.csv'))
    save_sequences_to_csv(test, os.path.join(output_dir, 'testing_100.csv'))

def process_files(as_files, genome_file, output_dir):
    all_sequences = []
    for file in as_files:
        data = read_as_files(file)
        extracted_sequences = extract_sequences(data, genome_file)
        all_sequences.extend(extracted_sequences)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    split_and_save_data(all_sequences, output_dir)
    print(f"Extracted sequences have been saved to {output_dir}")

def main():

    base_dir = '/path/preprocess_result'
    result_dir = '/path/data'

    as_files = [os.path.join(base_dir, f) for f in ['A3.txt', 'A5.txt', 'SE.txt', 'RI.txt']]

    genome_base_dir = '/path/genome'

    genome_file = os.path.join(genome_base_dir, 'Caenorhabditis_elegans.WBcel235.dna.toplevel.fa')

    output_dir = os.path.join(result_dir, 'genome_len_100')


    if os.path.exists(genome_file):
        print("file exist")
    else:
        print("file doesn't exist")

    process_files(as_files, genome_file, output_dir)

if __name__ == "__main__":
    main()
