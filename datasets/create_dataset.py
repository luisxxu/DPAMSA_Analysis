import random
import os

#This script is useful to create a dataset of synthetic sequences(for training and for inference) 

# Generate num_sequence sequences and create a dataset with also fasta_file (useful to do comparative test)
num_sequences = 6
sequence_length = 60
mutation_rate = 0.10  # Mutation rate 20%
number_of_dataset = 50
DATASET_NAME = 'training_dataset1_6x30bp'

FILE_NAME_SCRIPT_OUTPUT = f'./datasets/{DATASET_NAME}.py'
FASTA_OUTPUT = f'./datasets/fasta_files/{DATASET_NAME}'

if not os.path.exists(FASTA_OUTPUT):
    os.makedirs(FASTA_OUTPUT)


def generate_random_dna_sequence(BP):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(BP))

def mutate_sequence(sequence, mutation_rate):
    nucleotides = ['A', 'T', 'C', 'G']
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(nucleotides)
    return ''.join(mutated_sequence)

def write_fasta_file(filename, sequences):
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            header = f">Sequence_{i+1}\n"  # Header for each sequence
            f.write(header)
            f.write(seq + "\n")

def write_dataset_dpamsa(fasta_files, output_file):
    sequences = {}
    for file in fasta_files:
        sequences[file] = {}
        with open(file, 'r') as f:
            lines = f.readlines()
            seq = ''
            seq_name = ''
            for line in lines:
                if line.startswith('>'):
                    if seq_name != '':
                        sequences[file][seq_name] = seq
                    seq_name = line.strip().lstrip('>')
                    seq = ''
                else:
                    seq += line.strip()
            if seq_name != '':
                sequences[file][seq_name] = seq
    fasta_file_names = []
    for file in fasta_files:
        filename = os.path.basename(file).split('.')[0]
        fasta_file_names.append(filename)

    file_content = f"""
file_name = '{os.path.basename(output_file)}'

datasets = {fasta_file_names}
"""
    for i, filename in enumerate(fasta_files):  # Crea dataset0, dataset1, ... fino a quanto necessario
        dataset_name = f"dataset{i}"
        if filename in sequences:
            dataset_sequences = sequences[filename]
            file_content += f"\n{fasta_file_names[i]} = "
            sequence_list = list(dataset_sequences.values())
            file_content += f'{sequence_list}\n'

    # Scrivi il contenuto nel file Python
    with open(output_file, 'w') as f:
        f.write(file_content)
        
fasta_files = []
for dataset in range(number_of_dataset):
    sequences = []
    for _ in range(num_sequences):
        sequence = generate_random_dna_sequence(sequence_length)
        mutated_sequence = mutate_sequence(sequence, mutation_rate)
        sequences.append(mutated_sequence)

    fasta_filename = f'test{dataset}'+ '.fasta'
    fasta_filename = os.path.join(FASTA_OUTPUT,fasta_filename)
    write_fasta_file(fasta_filename, sequences)

    fasta_files.append(fasta_filename)
    print(f"Fasta file created in {fasta_filename}")

write_dataset_dpamsa(fasta_files,FILE_NAME_SCRIPT_OUTPUT)
print(f"Dataset file created in {FILE_NAME_SCRIPT_OUTPUT}")