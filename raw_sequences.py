'''
Convert aligned fasta file (afa) and into raw sequence file (fasta)

Usage: python raw_sequences.py input.afa output.fasta
       python raw_sequences.py input_directory output_directory
'''

import argparse
from pathlib import Path
from Bio import SeqIO

VALID_EXTENSIONS = {'.afa', '.fa', '.fasta', '.aln'}


# Essentially just remove all gaps from the sequences while keeping headers the same
def convert_file(input_path: Path, output_path: Path) -> None:
    with input_path.open('r') as input_handle, output_path.open('w') as output_handle:
        for record in SeqIO.parse(input_handle, 'fasta'):
            raw_seq = str(record.seq).replace('-', '')
            output_handle.write(f'>{record.id}\n{raw_seq}\n')


# Parse arguments and pass into convert_file function
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert aligned fasta file(s) (afa) into raw sequence fasta file(s)'
    )
    parser.add_argument('input_afa', help='Input aligned fasta file or directory')
    parser.add_argument('output_fasta', help='Output raw sequence fasta file or directory')
    args = parser.parse_args()

    input_path = Path(args.input_afa)
    output_path = Path(args.output_fasta)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = [p for p in sorted(input_path.rglob('*')) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if not files:
            raise SystemExit(f'No aligned fasta files found in directory: {input_path}')

        for file_path in files:
            target_path = output_path / file_path.relative_to(input_path)
            target_path = target_path.with_suffix('.fasta')
            target_path.parent.mkdir(parents=True, exist_ok=True)
            convert_file(file_path, target_path)
            print(f'Converted: {file_path} -> {target_path}')
    else:
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / f'{input_path.stem}.fasta'
        convert_file(input_path, output_path)
        print(f'Converted: {input_path} -> {output_path}')


if __name__ == '__main__':
    main()
