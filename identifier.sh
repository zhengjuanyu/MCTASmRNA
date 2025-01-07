#!/bin/bash

help() {
    echo "please do as follow:"
    echo "    Usage : identifier.sh transcript.fasta model"
    echo "    Model options: arabidopsis, rice, human, Poplar, Celegans, Fly, Moryzae, Rstol"
    echo "    All the output files located in the path of input fasta file"
    exit 1
}
if [[ $# == 0 || "$1" == "-h" || "$1" == "-help" || "$1" == "--help" ]]; then
    help
    exit 1
fi

if [[ $# != 2 ]]; then
    echo "We need two parameters: 1st for transcript.fasta, 2nd for predicting model"
    exit 1
fi

filename=$1

if [[ ${filename: -2} != "fa" && ${filename: -5} != "fasta" ]]; then
    echo "The first parameter should be a transcript.fasta file"
    exit 1
fi

model=$2

##
case $model in
    "arabidopsis" | "TAIR10")
        export MAX_LENGTH=50
        ;;
    "rice" | "Celegans")
        export MAX_LENGTH=70
        ;;
    "human" | "Fly" | "Moryzae" | "Rstol" | "Poplar")
        export MAX_LENGTH=80
        ;;
    *)
        echo "The second parameter should be a model, choosing from [arabidopsis, TAIR10, rice, human, Poplar, Celegans, Fly, Moryzae, Rstol]"
        exit 1
        ;;
esac

echo "MAX_LENGTH set to $MAX_LENGTH"

echo "Step 1: make blast database"
date
makeblastdb -in $1 -dbtype nucl

echo "Step 2: sequence alignment using blastn"
date
blastn -query $1 -db $1 -strand plus -evalue 1E-10 -outfmt 5 -ungapped -num_threads 20 -out ${filename%.*}_out.xml

echo "Step 3: predict AS transcript pair"
date

#
aa_positions_file="${filename%.*}_aa_positions.txt"
python predictAS.py ${filename%.*}_out.xml $1 ${filename%.*}_as.txt $aa_positions_file >${filename%.*}_as.seq

echo "Step 4: classify AS transcript pair"
date

#
python classifyAS.py ${filename%.*}_as.seq -m $2 -o ${filename%.*}_as_type.txt -pos $aa_positions_file

echo "Step 5: Generate MEME motifs and plot sequence logo with R"
date

meme_output_dir="${filename%.*}_meme_results"

# sequence logo chart
Rscript plot_seqlogo.R ${filename%.*}_as_type.txt $1 $meme_output_dir

echo "done, thank you"
date
