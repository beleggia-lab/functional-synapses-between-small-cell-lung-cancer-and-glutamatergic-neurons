# Functional synapses between small cell lung cancer and glutamatergic neurons

This repository contains the scripts used in the manuscript, "Functional synapses between small cell lung cancer and glutamatergic neurons". It includes two analysis pipelines:

1.  **Orthology mapping pipeline**: For generating orthology mappings between human and mouse genes.
2.  **PiggyBac transposon insertion analysis pipeline**: For analyzing sequencing data derived from piggyBac transposon insertion experiments.

The `config.yml` file must be present in the same directory as the scripts. Please use the `conda_environment.yml` file to create a new environment and install the dependencies.


---

## 1. Reference files

Both pipelines require access to reference files. Please ensure these are downloaded and specified in the `config.yml` file:


* **Mouse reference genome**: the ensembl `GRCm38 primary assembly` was used in the manuscript.
* **Human gene annotation GTF**: the GENCODE v22 annotation was used in the manuscript.
* **Mouse gene annotation GTF**: the GENCODE vM23 annotation was used in the manuscript.
* **Orthology database**: the HCOP orthology database, downloaded on 2020-01-06, was used in the manuscript.


---

## 2. Data for piggyBac insertion analysis

The piggyBac analysis pipeline additionally requires BAM files, which are available from the NCBI Sequence Read Archive (SRA) project `PRJNA1268719`.


---

## 3. Usage

Download the reference files from gencode, ensembl and hcop.
Download the .bam files for the piggyBac data.

Modify the `config.yml` file to specify the correct paths to the reference files and the .bam files.

Create a new conda environment and install the dependencies:
```bash
conda env create -f conda_environment.yml -n piggybac_env
conda activate piggybac_env 
```

Clone the repository:
```bash
git clone https://github.com/beleggia-lab/functional-synapses-between-small-cell-lung-cancer-and-glutamatergic-neurons.git
```


Run the scripts:
```bash
python orthology_analysis.py
```

```bash
python piggybac_analysis.py
```


