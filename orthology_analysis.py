#!/usr/bin/env python3
"""
Mouse-Human Orthologous Gene Identification

This script identifies orthologous genes between mouse and human using GENCODE annotations and HCOP orthology data. It processes both protein-coding and all gene sets to create orthology mappings. 

The script as been tested only with the conda environment defined in conda_environment.yml file.

"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import yaml
from typing import Union, List


def calculate_database_support(group: pd.Series) -> int:
    """
    Calculate the number of unique databases supporting an orthology relationship.
    
    Args:
        group: Pandas series containing comma-separated database names
        
    Returns:
        int: Number of unique supporting databases
    """
    total = set()
    for element in group:
        total.update(element.split(','))
    return len(total)


def select_best_ortholog(
        grouped_data: tuple[pd.Index,
                            pd.DataFrame]) -> list[Union[float, int]]:
    """
    Select the best ortholog based on database support.
    
    For cases with multiple potential orthologs, selects the one with highest
    database support. Returns NaN values if there's a tie for best support.
    
    Args:
        grouped_data: Tuple of (index, group) from pandas groupby
        
    Returns:
        list: [index, ortholog_id, database_support] or [NaN, NaN, NaN]
    """
    index, group = grouped_data
    best_support = group.database_support.max()
    has_unique_best = (group.database_support == best_support).sum() == 1

    if has_unique_best:
        best_row = group[group.database_support == best_support].iloc[0]
        return [index] + list(best_row.values)
    else:
        return [np.nan, np.nan, np.nan]


def parse_gencode_annotations(
        file_path: str,
        species: str) -> tuple[pd.Index, pd.Index, pd.DataFrame]:
    """
    Parse GENCODE GTF file and extract gene information.
    
    Args:
        file_path (str): Path to GENCODE GTF file
        species (str): Species name ('human' or 'mouse')
        
    Returns:
        tuple: (geneset, protein_coding_set, ensemble_to_symbol_dict)
    """
    print(f"Processing {species} GENCODE annotations...", flush=True)

    # Read and filter GTF file for exon entries
    gencode_data = pd.read_csv(file_path,
                               sep='\t',
                               index_col=2,
                               comment='#',
                               header=None).loc['transcript']

    # Extract relevant columns and filter out mitochondrial genes
    filtered_data = gencode_data[gencode_data.iloc[:, 0] !=
                                 'chrM'].loc[:, [0, 3, 4, 6, 8]].rename(
                                     columns={
                                         0: 'chromosome',
                                         3: 'start',
                                         4: 'end',
                                         6: 'strand',
                                         8: 'information'
                                     })

    # Parse gene information from GTF attributes
    print(f"{species}: Parsing gene attributes...", flush=True)
    gencode_strings = filtered_data.information.str.replace(' ',
                                                            '').str.split(';')

    # Create attribute index mapping
    info_index = {
        element.split('"')[0]: idx
        for idx, element in enumerate(gencode_strings.iloc[0])
    }

    # Extract gene attributes
    filtered_data.loc[:, 'protein_coding_transcript'] = (
        gencode_strings.str[info_index['transcript_type']].str[:-1].str.split(
            '"').str[1] == 'protein_coding')
    filtered_data.loc[:, 'transcript_id'] = (
        gencode_strings.str[info_index['transcript_id']].str.split(
            '"').str[1].str.split('.').str[0])
    filtered_data.loc[:, 'gene_id'] = (gencode_strings.str[
        info_index['gene_id']].str.split('"').str[1].str.split('.').str[0])
    filtered_data.loc[:, 'gene_symbol'] = (gencode_strings.str[
        info_index['gene_name']].str[:-1].str.split('"').str[1])

    print(f"{species}: Creating gene sets...", flush=True)

    # Create gene sets and mappings
    geneset = pd.Index(filtered_data['gene_id'].unique())
    protein_coding_set = pd.Index(filtered_data[
        filtered_data.protein_coding_transcript]['gene_id'].unique())
    ensemble_to_symbol = (filtered_data[[
        'gene_id', 'gene_symbol'
    ]].drop_duplicates().set_index('gene_id'))

    return geneset, protein_coding_set, ensemble_to_symbol


def process_orthology_data(hcop_file_path: str,
                           human_geneset: pd.Index,
                           mouse_geneset: pd.Index,
                           human_symbols: pd.DataFrame,
                           mouse_symbols: pd.DataFrame,
                           output_dir: str,
                           orthology_label: str,
                           gene_type: str = "all") -> pd.DataFrame:
    """
    Process HCOP orthology data and create orthology mappings.
    
    Args:
        hcop_file_path (str): Path to HCOP orthology file
        human_geneset (pd.Index): Set of human gene IDs
        mouse_geneset (pd.Index): Set of mouse gene IDs
        human_symbols (pd.DataFrame): Human gene ID to symbol mapping
        mouse_symbols (pd.DataFrame): Mouse gene ID to symbol mapping
        output_dir (str): Output directory path
        orthology_label (str): Label for output files
        gene_type (str): "all" or "protein_coding"
    """
    print(f"Processing {gene_type} genes orthology...", flush=True)

    # Read and preprocess orthology data
    raw_orthology = pd.read_csv(hcop_file_path, sep='\t')[[
        'human_ensembl_gene', 'mouse_ensembl_gene', 'support'
    ]]
    raw_orthology.columns = [
        'human_ensembl_gene', 'mouse_ensembl_gene', 'database_support'
    ]

    # Filter out missing entries
    raw_orthology = raw_orthology[(raw_orthology.human_ensembl_gene != '-')
                                  & (raw_orthology.mouse_ensembl_gene != '-')]

    # Filter for genes in our datasets
    valid_orthology = raw_orthology[
        raw_orthology.human_ensembl_gene.isin(human_geneset)
        & raw_orthology.mouse_ensembl_gene.isin(mouse_geneset)]

    # Calculate database support for each ortholog pair
    orthology_grouped = (valid_orthology.groupby(
        ['human_ensembl_gene',
         'mouse_ensembl_gene']).agg(calculate_database_support).reset_index())

    # Select best human ortholog for each mouse gene
    human_grouped = orthology_grouped.set_index('human_ensembl_gene').groupby(
        level=0)
    best_for_human = pd.DataFrame(
        [select_best_ortholog(element) for element in human_grouped],
        columns=[
            'human_ensembl_gene', 'mouse_ensembl_gene', 'database_support'
        ]).dropna()

    # Select best mouse ortholog for each human gene
    mouse_grouped = best_for_human.set_index('mouse_ensembl_gene').groupby(
        level=0)
    best_for_mouse = pd.DataFrame(
        [select_best_ortholog(element) for element in mouse_grouped],
        columns=[
            'mouse_ensembl_gene', 'human_ensembl_gene', 'database_support'
        ]).dropna()

    # Add gene symbols
    final_orthology = best_for_mouse.copy()
    final_orthology['mouse_symbol'] = mouse_symbols.loc[
        final_orthology['mouse_ensembl_gene'], 'gene_symbol'].values
    final_orthology['human_symbol'] = human_symbols.loc[
        final_orthology['human_ensembl_gene'], 'gene_symbol'].values

    # Save results
    output_columns = [
        'human_ensembl_gene', 'mouse_ensembl_gene', 'human_symbol',
        'mouse_symbol', 'database_support'
    ]
    output_path = os.path.join(
        output_dir, f'{orthology_label}_hcop_{gene_type}_genes.txt')
    final_orthology[output_columns].to_csv(output_path, index=False)
    print(f"Saved {gene_type} orthology results to: {output_path}")

    return final_orthology


def get_mouse_human_orthologous(gencode_mouse_version: str,
                                gencode_human_version: str,
                                hcop_orthology_version: str,
                                path_to_gencode_mouse: str,
                                path_to_gencode_human: str,
                                path_to_hcop_orthology: str,
                                output_dir: str) -> None:
    """
    Main function to identify orthologous genes between mouse and human.
    
    This function processes GENCODE annotations for both species and HCOP 
    orthology data to create orthology mappings for both all genes
    and protein-coding genes only.
    
    Args:
        gencode_mouse_version (str): GENCODE mouse version (e.g., 'vM23')
        gencode_human_version (str): GENCODE human version (e.g., 'v22') 
        hcop_orthology_version (str): HCOP version timestamp (e.g., '20200106')
        path_to_gencode_mouse (str): Path to mouse GENCODE GTF file
        path_to_gencode_human (str): Path to human GENCODE GTF file
        path_to_hcop_orthology (str): Path to HCOP orthology file
        output_dir (str): Output directory path
    """
    # Validate input files
    for file_path, file_type in [(path_to_gencode_mouse, "Mouse GENCODE"),
                                 (path_to_gencode_human, "Human GENCODE"),
                                 (path_to_hcop_orthology, "HCOP orthology")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} file not found: {file_path}")

    # Create results directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create orthology label
    orthology_label = f"{hcop_orthology_version}_{gencode_human_version}_{gencode_mouse_version}"

    print("Starting mouse-human orthology analysis...")
    print(f"Output label: {orthology_label}")

    # Process GENCODE annotations
    species_data = {}
    file_paths = {
        'human': path_to_gencode_human,
        'mouse': path_to_gencode_mouse
    }

    for species in ['human', 'mouse']:
        geneset, protein_coding_set, ensemble_to_symbol = parse_gencode_annotations(
            file_paths[species], species)
        species_data[species] = {
            'geneset': geneset,
            'protein_coding': protein_coding_set,
            'symbols': ensemble_to_symbol
        }

    # Process orthology for both gene sets
    for include_all_genes in [True, False]:
        gene_type = "all" if include_all_genes else "protein_coding"
        gene_key = 'geneset' if include_all_genes else 'protein_coding'

        process_orthology_data(hcop_file_path=path_to_hcop_orthology,
                               human_geneset=species_data['human'][gene_key],
                               mouse_geneset=species_data['mouse'][gene_key],
                               human_symbols=species_data['human']['symbols'],
                               mouse_symbols=species_data['mouse']['symbols'],
                               output_dir=output_dir,
                               orthology_label=orthology_label,
                               gene_type=gene_type)

    print("Completed successfully!")


def main() -> None:

    # Load configuration from YAML file
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file_path}")

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    settings = config['orthology_analysis_settings']
    base_dir = settings['base_dir']
    gencode_mouse_path = settings['gencode_mouse_path']
    gencode_human_path = settings['gencode_human_path']
    hcop_orthology_path = settings['hcop_orthology_path']

    gencode_mouse_version = settings['gencode_mouse_version']
    gencode_human_version = settings['gencode_human_version']
    hcop_orthology_version = settings['hcop_orthology_version']

    output_dir = os.path.join(base_dir, 'orthology')

    try:
        get_mouse_human_orthologous(
            gencode_mouse_version=gencode_mouse_version,
            gencode_human_version=gencode_human_version,
            hcop_orthology_version=hcop_orthology_version,
            path_to_gencode_mouse=gencode_mouse_path,
            path_to_gencode_human=gencode_human_path,
            path_to_hcop_orthology=hcop_orthology_path,
            output_dir=output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
