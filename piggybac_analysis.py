'''Pipeline for the analysis of PiggyBac transposon insertions in a cohort of samples.

The pipeline starts from .bam files and performs the following steps:
1.  Identify TTAA positions in the reference genome.
2.  Identify insertions in each sample.
3.  Gather insertions across the cohort.
4.  Detect germline insertions using control samples.
5.  Filter insertions to remove insertions supported by less than 5 reads, cross-contamination, germline insertions and donor loci.
6.  Generate reference of genomic intervals that are expected to cause knockout of each gene when inserted.
7.  Group TTAA positions by gene and chromosome.
8.  Group insertions by gene and chromosome.
9.  Generate germline mask by gene.
10.  Identify significantly inserted genes in each sample.
11.  Compare subcohorts to identify differentially inserted genes.
12.  Identify significantly inserted genes across the cohort.

This script has been tested only with the conda environment defined in conda_environment.yml file.

'''

import yaml
import re
import pyranges as pyra
import os
import pandas as pd
import pysam
import numpy as np
import ast
from multiprocessing import Pool
import scipy.stats as st
import statsmodels.stats.multitest as stmu
from io import TextIOWrapper


def findall(pattern: str, string: str) -> list[int]:
    """
    Find all positions of a pattern in a string.
    
    Args:
        pattern (str): The pattern to search for
        string (str): The string to search in
        
    Yields:
        int: Position indices where pattern is found
    """
    i = string.find(pattern)
    while i != -1:
        yield i
        i = string.find(pattern, i + 1)


def generate_ttaa_positions(reference_file: str, ttaa_dir: str,
                            chromosomes: list[str]) -> None:
    """
    Generate TTAA positions for all chromosomes in a reference genome.
    
    Args:
        reference_file (str): Path to the reference FASTA file
        ttaa_dir (str): Output directory
        chromosomes (list[str]): List of chromosomes to process
    """

    os.makedirs(ttaa_dir, exist_ok=True)

    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    with pysam.FastaFile(reference_file) as fasta:
        print(f"Processing chromosomes\n")

        processed_count = 0
        for chromosome in chromosomes:
            # Get the chromosome sequence
            ref_string = fasta.fetch(chromosome)

            # Find all TTAA positions and convert to 1-based coordinates
            ttaa_positions = pd.Series(list(findall('TTAA', ref_string))) + 1

            if len(ttaa_positions) == 0:
                print(f"No TTAA sites found in {chromosome}")
                continue

            output_file = os.path.join(ttaa_dir,
                                       f'ttaa_positions_chr{chromosome}.h5')
            ttaa_positions.to_hdf(output_file,
                                  key='ttaas',
                                  complib='blosc:blosclz',
                                  complevel=9)

            print(f"{chromosome} ({len(ttaa_positions)} sites)",
                  end=' ',
                  flush=True)
            processed_count += 1

        print(f"\n\nTTAA position generation complete!")
        print(f"Processed {processed_count} chromosomes/contigs")
        print(f"Output saved to: {ttaa_dir}")


def analyze_insertions_per_sample(sample_information: pd.DataFrame,
                                  samples_folder: str,
                                  samples_bam_folder: str,
                                  ttaa_positions_dir: str,
                                  chromosomes: list[str],
                                  min_mapping_quality: int = 60) -> None:
    """
    Analyze transposon insertions for each sample by processing BAM files.
    
    Args:
        sample_information (pd.DataFrame): Sample metadata with sample IDs as index
        samples_folder (str): Output folder for sample-specific files
        samples_bam_folder (str): Path to folder containing sample BAM files
        ttaa_positions_dir (str): Path to directory containing TTAA position HDF5 files
        chromosomes (list): List of chromosome names to process
        min_mapping_quality (int): Minimum combined mapping quality threshold
    """
    # Create sample directory if it doesn't exist
    os.makedirs(samples_folder, exist_ok=True)

    # Load reference TTAA positions for each chromosome
    reference_ttaas = {}
    for chromosome in chromosomes:
        ttaa_file = os.path.join(ttaa_positions_dir,
                                 f'ttaa_positions_chr{chromosome}.h5')
        reference_ttaas[chromosome] = set(
            pd.read_hdf(ttaa_file, key='ttaas').tolist())
        print(
            f"Loaded {len(reference_ttaas[chromosome])} TTAA positions for chr{chromosome}"
        )

    def write_alignment_pair(writer: TextIOWrapper,
                             first_read: pysam.AlignedSegment,
                             second_read: pysam.AlignedSegment) -> None:
        """Write alignment pair information to file.
        
        Args:
            writer (TextIOWrapper): The file writer object.
            first_read (pysam.AlignedSegment): The first read object.
            second_read (pysam.AlignedSegment): The second read object.
        """
        # Process first read
        ttaa_pos = 'NA'
        if first_read.reference_id == -1:
            first_pos = 'NA'
            first_chromosome = 'NA'
        else:
            first_chromosome = first_read.reference_name
            if first_read.get_reference_positions():
                first_pos = first_read.get_reference_positions(
                )[-first_read.is_reverse]
                ttaa_pos = first_pos + [+1, -2][first_read.is_reverse]
            else:
                first_pos = 'NA'

        # Process second read
        if second_read.reference_id == -1:
            second_pos = 'NA'
            second_chromosome = 'NA'
        else:
            second_chromosome = second_read.reference_name
            if second_read.get_reference_positions():
                second_pos = second_read.get_reference_positions(
                )[-second_read.is_reverse]
            else:
                second_pos = 'NA'

        # Write tab-separated line
        fields = [
            first_read.query_name, first_chromosome,
            str(ttaa_pos),
            str(first_pos),
            str(first_read.mapping_quality),
            str(int(first_read.is_reverse)), second_chromosome,
            str(second_pos),
            str(second_read.mapping_quality),
            str(int(second_read.is_reverse))
        ]
        writer.write('\t'.join(fields) + '\n')

    # Process each sample
    for sample_id in sample_information.index:
        print(f"Processing sample: {sample_id}")

        seen_insertions = {}
        output_file = os.path.join(samples_folder, f'{sample_id}_hits.csv')

        with open(output_file, 'w') as hits_writer:
            # Write header
            header = 'chromosome_F\tttaa_position\tposition_F\tposition_R\tis_on_ttaa\tF_reverse\tR_reverse\tPB5\tPB3\n'
            hits_writer.write(header)

            # Process both PB5 and PB3 BAM files
            for pb_type in ['PB5', 'PB3']:
                bam_file = os.path.join(samples_bam_folder,
                                        f'{sample_id}_{pb_type}.bam')
                bed_file = os.path.join(samples_folder,
                                        f'{sample_id}_hits_{pb_type}.bed')

                print(f"  Processing {pb_type}: {bam_file}")

                # Process BAM file and write to temporary BED file

                with pysam.AlignmentFile(bam_file, "rb") as sam_file:
                    with open(bed_file, 'w') as bed_writer:
                        read_dictionary = [{}, {}]  # [read1, read2]

                        for current in sam_file:
                            # Skip secondary and supplementary alignments
                            if current.is_secondary or current.is_supplementary:
                                continue

                            # If we have the pair, write it out
                            if current.query_name in read_dictionary[
                                    current.is_read1]:
                                to_be_written = [0, 0]
                                to_be_written[
                                    current.is_read1] = read_dictionary[
                                        current.is_read1][current.query_name]
                                to_be_written[not current.is_read1] = current
                                write_alignment_pair(bed_writer,
                                                     to_be_written[0],
                                                     to_be_written[1])

                            # Store read for later pairing
                            read_dictionary[not current.is_read1][
                                current.query_name] = current

                # Process BED file to identify valid insertions
                with open(bed_file, 'r') as bed_reader:
                    for line in bed_reader:
                        fields = line.strip().split('\t')
                        if len(fields) < 10:
                            continue

                        (query_name, first_chromosome, ttaa_position,
                         first_position, first_mapping_quality, first_reverse,
                         second_chromosome, second_position,
                         second_mapping_quality, second_reverse) = fields

                        # Apply mapping quality filter
                        combined_quality = int(first_mapping_quality) + int(
                            second_mapping_quality)
                        if combined_quality >= min_mapping_quality and first_chromosome in reference_ttaas:

                            # Check if insertion is on a TTAA site
                            if ttaa_position == 'NA':
                                ttaa_index = -1
                                is_on_ttaa = 'False'
                            else:
                                ttaa_index = int(ttaa_position)
                                is_on_ttaa = str(
                                    ttaa_index in
                                    reference_ttaas[first_chromosome])

                            # Create unique insertion signature
                            insertion_key = '\t'.join([
                                first_chromosome, ttaa_position,
                                first_position, second_position, is_on_ttaa,
                                first_reverse, second_reverse
                            ]) + '\t'

                            # Count insertions by PB type
                            if insertion_key not in seen_insertions:
                                seen_insertions[insertion_key] = [
                                    0, 0
                                ]  # [PB5_count, PB3_count]

                            pb_index = 1 if pb_type == 'PB3' else 0
                            seen_insertions[insertion_key][pb_index] += 1

                # Remove the temporary BED file
                os.remove(bed_file)

            # Write final results
            for insertion_key, counts in seen_insertions.items():
                output_line = insertion_key + '\t'.join(map(str,
                                                            counts)) + '\n'
                hits_writer.write(output_line)

        print(f"  Found {len(seen_insertions)} unique insertions")

    print("\nInsertion analysis complete!")


def gather_cohort_insertions(sample_information: pd.DataFrame,
                             chromosomes: list[str],
                             samples_folder: str,
                             cohort_insertions_folder: str,
                             insertions_cutoff: int = 5) -> None:
    """
    Gather insertions across cohort samples and create merged insertion matrices.
    
    This function reads insertion data from individual samples, merges them into
    cohort-wide matrices, and applies cutoff filters for downstream analysis.
    
    Args:
        sample_information (pd.DataFrame): Sample metadata with sample IDs as index
        chromosomes (list): List of chromosome names to process
        samples_folder (str): Path to folder containing sample files
        cohort_insertions_folder (str): Path to folder to save cohort insertions
        insertions_cutoff (int): Minimum number of insertions to consider (default: 5)
    
    Output:
        Creates HDF5 files for each chromosome containing:
        - insertions_chr{X}_cutoff{N}.hdf: Insertion count matrix
        - depth_chr{X}_cutoff{N}.hdf: Read depth matrix
    """

    os.makedirs(cohort_insertions_folder, exist_ok=True)

    insertions_dict = {}
    depth_dict = {}
    for chromosome in chromosomes:
        insertions_dict[chromosome] = {}
        depth_dict[chromosome] = {}

    chromosomes_index = pd.Index(chromosomes)

    print('Reading insertions from each sample:')

    processed_samples = 0
    for sample_id in sample_information.index:
        print(sample_id, end=' ', flush=True)

        # Read sample hits file
        hits_file = os.path.join(samples_folder, f'{sample_id}_hits.csv')
        insertions = pd.read_csv(hits_file, sep='\t', index_col=0)

        ttaa_insertions = insertions[insertions.is_on_ttaa]
        found_chromosomes = chromosomes_index.intersection(
            ttaa_insertions.index.unique())

        for chromosome in chromosomes:
            if chromosome in found_chromosomes:
                chromosomal_insertions = ttaa_insertions.loc[[chromosome]]

                # Determine insertion direction
                # PB5 PCR amplifies the 3' end of the transposon
                is5 = chromosomal_insertions.PB5 >= chromosomal_insertions.PB3
                read_1_R = chromosomal_insertions.F_reverse == 1
                direction_reverse = (is5 == read_1_R).astype(int)

                # Count fragments by position and direction
                grouped_data = chromosomal_insertions.groupby([
                    chromosomal_insertions.ttaa_position.astype(int),
                    direction_reverse
                ])

                counted_fragments = grouped_data.count().is_on_ttaa
                depth = grouped_data.sum().loc[:, ['PB3', 'PB5']].sum(axis=1)

            else:
                # No insertions found for this chromosome
                counted_fragments = pd.Series([], dtype=float)
                depth = pd.Series([], dtype=float)

            # Store results with sample_id as name
            counted_fragments.name = sample_id
            depth.name = sample_id
            insertions_dict[chromosome][sample_id] = counted_fragments
            depth_dict[chromosome][sample_id] = depth

        processed_samples += 1

    print(
        f'\n\nProcessed {processed_samples}/{len(sample_information)} samples')
    print('Saving chromosome data:', end=' ', flush=True)

    # Save merged data for each chromosome
    saved_chromosomes = 0
    for chromosome in chromosomes:
        print(chromosome, end=' ', flush=True)
        # Create merged DataFrames from dictionaries
        merged_insertions = pd.DataFrame.from_dict(insertions_dict[chromosome])
        merged_depth = pd.DataFrame.from_dict(depth_dict[chromosome])

        # Apply cutoff filter (values below cutoff become NaN)
        merged_insertions[merged_insertions < insertions_cutoff] = pd.NA
        merged_depth[merged_insertions < insertions_cutoff] = pd.NA

        # Filter rows: keep only positions where at least one sample meets cutoff
        valid_positions = merged_insertions.max(axis=1) >= insertions_cutoff
        filtered_insertions = merged_insertions[valid_positions]
        filtered_depth = merged_depth[valid_positions]

        # Save to HDF5 files
        insertions_file = os.path.join(
            cohort_insertions_folder,
            f'insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf')
        depth_file = os.path.join(
            cohort_insertions_folder,
            f'depth_chr{chromosome}_cutoff{insertions_cutoff}.hdf')

        filtered_insertions.to_hdf(insertions_file,
                                   key='insertions',
                                   complib='blosc:blosclz',
                                   complevel=9)
        filtered_depth.to_hdf(depth_file,
                              key='depth',
                              complib='blosc:blosclz',
                              complevel=9)

        saved_chromosomes += 1

    print(f'\n\nCohort insertion gathering complete')
    print(
        f'Processed {saved_chromosomes} chromosomes with cutoff >= {insertions_cutoff}'
    )
    print(f'Output saved to: {cohort_insertions_folder}')


def detect_germline_insertions(sample_information: pd.DataFrame,
                               chromosomes: list[str],
                               cohort_insertions_folder: str,
                               germline_filter_folder: str,
                               insertions_cutoff: int = 5,
                               germline_size: int = 1_000_000,
                               germline_insertions_cutoff: int = 10) -> None:
    """
    Detects and masks germline transposon insertions, and calculates masked genomic regions.

    Args:
        sample_information (pd.DataFrame): DataFrame with sample metadata, including 'sample_type', 'mouse_id', and 'transposon'.
        chromosomes (list[str]): List of chromosome names to process.
        cohort_insertions_folder (str): Path to directory containing merged cohort insertion HDF5 files.
        germline_filter_folder (str): Path to output directory for germline data and masks.
        insertions_cutoff (int, optional): Minimum insertion count in controls to consider a site germline. Defaults to 5.
        germline_size (int, optional): Size of the genomic window (bp) around a germline insertion to mask. Defaults to 1,000,000.
        germline_insertions_cutoff (int, optional): Minimum number of insertions to consider a site germline. Defaults to 10.
    """

    os.makedirs(germline_filter_folder, exist_ok=True)
    germline_list = os.path.join(germline_filter_folder, 'germline_list.tsv')
    total_masked_bases = os.path.join(germline_filter_folder,
                                      'total_masked_bases.tsv')

    # Identify control samples and create mappings
    controls = sample_information[sample_information.sample_type ==
                                  'control'].index
    mouse_to_control = pd.Series(
        sample_information.index, index=sample_information.mouse_id)[(
            sample_information.sample_type == 'control').values]
    control_to_sample = pd.Series(
        sample_information.index,
        index=mouse_to_control.loc[sample_information.mouse_id].values)
    masked_per_sample = {sample: [] for sample in controls}

    with open(germline_list, 'wt') as w, open(total_masked_bases, 'wt') as w2:
        for chromosome in chromosomes:
            print(chromosome, flush=True, end=' ')

            # Load merged insertions for the current chromosome
            merged = pd.read_hdf(os.path.join(
                cohort_insertions_folder,
                f'insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'),
                                 key='insertions')
            merged_positions = pd.Series(merged.index.get_level_values(0))

            # Filter for control samples and identify germline insertions
            control_merged = merged.loc[:, controls]
            germline_data = control_merged[control_merged >=
                                           germline_insertions_cutoff].stack()

            # Initialize germline mask
            germline_mask = merged.copy()
            germline_mask.loc[:, :] = 1

            # Apply masking for donor locus on chromosome 5 (transposon 'H')
            if chromosome == '5':
                germline_mask.loc[(
                    merged_positions[merged_positions.
                                     searchsorted(50_000_000):merged_positions.
                                     searchsorted(70_000_000)],
                    slice(None)), sample_information[
                        sample_information.transposon == 'H'].index] = 0
                for control_H in sample_information[
                    (sample_information.sample_type == 'control')
                        & (sample_information.transposon == 'H')].index:
                    masked_per_sample[control_H].append(
                        [chromosome, 50_000_000, 70_000_000])

            # Apply masking for donor locus on chromosome 10 (transposon 'S')
            if chromosome == '10':
                germline_mask.loc[(
                    merged_positions[merged_positions.
                                     searchsorted(0):merged_positions.
                                     searchsorted(10_000_000)],
                    slice(None)), sample_information[
                        sample_information.transposon == 'S'].index] = 0
                for control_S in sample_information[
                    (sample_information.sample_type == 'control')
                        & (sample_information.transposon == 'S')].index:
                    masked_per_sample[control_S].append(
                        [chromosome, 0, 10_000_000])

            for (position, direction,
                 control), fragments in germline_data.items():
                masked_region_start = position - germline_size
                masked_region_end = position + germline_size

                # Mask regions around germline insertions for associated samples
                germline_mask.loc[(
                    merged_positions[merged_positions.
                                     searchsorted(masked_region_start
                                                  ):merged_positions.
                                     searchsorted(masked_region_end)],
                    slice(None)), control_to_sample[control]] = 0

                # Mask the exact germline position and direction across all samples
                germline_mask.loc[(position, direction), :] = 0

                # Write germline events to file
                newline = f'{chromosome}\t{position}\t{direction}\t{control}\t{fragments}\n'
                w.write(newline)
                masked_per_sample[control].append(
                    [chromosome, masked_region_start, masked_region_end])

            # Save the germline mask for the current chromosome
            germline_mask.to_hdf(os.path.join(
                germline_filter_folder,
                f'germline_chr{chromosome}_cutoff{insertions_cutoff}.hdf'),
                                 key='insertions',
                                 complib='blosc:blosclz',
                                 complevel=9)

        # Calculate and write total masked bases per sample
        for control in controls:
            merged_intervals = pyra.PyRanges(
                pd.DataFrame(masked_per_sample[control],
                             columns='Chromosome Start End'.split())).merge()
            total_bases = (merged_intervals.End - merged_intervals.Start).sum()

            # Write masked bases for all samples associated with this control
            for sample in control_to_sample[control]:
                w2.write(
                    f'{sample}\t{total_bases}\t{sample_information.loc[sample, "tissue"]}\n'
                )


def filter_and_mask_insertions(
    chromosomes: list[str],
    insertions_cutoff: int,
    cohort_insertions_folder: str,
    germline_filter_folder: str,
) -> None:
    """
    Filters insertion sites based on fragment counts across samples and applies germline masking.

    Args:
        chromosomes (list[str]): List of chromosome names to process.
        insertions_cutoff (int): Minimum fragment count cutoff used for cohort insertions.
        cohort_insertions_folder (str): Path to directory containing merged cohort insertion HDF5 files.
        germline_filter_folder (str): Path to directory containing germline mask HDF5 files.
    """

    # Calculate total insertions per sample across all chromosomes
    total_per_sample = []
    for chromosome in chromosomes:
        merged = pd.read_hdf(os.path.join(
            cohort_insertions_folder,
            f'insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'),
                             key='insertions').fillna(0)
        total_per_sample.append(merged.sum())
    total_per_sample = pd.concat(total_per_sample, axis=1).sum(
        axis=1) + 1  # Add 1 to avoid division by zero

    for chromosome in chromosomes:
        merged = pd.read_hdf(os.path.join(
            cohort_insertions_folder,
            f'insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'),
                             key='insertions').fillna(0)
        print(chromosome, flush=True, end=' ')

        # Load the germline mask for the current chromosome
        germline_mask = pd.read_hdf(os.path.join(
            germline_filter_folder,
            f'germline_chr{chromosome}_cutoff{insertions_cutoff}.hdf'),
                                    key='insertions').fillna(0)

        filtered = merged.copy()

        # Calculate maximum normalized insertion count per site to resolve ties
        max_unfiltered = merged + merged / total_per_sample

        # Identify and filter out samples that are not the 'max' (or tied with max preference) for a given site
        less_than_max = (max_unfiltered.max(axis=1).values[:, None]
                         != max_unfiltered)
        filtered[less_than_max] = 0

        # save the filtered insertions
        filtered[filtered.sum(
            axis=1
        ).astype(bool)].to_hdf(os.path.join(
            cohort_insertions_folder,
            f'filtered_insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'
        ),
                               key='insertions',
                               complib='blosc:blosclz',
                               complevel=9)

        # Apply germline mask to the filtered insertions
        filtered_masked = (filtered * germline_mask).fillna(0)

        # Save the final filtered and masked insertions
        filtered_masked[filtered_masked.sum(
            axis=1
        ).astype(bool)].to_hdf(os.path.join(
            cohort_insertions_folder,
            f'filtered_masked_insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'
        ),
                               key='insertions',
                               complib='blosc:blosclz',
                               complevel=9)

    print('filtering and germline masking complete')


def generate_gene_reference(orthology: pd.DataFrame, gencode_path: str,
                            base_dir: str) -> None:
    """
    Calculates KO intervals for mouse genes.

    Parameters:
    - orthology: DataFrame with orthology mapping.
    - gencode_path: str, path to the GENCODE GTF file.
    - base_dir: str, path to the base directory for the analysis.
    """

    gene_re = re.compile(r'gene_id "(ENSMUSG\d+)')
    transcript_re = re.compile(r'transcript_id "(ENSMUST\d+)')

    # Keep only genes with human orthologs
    valid_genes = set(orthology['mouse_ensembl_gene'])

    # Read exon and conding sequences lines
    gtf = (pd.read_csv(
        gencode_path, sep='\t', comment='#',
        header=None).loc[lambda df: df[2].isin(['exon', 'CDS'])])
    gtf.columns = [
        'chromosome', 'source', 'feature', 'start', 'end', 'score', 'strand',
        'frame', 'info'
    ]

    # Filter entries: protein_coding transcripts of interest
    keep = gtf['info'].str.contains('transcript_type "protein_coding"') & gtf[
        'info'].str.extract(gene_re)[0].isin(valid_genes)
    filtered = gtf.loc[
        keep,
        ['chromosome', 'feature', 'start', 'end', 'strand', 'info']].copy()

    filtered['gene'] = filtered['info'].str.extract(gene_re)
    filtered['transcript'] = filtered['info'].str.extract(transcript_re)

    # Collect KO intervals per gene
    records = []
    for gene, group in filtered.groupby('gene'):
        strand = group['strand'].iat[0]
        chrom = group['chromosome'].iat[0]
        ko_intervals = []
        gene_starts = []
        gene_ends = []
        gene_coding_starts = []
        gene_coding_ends = []

        # For each transcript, define KO region
        for transcript, transcript_group in group.groupby('transcript'):
            exons = transcript_group[transcript_group['feature'] == 'exon']
            cds = transcript_group[transcript_group['feature'] == 'CDS']
            transcript_start = exons['start'].min()
            transcript_end = exons['end'].max()
            cds_end = cds['end'].max()
            cds_start = cds['start'].min()
            gene_starts.append(transcript_start)
            gene_ends.append(transcript_end)
            gene_coding_starts.append(cds_start)
            gene_coding_ends.append(cds_end)

            if strand == '+':
                ko_intervals.append((transcript_start, cds_end))
            else:
                ko_intervals.append((cds_start, transcript_end))

        # Merge overlapping KO segments
        pyra_df = pyra.PyRanges(
            pd.DataFrame({
                'Chromosome': chrom,
                'Start': [r[0] for r in ko_intervals],
                'End': [r[1] for r in ko_intervals]
            }))
        merged = pyra_df.merge().df[['Start', 'End']].values.tolist()

        records.append({
            'mouse_ensembl_gene': gene,
            'chromosome': chrom,
            'strand': strand,
            'start': min(gene_starts),
            'end': max(gene_ends),
            'coding_start': min(gene_coding_starts),
            'coding_end': max(gene_coding_ends),
            'KO_intervals': merged
        })

    ko_df = pd.DataFrame(records).set_index('mouse_ensembl_gene')

    ko_df['mouse_symbol'] = (
        orthology.set_index('mouse_ensembl_gene')['mouse_symbol'].reindex(
            ko_df.index))
    output_path = f'{base_dir}/gene_reference.csv'
    ko_df.to_csv(output_path)
    print(f"gene reference saved to: {output_path}")


def group_ttas_by_gene_and_chromosome(base_dir: str) -> None:
    """
    Groups TTAA positions by gene and chromosome.

    Parameters:
    - base_dir: str, path to the base directory for the analysis.
    """

    mouse_genes = pd.read_csv(os.path.join(base_dir, 'gene_reference.csv'),
                              index_col=1)
    gene_background_data = []
    chromosome_background_data = []
    for chromosome_name, chromosomal_genes in mouse_genes.groupby(level=0):
        chromosome = chromosome_name.lstrip('chr')
        chromosomal_KO_intervals = []

        print('\nchromosome', chromosome, flush=True)
        ttaas = pd.read_hdf(os.path.join(
            base_dir, f'TTAA_positions/ttaa_positions_chr{chromosome}.h5'),
                            key='ttaas')
        ttaas.index = ttaas
        for gene, gene_group in chromosomal_genes.groupby(
                'mouse_ensembl_gene'):
            gene_data = gene_group.loc[chromosome_name]
            KO_ttaas = 0
            for KO_start, KO_end in eval(gene_data.KO_intervals):
                chromosomal_KO_intervals.append([chromosome, KO_start, KO_end])
                KO_ttaas += ((ttaas >= KO_start) & (ttaas <= KO_end)).sum()
            gene_background_data.append(
                [chromosome, gene, gene_data.KO_intervals, KO_ttaas])

        chromosome_background_entry = [chromosome]
        for intervals in [chromosomal_KO_intervals]:
            chromosome_range_merged = pyra.PyRanges(
                pd.DataFrame(np.array(intervals),
                             columns='Chromosome Start End'.split())).merge()
            chromosome_range_intervals = list(
                zip(chromosome_range_merged.Start,
                    chromosome_range_merged.End))
            intervals_ttaas = 0
            for interval_start, interval_end in chromosome_range_intervals:
                intervals_ttaas += ((ttaas >= interval_start) &
                                    (ttaas <= interval_end)).sum()
            chromosome_background_entry += [
                chromosome_range_intervals, intervals_ttaas
            ]

        chromosome_background_data.append(chromosome_background_entry)
    gene_background_df = pd.DataFrame(
        gene_background_data,
        columns='chromosome mouse_ensembl_gene KO_intervals KO_ttaas'.split(
        )).set_index('mouse_ensembl_gene')
    gene_background_df.to_csv(f'{base_dir}/gene_ttaas.csv')

    chromosome_background_df = pd.DataFrame(
        chromosome_background_data,
        columns='chromosome KO_intervals KO_ttaas'.split()).set_index(
            'chromosome')
    chromosome_background_df.to_csv(f'{base_dir}/chromosome_ttaas.csv')


def group_insertions_by_gene_and_chromosome(base_dir: str,
                                            sample_information: pd.DataFrame,
                                            insertions_cutoff: int) -> None:
    """
    Calculates total insertion fragments within KO (Knockout) regions for each gene
    and for merged chromosomal KO regions. Saves results to HDF5 and CSV files.

    Args:
        base_dir (str): The base directory where input files (gene_ttaas.csv,
                        chromosome_ttaas.csv, filtered_masked_insertions_chr{X}_cutoff{Y}.hdf)
                        are located and where output files will be saved.
        sample_information (pd.DataFrame): DataFrame containing sample metadata.
                                           Used to derive sample names (index).
        insertions_cutoff (int): The insertion count cutoff used for filtering.
    """
    print('Calculating insertions by gene and chromosome')

    insertions_by_gene_folder = os.path.join(base_dir,
                                             f'insertions_by_gene_matrix')
    os.makedirs(insertions_by_gene_folder, exist_ok=True)

    sample_names = sample_information.index.tolist()

    # Template Series for summing fragments, initialized with zeros
    zero_template = pd.Series(index=sample_names, dtype=int).fillna(0)

    gene_background = pd.read_csv(os.path.join(base_dir, 'gene_ttaas.csv'),
                                  index_col=1)
    genome_background = pd.read_csv(os.path.join(base_dir,
                                                 'chromosome_ttaas.csv'),
                                    index_col=0)

    for prefix in ['filtered_', 'filtered_masked_']:
        print(f'Processing {prefix}insertions')
        insertions_by_gene = []
        insertions_by_chromosome = []

        # Group gene background data by chromosome
        gene_grouped_by_chromosome = gene_background.groupby('chromosome')

        for chromosome, chromosome_group in gene_grouped_by_chromosome:
            print(chromosome, end=' ', flush=True)

            filtered_insertions_path = os.path.join(
                base_dir,
                f'cohort_insertions/{prefix}insertions_chr{chromosome}_cutoff{insertions_cutoff}.hdf'
            )
            filtered_insertions = pd.read_hdf(
                filtered_insertions_path,
                key='insertions').fillna(0).astype(int)

            # Iterate over each gene in the current chromosome
            for gene, gene_data in chromosome_group.set_index(
                    'mouse_ensembl_gene').iterrows():

                ko_intervals = ast.literal_eval(gene_data.KO_intervals)

                all_fragments_gene = zero_template.copy()
                for range_start, range_end in ko_intervals:
                    # Sum fragments within the current KO interval
                    in_range_fragments = filtered_insertions.loc[
                        range_start:range_end].sum()
                    all_fragments_gene += in_range_fragments

                # Append gene-level data
                insertions_by_gene.append([gene, chromosome] +
                                          all_fragments_gene.to_list())

            # Process insertions by chromosome (merged KO ranges)
            chromosome_background = genome_background.loc[chromosome]

            chrom_ko_intervals = ast.literal_eval(
                chromosome_background.KO_intervals)

            all_fragments_chrom = zero_template.copy()
            for range_start, range_end in chrom_ko_intervals:

                # Sum fragments within the current chromosomal KO range
                in_range_fragments = filtered_insertions.loc[
                    range_start:range_end].sum()
                all_fragments_chrom += in_range_fragments

            insertions_by_chromosome.append([chromosome] +
                                            all_fragments_chrom.to_list())

        # Header for gene-level DataFrame
        header_gene = 'gene chromosome'.split() + sample_names
        insertions_by_gene_df = pd.DataFrame(insertions_by_gene,
                                             columns=header_gene)

        # Header for chromosome-level DataFrame
        header_chromosome = 'chromosome'.split() + sample_names
        insertions_by_chromosome_df = pd.DataFrame(insertions_by_chromosome,
                                                   columns=header_chromosome)

        # Save gene-level data to HDF5
        insertions_by_gene_df.to_hdf(os.path.join(
            insertions_by_gene_folder,
            f'{prefix}insertions_by_gene_cutoff{insertions_cutoff}.hdf'),
                                     key='insertions',
                                     complib='blosc:blosclz',
                                     complevel=9)

        # Save chromosome-level data to CSV
        insertions_by_chromosome_df.to_csv(
            os.path.join(
                insertions_by_gene_folder,
                f'{prefix}insertions_by_chromosome_cutoff{insertions_cutoff}.csv'
            ),
            index=False  # Do not write DataFrame index to CSV
        )

    print('\nInsertion counting by gene and chromosome complete')


def generate_germline_mask_by_gene(
    base_dir: str,
    sample_information: pd.DataFrame,
    germline_size: int = 1_000_000,
) -> None:
    """
    Generates a gene by sample mask based on the germline insertions.
    
    Args:
        base_dir (str): The base directory where input files are located and where output files will be saved.
        sample_information (pd.DataFrame): DataFrame containing sample metadata.
        germline_size (int): The size of the germline region to consider.
    """

    controls = sample_information[sample_information.sample_type ==
                                  'control'].index
    control_to_mouse = sample_information[sample_information.sample_type ==
                                          'control'].mouse_id
    mouse_to_control = pd.Series(control_to_mouse.index,
                                 index=control_to_mouse)
    control_to_sample = pd.Series(
        sample_information.index,
        index=mouse_to_control.loc[sample_information.mouse_id].values)
    mouse_genes = pd.read_csv(os.path.join(base_dir, 'gene_reference.csv'),
                              index_col=0)
    germline_genes = pd.DataFrame(index=mouse_genes.index,
                                  columns=sample_information.index).fillna(1)

    germline_insertions = pd.read_csv(os.path.join(
        base_dir, 'germline_filter/germline_list.tsv'),
                                      sep='\t',
                                      index_col=0,
                                      header=None)
    chromosome_grouped = germline_insertions.groupby(level=0)
    for chromosome, chromosome_group in chromosome_grouped:
        position_grouped = chromosome_group.groupby(1)
        for position, position_group in position_grouped:
            mask_start = position - germline_size
            mask_end = position + germline_size
            genes_bool = (mouse_genes.start <= mask_end) & (mouse_genes.end
                                                            >= mask_start)
            genes_to_mask = genes_bool[genes_bool].index
            affected_tumors = []
            for chromosome, control_info in position_group.iterrows():
                affected_tumors += control_to_sample.loc[
                    control_info.loc[3]].to_list()
            germline_genes.loc[genes_to_mask, affected_tumors] = 0

        if chromosome == '5':
            genes_bool = (mouse_genes.start <= 70000000) & (mouse_genes.end
                                                            >= 50000000)
            genes_to_mask = genes_bool[genes_bool].index
            affected_tumors = sample_information[sample_information.transposon
                                                 == 'H'].index
            germline_genes.loc[genes_to_mask, affected_tumors] = 0

        if chromosome == '10':
            genes_bool = (mouse_genes.start <= 10000000)
            genes_to_mask = genes_bool[genes_bool].index
            affected_tumors = sample_information[sample_information.transposon
                                                 == 'S'].index
            germline_genes.loc[genes_to_mask, affected_tumors] = 0

    germline_genes.to_csv(os.path.join(base_dir,
                                       'germline_filter_by_gene.csv'))


def single_sample_statistics(
    arguments: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                     str]
) -> None:
    """
    Poisson-based statistical analysis in a single sample.

    Args:
        arguments (tuple): A tuple containing the following arguments:
            - gene_ttaas: DataFrame containing gene-level TTAA counts.
            - chromosomal_ttaas: DataFrame containing chromosomal TTAA counts.
            - sample_insertions: DataFrame containing sample-level insertions.
            - sample_masked_insertions: DataFrame containing sample-level masked insertions.
            - statistics_per_sample_folder: str, path to the folder where the statistics will be saved.
    """

    gene_ttaas, chromosomal_ttaas, sample_insertions, sample_masked_insertions, statistics_per_sample_folder = arguments

    chromosomal_insertions = sample_insertions.groupby(level=1).sum()

    gene_rate = (sample_insertions / gene_ttaas).to_frame().fillna(0)

    chromosomal_rate = (
        chromosomal_insertions /
        chromosomal_ttaas).loc[gene_rate.index.get_level_values('chromosome')]

    expected_insertions = gene_ttaas.loc[
        gene_rate.index] * chromosomal_rate.values

    results = pd.DataFrame(index=gene_rate.index,
                           data={
                               'insertions':
                               sample_masked_insertions.loc[gene_rate.index],
                               'expected':
                               expected_insertions
                           })
    ps = []
    for (gene, chromsome), (insertions, expected) in results.iterrows():
        p = st.poisson.sf(insertions - 1, expected)
        ps.append(p)
    results['p'] = ps
    qs = stmu.multipletests(results.loc[:, 'p'], alpha=0.05,
                            method='fdr_bh')[1]
    results['q'] = qs
    results.to_csv(
        f'{statistics_per_sample_folder}/{sample_insertions.name}_KO_intervals_statistics.csv'
    )
    print(sample_insertions.name, flush=True)


def statistics_per_sample(base_dir: str,
                          statistics_folder: str,
                          sample_information: pd.DataFrame,
                          insertions_cutoff: int,
                          threads: int = 8) -> None:
    """
    Performs statistical analysis for each sample in the cohort.

    Args:
        base_dir (str): The base directory where input files are located and where output files will be saved.
        statistics_folder (str): The folder where the statistics will be saved.
        sample_information (pd.DataFrame): DataFrame containing sample metadata.
        insertions_cutoff (int): The cutoff for the number of insertions.
        threads (int): The number of threads to use.
    """

    gene_ttaas = pd.read_csv(os.path.join(base_dir, 'gene_ttaas.csv'),
                             index_col=(0, 1)).loc[:, 'KO_ttaas']
    gene_ttaas.index.name = 'gene'

    chromosomal_ttaas = gene_ttaas.groupby(level=1).sum()

    statistics_per_sample_folder = os.path.join(statistics_folder,
                                                'per_sample')
    os.makedirs(statistics_per_sample_folder, exist_ok=True)

    insertions_by_gene = pd.read_hdf(
        os.path.join(
            base_dir,
            f'insertions_by_gene_matrix/filtered_insertions_by_gene_cutoff{insertions_cutoff}.hdf'
        )).set_index(['gene', 'chromosome'])

    masked_insertions_by_gene = pd.read_hdf(
        os.path.join(
            base_dir,
            f'insertions_by_gene_matrix/filtered_masked_insertions_by_gene_cutoff{insertions_cutoff}.hdf'
        )).set_index(['gene', 'chromosome'])

    arguments_list = []
    for sample in insertions_by_gene.columns:
        sample_insertions = insertions_by_gene.loc[:, sample]
        sample_masked_insertions = masked_insertions_by_gene.loc[:, sample]
        arguments = (gene_ttaas, chromosomal_ttaas, sample_insertions,
                     sample_masked_insertions, statistics_per_sample_folder)
        arguments_list.append(arguments)

    p = Pool(threads)
    p.map(single_sample_statistics, arguments_list, chunksize=1)
    p.close()
    p.join()

    cohort_significant = {}
    for sample in insertions_by_gene.columns:
        significant_genes = (pd.read_csv(
            f'{statistics_per_sample_folder}/{sample}_KO_intervals_statistics.csv',
            index_col=0).q < 0.05).astype(int)
        cohort_significant[sample] = significant_genes
    df = pd.DataFrame(cohort_significant)
    df.to_csv(f'{statistics_folder}/cohort_KO_intervals_significant.csv')


def update_comparison_results(masked_cohort: pd.DataFrame,
                              equal_or_more_extreme: pd.Series,
                              current_reps: int, orthology: pd.DataFrame,
                              cohort_label_1: str, cohort_label_2: str,
                              H_in_1: pd.Index, S_in_1: pd.Index,
                              H_in_2: pd.Index, S_in_2: pd.Index,
                              total_in_1: int, total_in_2: int,
                              statistics_folder: str) -> None:
    """
    Prints the current comparison results.

    Args:
        masked_cohort (pd.DataFrame): DataFrame containing masked cohort data.
        equal_or_more_extreme (pd.Series): Series containing the number of times each gene is equal or more extreme than the observed value.
        current_reps (int): The current number of repetitions.
        orthology (pd.DataFrame): DataFrame containing orthology information.
        cohort_label_1 (str): The label of the first cohort.
        cohort_label_2 (str): The label of the second cohort.
        H_in_1 (pd.Index): The index of the H samples in the first cohort.
        S_in_1 (pd.Index): The index of the S samples in the first cohort.
        H_in_2 (pd.Index): The index of the H samples in the second cohort.
        S_in_2 (pd.Index): The index of the S samples in the second cohort.
        total_in_1 (int): The total number of samples in the first cohort.
        total_in_2 (int): The total number of samples in the second cohort.
        statistics_folder (str): The folder where the statistics will be saved.
    """

    final = pd.DataFrame(
        index=masked_cohort.index,
        data={'times_equal_or_more_extreme': equal_or_more_extreme - 1})

    #one is added here to the denominator - reflecting the real, observed configuration
    final.loc[:, 'p'] = equal_or_more_extreme / (current_reps + 1)
    final.loc[:, 'mouse_symbol'] = orthology.set_index(
        'mouse_ensembl_gene').loc[final.index, 'mouse_symbol']
    final.loc[:, 'q'] = stmu.multipletests(final.loc[:, 'p'].copy(),
                                           alpha=0.05,
                                           method='fdr_bh')[1]
    final.index.name = 'mouse_ensembl_gene'

    final.loc[:, cohort_label_1 + '_H' + '_' +
              str(H_in_1.shape[0])] = masked_cohort.loc[:, H_in_1].sum(axis=1)
    final.loc[:, cohort_label_1 + '_S' + '_' +
              str(S_in_1.shape[0])] = masked_cohort.loc[:, S_in_1].sum(axis=1)
    final.loc[:, cohort_label_1 + '_' +
              str(total_in_1
                  )] = final.loc[:, cohort_label_1 + '_H' + '_' +
                                 str(H_in_1.shape[0]
                                     )] + final.loc[:, cohort_label_1 + '_S' +
                                                    '_' + str(S_in_1.shape[0])]

    final.loc[:, cohort_label_2 + '_H' + '_' +
              str(H_in_2.shape[0])] = masked_cohort.loc[:, H_in_2].sum(axis=1)
    final.loc[:, cohort_label_2 + '_S' + '_' +
              str(S_in_2.shape[0])] = masked_cohort.loc[:, S_in_2].sum(axis=1)
    final.loc[:, cohort_label_2 + '_' +
              str(total_in_2
                  )] = final.loc[:, cohort_label_2 + '_H' + '_' +
                                 str(H_in_2.shape[0]
                                     )] + final.loc[:, cohort_label_2 + '_S' +
                                                    '_' + str(S_in_2.shape[0])]

    final.to_csv(
        f'{statistics_folder}/{cohort_label_1}_vs_{cohort_label_2}_rep{current_reps}.csv'
    )


def _run_permutation_batch(
    arguments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int,
                     int, np.ndarray]
) -> np.ndarray:
    """
    Helper function to run a batch of permutation iterations.

    Args:
        arguments (tuple): A tuple containing the following arguments:
            - insertions_H_values: numpy array containing the insertions for H samples.
            - insertions_S_values: numpy array containing the insertions for S samples.
            - pre_sampled_H_batch_indices: numpy array containing the pre-sampled H indices.
            - pre_sampled_S_batch_indices: numpy array containing the pre-sampled S indices.
            - total_in_1: int, total number of samples in the first cohort.
            - total_in_2: int, total number of samples in the second cohort.
            - total_found: numpy array containing the total number of found insertions.
            - difference: numpy array containing the difference between the observed and expected insertions.
    """
    insertions_H_values, insertions_S_values, pre_sampled_H_batch_indices, pre_sampled_S_batch_indices, total_in_1, total_in_2, total_found, difference = arguments

    batch_equal_or_more_extreme = np.zeros(insertions_H_values.shape[0],
                                           dtype=int)

    num_permutations_in_batch = pre_sampled_H_batch_indices.shape[1]

    for i in range(num_permutations_in_batch):
        sampled_H_indices = pre_sampled_H_batch_indices[:, i]
        sampled_S_indices = pre_sampled_S_batch_indices[:, i]

        shuffled_found_1 = insertions_H_values[:, sampled_H_indices].sum(
            axis=1) + insertions_S_values[:, sampled_S_indices].sum(axis=1)
        shuffled_found_2 = total_found - shuffled_found_1

        shuffled_difference = abs(shuffled_found_1 / total_in_1 -
                                  shuffled_found_2 / total_in_2)
        batch_equal_or_more_extreme += (shuffled_difference >= difference)

    return batch_equal_or_more_extreme


def subcohort_comparison(base_dir: str,
                         statistics_folder: str,
                         subcohorts: dict[str, pd.DataFrame],
                         cohort_label_1: str,
                         cohort_label_2: str,
                         repetitions: int,
                         orthology: pd.DataFrame,
                         threads: int = 8) -> None:
    """
    Permutation-based statistical analysis of differentially inserted genes between two subcohorts.

    Args:
        base_dir (str): The base directory where input files are located and where output files will be saved.
        statistics_folder (str): The folder where the statistics will be saved.
        subcohorts (dict[str, pd.DataFrame]): A dictionary containing the subcohorts.
        cohort_label_1 (str): The label of the first cohort.
        cohort_label_2 (str): The label of the second cohort.
        repetitions (int): The number of repetitions.
        orthology (pd.DataFrame): DataFrame containing orthology information.
        threads (int): The number of threads to use.
    """

    germline_filter = pd.read_csv(f'{base_dir}/germline_filter_by_gene.csv',
                                  index_col=0)

    cohort_1 = subcohorts[cohort_label_1]
    cohort_2 = subcohorts[cohort_label_2]

    joint_H_index = cohort_1[cohort_1.transposon == 'H'].index.union(
        cohort_2[cohort_2.transposon == 'H'].index)
    joint_S_index = cohort_1[cohort_1.transposon == 'S'].index.union(
        cohort_2[cohort_2.transposon == 'S'].index)

    H_in_1 = cohort_1[cohort_1.transposon == 'H'].index
    S_in_1 = cohort_1[cohort_1.transposon == 'S'].index
    H_in_2 = cohort_2[cohort_2.transposon == 'H'].index
    S_in_2 = cohort_2[cohort_2.transposon == 'S'].index

    n_H_1 = H_in_1.shape[0]
    n_S_1 = S_in_1.shape[0]

    total_in_1 = cohort_1.shape[0]
    total_in_2 = cohort_2.shape[0]
    joint_index = joint_H_index.union(joint_S_index)

    cohort = pd.read_csv(
        f'{statistics_folder}/cohort_KO_intervals_significant.csv',
        index_col=0)
    masked_cohort = (cohort * germline_filter).loc[:,
                                                   joint_index].copy().astype(
                                                       np.int32)

    total_found = masked_cohort.loc[:, joint_index].sum(axis=1).values

    fraction_insertions_1 = masked_cohort.loc[:, cohort_1.index].mean(
        axis=1).values
    fraction_insertions_2 = masked_cohort.loc[:, cohort_2.index].mean(
        axis=1).values
    difference = np.abs(fraction_insertions_1 - fraction_insertions_2)

    insertions_H_values = masked_cohort.loc[:, joint_H_index].values
    insertions_S_values = masked_cohort.loc[:, joint_S_index].values

    num_H_samples = len(joint_H_index)
    num_S_samples = len(joint_S_index)

    # Initialize with 1 for observed value
    equal_or_more_extreme = pd.Series(data=1, index=masked_cohort.index)
    print(
        f"Starting permutation test with {repetitions} total repetitions and {threads} threads"
    )

    reporting_chunk_size = 100000  # For progress reports
    permutations_per_worker_batch = 5000  # Each worker processes this many permutations per task

    with Pool(processes=threads) as pool:
        # Calculate the total number of batches that will be sent to workers
        total_batches = (repetitions + permutations_per_worker_batch -
                         1) // permutations_per_worker_batch

        args_for_pool = []
        for batch_idx in range(total_batches):
            current_batch_size = min(
                permutations_per_worker_batch,
                repetitions - batch_idx * permutations_per_worker_batch)
            if current_batch_size <= 0:
                break

            # For H: Generate ranks for num_H_samples in 'current_batch_size' repetitions
            pre_sampled_H_batch_indices = np.array([
                np.random.choice(num_H_samples, n_H_1, replace=False)
                for _ in range(current_batch_size)
            ]).T
            # For S: Generate ranks for num_S_samples in 'current_batch_size' repetitions
            pre_sampled_S_batch_indices = np.array([
                np.random.choice(num_S_samples, n_S_1, replace=False)
                for _ in range(current_batch_size)
            ]).T

            args_for_pool.append((
                insertions_H_values,
                insertions_S_values,
                pre_sampled_H_batch_indices,
                pre_sampled_S_batch_indices,
                total_in_1,
                total_in_2,
                total_found,
                difference,
            ))

        current_total_reps_completed = 0

        for i, batch_result_series in enumerate(
                pool.imap_unordered(_run_permutation_batch,
                                    args_for_pool,
                                    chunksize=1)):
            equal_or_more_extreme += batch_result_series  # Add the accumulated results from the batch
            current_total_reps_completed += permutations_per_worker_batch  # Update total count

            if current_total_reps_completed > repetitions:
                current_total_reps_completed = repetitions

            if current_total_reps_completed % reporting_chunk_size == 0 or current_total_reps_completed == repetitions:
                print(
                    f"{cohort_label_1} vs {cohort_label_2}: {current_total_reps_completed} repetitions completed",
                    flush=True)

    update_comparison_results(masked_cohort, equal_or_more_extreme,
                              current_total_reps_completed, orthology,
                              cohort_label_1, cohort_label_2, H_in_1, S_in_1,
                              H_in_2, S_in_2, total_in_1, total_in_2,
                              statistics_folder)

    print(
        f"\nPermutation test complete for {cohort_label_1} vs {cohort_label_2}. Total repetitions: {repetitions}"
    )


def cohort_statistics(base_dir: str, statistics_folder: str,
                      cohort_information: pd.DataFrame,
                      orthology: pd.DataFrame) -> None:
    """
    Performs statistical analysis for the entire cohort.

    Args:
        base_dir (str): The base directory where input files are located and where output files will be saved.
        statistics_folder (str): The folder where the statistics will be saved.
        cohort_information (pd.DataFrame): DataFrame containing cohort information.
        orthology (pd.DataFrame): DataFrame containing orthology information.
    """

    germline_filter = pd.read_csv(f'{base_dir}/germline_filter_by_gene.csv',
                                  index_col=0).loc[:, cohort_information.index]

    cohort = pd.read_csv(
        f'{statistics_folder}/cohort_KO_intervals_significant.csv',
        index_col=0).loc[:, cohort_information.index]
    masked_cohort = cohort * germline_filter

    total_hits = masked_cohort.sum().sum()
    total_masked_genes = germline_filter.sum().sum()
    hits_per_gene = total_hits / total_masked_genes
    samples_per_gene = germline_filter.sum(axis=1)
    expected_hits_per_gene = samples_per_gene * hits_per_gene
    found_hits_per_gene = masked_cohort.sum(axis=1)
    results = pd.DataFrame({
        'expected': expected_hits_per_gene,
        'observed': found_hits_per_gene
    })
    ps = []
    for gene, (expected, observed) in results.iterrows():
        ps.append(st.poisson.sf(observed - 1, expected))
    qs = stmu.multipletests(ps, alpha=0.05, method='fdr_bh')[1]
    results.loc[:, 'p'] = ps
    results.loc[:, 'q'] = qs
    results.loc[:, 'gene'] = orthology.set_index('mouse_ensembl_gene').loc[
        results.index, 'mouse_symbol']
    results.index.name = 'mouse_ensembl_gene'
    print('total significant hits',
          cohort.loc[:, cohort_information.index].sum().sum())
    results.to_csv(f'{statistics_folder}/cohort_statistics.csv')


def main() -> None:
    """Main function to run PiggyBac insertion analysis."""

    # Load configuration from YAML file
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file_path}")

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    settings = config['piggybac_pipeline_run_settings']

    # Use settings from the config file
    base_dir = settings['base_dir']
    reference_file = settings['reference_genome_path']
    gencode_path = settings['gencode_gtf_path']
    chromosomes = settings['chromosomes']
    insertions_cutoff = settings['insertions_cutoff']
    germline_size = settings['germline_size']
    germline_insertions_cutoff = settings['germline_insertions_cutoff']
    threads = settings['threads']
    repetitions_subcohort_comparison = settings[
        'repetitions_subcohort_comparison']
    subcohort_comparisons = settings['subcohort_comparisons']
    orthology_path = settings['orthology_path']
    samples_bam_folder = settings['samples_bam_folder']
    subcohort_comparisons = settings['subcohort_comparisons']

    # Set up paths
    sample_info_file = os.path.join(base_dir,
                                    'piggybac_sample_information.csv')
    ttaa_dir = os.path.join(base_dir, 'TTAA_positions')
    samples_folder = os.path.join(base_dir, 'samples')
    cohort_insertions_folder = os.path.join(base_dir, 'cohort_insertions')
    germline_filter_folder = os.path.join(base_dir, 'germline_filter')
    insertions_by_gene_folder = os.path.join(base_dir,
                                             'insertions_by_gene_matrix')
    statistics_folder = os.path.join(base_dir, 'statistics')

    orthology = pd.read_csv(orthology_path, index_col=0)

    # Load sample information
    sample_information = pd.read_csv(sample_info_file).set_index('sample_id')
    print(f"Loaded information for {len(sample_information)} samples")
    print(f"working in {base_dir}")

    print("Step 1: Detecting TTAA positions")
    generate_ttaa_positions(reference_file, ttaa_dir, chromosomes)

    print("Step 2: Analyzing insertions per sample")
    analyze_insertions_per_sample(sample_information, samples_folder,
                                  samples_bam_folder, ttaa_dir, chromosomes)

    print("Step 3: Gathering insertions across cohort")
    gather_cohort_insertions(sample_information,
                             chromosomes,
                             samples_folder,
                             cohort_insertions_folder,
                             insertions_cutoff=insertions_cutoff)

    print("Step 4: Detecting germline insertions")
    detect_germline_insertions(
        sample_information,
        chromosomes,
        cohort_insertions_folder,
        germline_filter_folder,
        insertions_cutoff=insertions_cutoff,
        germline_size=germline_size,
        germline_insertions_cutoff=germline_insertions_cutoff)

    print("Step 5: Filtering and masking insertions")
    filter_and_mask_insertions(chromosomes, insertions_cutoff,
                               cohort_insertions_folder,
                               germline_filter_folder)

    print("Step 6: Generating KO intervals")
    generate_gene_reference(orthology, gencode_path, base_dir)

    print("Step 7: Grouping TTAA by gene")
    group_ttas_by_gene_and_chromosome(base_dir)

    print("Step 8: Grouping insertions by gene")
    group_insertions_by_gene_and_chromosome(base_dir, sample_information,
                                            insertions_cutoff)

    print("Step 9: Generating germline mask by gene")
    generate_germline_mask_by_gene(
        base_dir,
        sample_information,
        germline_size,
    )

    print("Step 10: Identifying significant insertions per sample")
    statistics_per_sample(base_dir,
                          statistics_folder,
                          sample_information,
                          insertions_cutoff,
                          threads=threads)

    subcohorts = {}
    subcohorts['all_samples'] = sample_information[
        sample_information.sample_type != 'control']
    subcohorts['untreated'] = sample_information[
        (sample_information.treatment == 'untreated')
        & (sample_information.sample_type != 'control')]
    subcohorts['chemotherapy'] = sample_information[
        (sample_information.treatment == 'etoposide_cisplatin')
        & (sample_information.sample_type != 'control')]
    subcohorts['aPD1'] = sample_information[
        (sample_information.treatment == 'aPD1')
        & (sample_information.sample_type != 'control')]
    subcohorts['metastasis'] = sample_information[(
        sample_information.sample_type == 'metastasis')]
    subcohorts['lung_untreated'] = sample_information[
        (sample_information.sample_type == 'primary')
        & (sample_information.treatment == 'untreated')]

    print("Step 11: Performing subcohort comparisons")
    for cohort_label_1, cohort_label_2 in subcohort_comparisons:
        print(
            f'{cohort_label_1} ({len(subcohorts[cohort_label_1])}) vs {cohort_label_2} ({len(subcohorts[cohort_label_2])})'
        )
        subcohort_comparison(base_dir,
                             statistics_folder,
                             subcohorts,
                             cohort_label_1,
                             cohort_label_2,
                             repetitions=repetitions_subcohort_comparison,
                             orthology=orthology,
                             threads=threads)

    print("Step 12: Performing overall statistical analysis")
    cohort_statistics(base_dir, statistics_folder, subcohorts['all_samples'],
                      orthology)

    print("Analysis completed")


if __name__ == "__main__":
    main()
