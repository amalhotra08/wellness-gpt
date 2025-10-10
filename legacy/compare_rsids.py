# If any clarifications are needed, please let me know at rajarshi.m.1000@gmail.com.
### PUT THE INPUT FILE NAME HERE
# INPUT_FILE_NAME = '8.23andme.2.txt.tsv'
### PUT THE INPUT FILE NAME HERE

# Library imports
import numpy as np
import pandas as pd
import re
import warnings
import time
warnings.filterwarnings("ignore")

# Simple function
def remove_symbols(gen):
    regex = re.compile('[^a-zA-Z]')
    return regex.sub('', gen)

def compare_gene_conditions(input_file_name, output_file_name):
  start = time.time()
  # Fixing formatting inconsistencies
  filtered_snps = pd.read_csv('Filtered_SNPs.csv')
  print("LOG:", filtered_snps.columns)
  user_genome = pd.read_csv(input_file_name, sep='\t')
  filtered_snps['Genotype'] = filtered_snps['Genotype'].apply(remove_symbols)
  filtered_snps['ID'] = filtered_snps['ID'].apply(lambda id: id.lower())
  user_genome['genotype'] = user_genome['genotype'].apply(remove_symbols)
  gene_conditions = pd.DataFrame(columns=filtered_snps.columns)
  print('WAIT UNTIL THE NUMBER IS 96')

  # Iterating through the user's genome
  for ind,info in user_genome.iterrows():
    if ind%10000==0:
      print(ind/10000)
    inds = filtered_snps.index[filtered_snps['ID']==info['rsid']].to_list()
    if len(inds)!=0:
        for ind in inds:
            row = filtered_snps.iloc[ind]
            if row['Genotype'] == info['genotype']:
                gene_conditions.loc[len(gene_conditions.index)] = row
  print('DONE!')
  
  # Saving the file
  gene_conditions.to_csv(output_file_name,index=False)

# compare_gene_conditions('8.23andme.2.txt.tsv', 'gene_conditions.csv')

