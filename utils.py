##################################
## LLMs CONFIG
##################################
from google.colab import userdata

import datetime
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llama_index.llms.langchain import LangChainLLM
import google.generativeai as genai

#from langchain.chains import AnalyzeDocumentChain
import os
from Bio import Entrez, Medline
import re
import json
import csv
import pandas as pd
#import arxiv
#from langchain.document_loaders import ArxivLoader
#from langchain_community.document_loaders import ArxivLoader
#import requests
#from bs4 import BeautifulSoup
from typing import Annotated, Any
#from langchain_community.retrievers import ArxivRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv
load_dotenv()

#from prisma_ds import ScreenInfo, ScientificPaper
#from prisma_screener import PrismaScreener

entrez_email = userdata.get('ENTREZ_EMAIL')

def get_llm(engine, temperature=0.01):
    if engine == 'Gemini': 
      google_api_key=userdata.get('GOOGLE_KEY')
      genai.configure(api_key=google_api_key)    
      llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
      google_api_key=google_api_key,
      temperature=temperature)
      print("Get_llm - Gemini ", llm.model)
      return llm
    elif engine == 'OpenAI':
      openai_api_key = userdata.get('OPENAI_KEY')
      llm = ChatOpenAI(
        model = "gpt-4o-mini",
        api_key=openai_api_key,
        temperature=temperature
        )
      print("Get_llm - OpenAI ", llm.model_name)
      return llm
    else:
      print("No valid LLM")
      return None  
'''
def get_langchain_llm(engine='Gemini', temperature=0.01):
  
  openai_api_key = userdata.get('OPENAI_KEY')
  google_api_key=userdata.get('GOOGLE_KEY')

  if engine == 'Gemini': 
    genai.configure(api_key=google_api_key)    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
    google_api_key=google_api_key) #,
    #temperature=temperature)

  else: # OpenAI
    #openai.api_key = openai_api_key
    llm = ChatOpenAI(api_key=openai_api_key,
    #model="gpt-3.5-turbo", 
    model = 'gpt-4o-2024-08-06',
    temperature=temperature)
  
  return llm


def get_llamaindex_llm(engine='Gemini', temperature=0.01):
  
  llm = get_langchain_llm(engine, temperature)
  new_llm = LangChainLLM(llm = llm)
  return new_llm
'''

# basic semantic router for proceed or restart. TBD more options
def ask_llm(user_input, llm) -> str:
  prompt = f"""
  Analyze the user input as follows: {user_input} 
  If the user wants to continue with the review process, return CONT.
  Else, return BREAK
  """
  print(user_input)
  llm_output = get_llm(llm).invoke(prompt)
  print(llm_output.content)
  return llm_output.content


# Get today's date and time
def today() ->str:

  now = datetime.datetime.now()
  formatted_date = now.strftime("%d%m%y_%H%M")
  return formatted_date

## This method is used to merge the results of screening of the two LLMs into one file.
## Only papers that were screened by both LLMs are included. The output file is used
## as the input file for updating screening stats in the Eligibility step.
def write_combined_screened_info_to_csv(file_path1, file_path2, output_file_path):
  with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
      reader1 = csv.DictReader(f1)
      reader2 = csv.DictReader(f2)

      # Convert both files into dictionaries with PMID as the key
      data1 = {row['PMID']: row for row in reader1}
      data2 = {row['PMID']: row for row in reader2}

      # Prepare to write to the output file
      fieldnames = ['PMID', 'Title', 'PubDate', 'Source', 'Keywords', 'I1', 'I2', 'I3', 'I4', 'I5', 'ScreenStatus']
      with open(output_file_path, 'w', newline='') as outfile:
          writer = csv.DictWriter(outfile, fieldnames=fieldnames)
          writer.writeheader()

          for pmid in data1.keys():
            decision = ''
            if pmid in data2:
              status1 = data1[pmid]['ScreenStatus'].strip().lower()
              status2 = data2[pmid]['ScreenStatus'].strip().lower()
              if status1 == 'screened' and status2 == 'screened':
                #print(f"PMID {pmid} Screened")
                decision = 'Screened'
              else:
                decision = 'Rejected'

              # Write the combined data from the first file to the output file
              writer.writerow({
                  'PMID': pmid,
                  'Title': data1[pmid]['Title'],
                  'PubDate': data1[pmid]['PubDate'],
                  'Source': data1[pmid]['Source'],
                  'Keywords': data1[pmid]['Keywords'],
                  'I1': data1[pmid]['I1'].strip().lower() == 'true',
                  'I2': data1[pmid]['I2'].strip().lower() == 'true',
                  'I3': data1[pmid]['I3'].strip().lower() == 'true',
                  'I4': data1[pmid]['I4'].strip().lower() == 'true',
                  'I5': data1[pmid]['I5'].strip().lower() == 'true',
                  'ScreenStatus': decision
              })

  return

def write_screening_stats(file_path: str, output_path: str):
  # Load the CSV file
  df = pd.read_csv(file_path)

  # Group by PMID
  grouped = df.groupby('PMID')

  # Initialize an empty DataFrame to store the results
  results = pd.DataFrame()

  # List of columns to check for differences
  columns_to_check = ['I1', 'I2', 'I3', 'I4', 'I5', 'ScreenStatus']

  # Iterate over each group (each PMID)
  for name, group in grouped:
      # Dictionary to store total differences count for each column
      diff_counts = {}

      # Check differences across columns I1-I5 and ScreenStatus
      for column in columns_to_check:
          # Compare the column values with the first value in the group
          group[f'{column}_diff'] = group[column] != group[column].iloc[0]
          # Count the total number of differences for the current column
          diff_counts[f'{column}_diff_count'] = group[f'{column}_diff'].sum()

      # Add the difference counts to the first row of the group
      for col, count in diff_counts.items():
          group.loc[group.index[0], col] = count

      # Determine the decision based on ScreenStatus
      unique_screen_statuses = group['ScreenStatus'].unique()
      print(unique_screen_statuses, name)
      if len(unique_screen_statuses) == 1:
          # If all ScreenStatus are the same, assign 'Accepted' or 'Rejected'
          if unique_screen_statuses[0] == 'Screened':
              decision = 'Screened'
          elif unique_screen_statuses[0] == 'Rejected':
              decision = 'Rejected'
      else:
          # If there's a mix of ScreenStatus values, assign 'Conflict'
          decision = 'Conflict'
          print(f"Conflict {group['PMID']}")

      # Add the decision to the first row of the group
      group.loc[group.index[0], 'decision'] = decision

      # Append the group with the differences to the results DataFrame
      results = pd.concat([results, group])

  # Save the results to a new CSV file
  results.to_csv(output_path, index=False)


#########################################
# Pubmed utils
#########################################

# Function to load pubmed data as context
def search_pubmed_by_pmid(search_pmid: Annotated[str, "Valid pubmed PMID as search_query."]) -> str:
    """Loads one paper from Pubmed using its PMID"""

    ## TBD add parsing on user input to only send pmid to pubmed
    print(f"User input: {search_pmid}")

    pmid_pattern = r".*(\d{8}).*?"

    match = re.search(pmid_pattern, search_pmid)
    if match:
      pmid = match.group(1)
      print(f"PMID found in user input: {pmid}")
    else:
      print("No PMID found in user input")
      return "EMPTY"

    #try:

      #pap = ScientificPaper(pmid=pmid)
      #if pap.load_paper(pap.pmid) == True:
      #  str_pap = pap.to_string()
      #  print("Paper loaded successfully")
      #else:
      #  print("Paper not loaded")
      #return str_pap.format()

    #except Exception as e:
    #  print(f"Error retrieving or paper: {e}")
    #  return "EMPTY"

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def split_records(content):

  # Use double new line to separate records and ensure each starts with "PMID-"
  records = content.split("\n\n")
  return [record.strip() for record in records if record.strip().startswith("PMID-")]

def parse_record(record):
  # Bio.Entrez expects a file-like object, so we'll use StringIO to wrap the record string
  from io import StringIO
  handle = StringIO(record)
  parser = Medline.parse(handle)
  try:
    return next(parser)
  except StopIteration:
    #print("Warning: Empty Medline record encountered.")
    return None  # Return None for empty records

def trim_record(record):

  rec = {}

  rec['pmid'] = record.get("PMID", "")
  rec['title'] = record.get("TI", "")
  rec['abstract'] = record.get("AB", "")
  rec['pmc'] = record.get("PMC", "")
  rec['source'] = record.get("SO","")
  rec['pubilcation_date'] = record.get("DP", "")
  
  # Convert author list to a string
  author = record.get("AU", [])
  if isinstance(author, list):
      rec['author'] = ', '.join(author)

  # Convert language list to a string
  language = record.get("LA", [])
  if isinstance(language, list):
      rec['language'] = ', '.join(language)

  # Convert MeSH list to a string
  mesh = record.get("MH", [])
  if isinstance(mesh, list):
     rec['mesh'] = ', '.join(mesh)

  # Convert keyword list to a string
  keywords = record.get("OT", [])
  if isinstance(keywords, list):
      rec['keywords'] = ', '.join(keywords)

  #print(f" Saved record {rec.pmid}")
  return rec


def process_records(file_path):

  content = read_file(file_path)
  records = split_records(content)
  parsed_records = [parse_record(record) for record in records]
  papers = [trim_record(record) for record in parsed_records]
  return papers ### THIS IS A list of DICTS to be transformed into PaperContent


## Loads multiple papers from Pubmed using search_query
def search_pubmed(search_query: str, file_path: str) -> dict:

  # initialize some default parameters
  Entrez.email = entrez_email
  db = 'pubmed'                              # set search to pubmed database

  print(f"User input: {search_query}")

  paramEutils = { 'usehistory':'Y' }        # Use Entrez search history to cache results
  paramEutils['retstart'] = 1             #get result starting at 100,  so we skip latest (top of IdList)
  paramEutils['retmax'] = 500               # get 500 results each time
  
  # generate query to Entrez eSearch
  eSearch = Entrez.esearch(db=db, term=search_query, **paramEutils)

  # get eSearch result as dict object
  res = Entrez.read(eSearch)
  eSearch.close()
  records_count = int(res['Count'])

  # take a peek of what's in the result (ie. WebEnv, Count, etc.)
  for k in res:
      print (k, "=",  res[k])

  paramEutils['WebEnv'] = res['WebEnv']         #add WebEnv and query_key to eUtils parameters to request esummary using
  paramEutils['query_key'] = res['QueryKey']    #search history (cache results) instead of using IdList
  paramEutils['rettype'] = 'medline'            #get report as medline
  paramEutils['retmode'] = 'xml'                #get report as xml

  id_list = []
  while paramEutils['retstart'] < records_count:
      paramEutils['retmax'] = min(500, records_count - paramEutils['retstart'])
      eSearch = Entrez.esearch(db=db, term=search_query, **paramEutils)
      res = Entrez.read(eSearch)
      eSearch.close()
      id_list.extend(res['IdList'])
      paramEutils['retstart'] += paramEutils['retmax']

  # Fetch the records using the PMIDs
  i = 0
  records = []
  if len(id_list) > 0:
    print(f"Total PMIDs retrieved: {len(id_list)} first: {id_list[0]} last: {id_list[-1]}")
    for start in range(0, len(id_list), 500):
      handle = Entrez.efetch(db="pubmed", id=",".join(id_list[start:start+500]), rettype='medline', retmode='text', webenv=paramEutils['WebEnv'], query_key=paramEutils['query_key'])
      records_ = handle.read()
      print(i, id_list[start], type(records_), len(records_))
      i += 1
      records.extend(records_.split('\n\n'))  # Split records into individual entries
  
  else:
    print("No IDs retrieved.")
    return {} 
  
  records = "\n\n".join(records)
  
  # Save records to a file
  with open(file_path, "w") as f:
    f.write(records)
    print(f"Records saved to {file_path}")
    f.close()
    papers_dict = process_records(file_path)
  print(f'Retrieved {len(papers_dict)} records.')

  

  '''
  # Get the list of PubMed IDs (PMIDs)
  id_list = res["IdList"]
  print(f"Retrieved {len(id_list)} PMIDs. first: {id_list[0]}, last: {id_list[-1]}")

  # Fetch the records using the PMIDs
  handle = Entrez.efetch(db="pubmed", id=",".join(res["IdList"]), **paramEutils)
  records = handle.read()
  handle.close()

  cnt, papers_dict = process_records(records)
  print(f'Retrieved {cnt} records.')
  '''
  ## HELIT PLAY
  '''
  rct = cnt
  while rct < records_count:
    print("Ite 2 ,", cnt,records_count )

    #get next batch
    paramEutils['retstart'] += cnt
    # generate second query to Entrez eSearch
    eSearch = Entrez.esearch(db=db, term=new_search_query, **paramEutils)

    # get eSearch result as dict object
    res = Entrez.read(eSearch)
    eSearch.close()
    # Get the list of PubMed IDs (PMIDs)
    id_list = res["IdList"]
    print(f"Retrieved {len(id_list)} PMIDs. first: {id_list[0]}, last: {id_list[-1]}")


    cnt += len(id_list)
    print(f"Got {cnt} records")
  '''

  return papers_dict #process_records(records)
  

def get_mesh_id(term: Annotated[str, "term for MeSH lookup."]) -> str:
  '''Returns MeSH ID of term, if there is such'''
  Entrez.email = "helitb@my.hit.ac.il"
  db="mesh"
  handle = Entrez.esearch(db=db, term=term)
  record = Entrez.read(handle)
  if record["IdList"]:
    mesh_id = record["IdList"][0]
    return mesh_id
  else:
    return "No MeSH ID found for the term"



#  Use search queries to retrieve matching papers from Pubmed
# A dict of papers lists for all queries returned
def get_papers(queries: list, file_path: str):

  pubmed_papers = {}
  
  for query in queries:
    records_path = file_path.split('.txt')[0] + ' Q: ' + query + '.txt'
    print(f'search will be saved in {records_path}')
    pubmed_papers[query] = search_pubmed(query, records_path)

  return pubmed_papers 


#########################################
# Arxiv util
#########################################
# Search Arxiv for either a query or a list of IDs. 
def search_arxiv(query: str):

  arxiv = ArxivRetriever() # LALA TBD result count
  docs = arxiv.invoke(query)
  print("Arxiv doc count:", len(docs))
  return docs


  if isinstance(docs, list):
    paper = docs[0]
  else:
    paper = docs

  return f"""
  Title: {paper.metadata.title}
  Arxiv ID: {paper.metadata.arxiv_id}
  DOI: {paper.metadata.doi}
  Authors: {', '.join(paper.metadata.authors)}
  Abstract: {paper.metadata.abstract}
  Published: {paper.metadata.published}
  """

