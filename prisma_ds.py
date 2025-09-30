# prisma_ds.py
# Here we define the main data structures and process flows for our app

import os
#from re import I
import time
import json
import csv
from enum import Enum
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from prisma_screener import PrismaScreener, ScreenInfo
from prisma_eligibility import EligibilityInfo, ODH_eligibility_qs
from utils import today, get_llm 


##############################
# Paths and files

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/PRISMA'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)

eligible_records_path = GOOGLE_DRIVE_PATH + '/Eligible/OAH_AI - Eligible '
#eligible_papers_path = GOOGLE_DRIVE_PATH + '/papers'
screened_records_pmid_path = GOOGLE_DRIVE_PATH + f'/Screened/OAH_AI Screened PMIDs {today()}.txt'

##############################
# Data Structures 

class PRISMA_Stage(Enum):
  """
  Enum representing the processing stages of a PRISMA review.
  """
  IDENTIFY = "Identification"
  SCREEN = "Screening"
  ELIGIBLE = "Eligibilty"
  ASSES = "Full-Assessment"


class ProcessingStage(Enum):
  """
  Enum representing the processing stages of a scientific paper.
  """
  IDENTIFIED = "Identified"
  SCREENED = "Screened"
  ELIGIBLE = "Eligible"
  EXCLUDE = "Exclude"

class PaperContent():
  def __init__(self, pmid, title, abstract, mesh, keywords, authors, lang, source, pmc, pubilcation_date):
    
    # metadata and abstract from pubmed
    self.pmid=pmid
    self.title = title
    self.abstract=abstract
    self.mesh=mesh
    self.keywords=keywords
    self.lang=lang
    self.authors=authors
    self.source= source
    self.pmc = pmc
    self.pubilcation_date = pubilcation_date
    

# Using llama-index-embeddings-huggingface
def get_embed():
  embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
  Settings.embed_model = embed_model
  return 

class ScientificPaper():

  def __init__(self, pmid, title, abstract, mesh, keywords, authors, lang, source, pmc, pubilcation_date, embed_model, llm_model):
    
    # metadata and abstract from pubmed
    self.content = PaperContent(pmid, title, abstract, mesh, keywords, authors, lang, source, pmc, pubilcation_date)
    
    # PRISMA process metadata
    self.status = 'Identified'
    self.screen_info = ScreenInfo()
    self.embed_model = embed_model
    self.llm_model = llm_model

    self.eligibility_info = EligibilityInfo(pmid = self.content.pmid, 
          title = self.content.title, 
          embed_model = self.embed_model,
          llm_model = llm_model)

    # TBD predicates?
    self.OAH_info = {
      'domain': '',
      'predicates': [],
    #  'indicators': [],
    #  'relations': [],
    }
    
    self.ODH_info = {
      'keys': [],
      'perspectives': [],
      'dimensions': [],
    }
  
  def __str__(self):
    
    return f"""PMID: {self.content.pmid}
Title: {self.content.title}
Source: {self.content.source}
PMC: {self.content.pmc}
Authors: {self.content.authors}
Publication Date: {self.content.pubilcation_date}
Keywords: {self.content.keywords}
MeSH Terms: {self.content.mesh}
Language: {self.content.lang}
Abstract: {self.content.abstract}
"""

  def update_screen_info(self, info: ScreenInfo):
    #self.status = info.status
    self.screen_info = info
    #print(self.content.pmid, self.status )
    
  # Extracts indicators from a string.
  #  text: The string containing indicators delimited by double stars.
  #  Returns: a list of extracted indicators.
  def extract_ODH_indicators(self, text):
    indicators = []
    start = 0
    while True:
      start_index = text.find("**", start)
      if start_index == -1:
        break
      end_index = text.find("**", start_index + 2)
      if end_index == -1:
        break
      indicator = text[start_index + 2:end_index]
      indicators.append(indicator)
      start = end_index + 2
    return indicators
    
  # This method exracts data from answers into our DS using either python or llm
  def extract_findings(self, full_ass = False):

    self.OAH_info['domain'] = self.eligibility_info.domain
    
    findings = {
      "PMID": self.content.pmid,
      "Title": self.content.title,     
      "metadata": {
        "Source": self.content.source,
        "PMC": self.content.pmc,
        "Authors": self.content.authors,
        "PubDate": self.content.pubilcation_date,
        "Keywords": self.content.keywords,
        "MeSH": self.content.mesh,
        "Language": self.content.lang,
        "Abstract": self.content.abstract,
        },
        "domain": self.OAH_info['domain'],
        "OAH_score": {
            "score": self.eligibility_info.OAH_score,
            "rationale": str(self.eligibility_info.OAH_score_rational)
        },
        "questions": {},
        "ODH_score": {
            "score": self.eligibility_info.ODH_score,
            "rationale": str(self.eligibility_info.ODH_score_rational)
        },
        "ODH_info": {
            "keys": [],
            "perspectives": [],
            "dimensions": [],
        },
        #"predicates": '',
    }

    if not full_ass :
      return json.dumps(findings)

    for q in ODH_eligibility_qs.keys():
        if q == 'ODH_keys':
          findings["questions"][q] = self.extract_ODH_indicators(str(self.eligibility_info.llm_replies['ODH_keys']))
        elif q == 'ODH_perspectives':
          findings["questions"][q] = self.extract_ODH_indicators(str(self.eligibility_info.llm_replies['ODH_perspectives']))
        elif q == 'ODH_dimensions':
          findings["questions"][q] = self.extract_ODH_indicators(str(self.eligibility_info.llm_replies['ODH_dimensions']))
        else:
          findings["questions"][q] = str(self.eligibility_info.llm_replies[q])
        
    print(f"Extract: keys: {findings['ODH_info']['keys']} pers {findings['ODH_info']['perspectives']} dimen {findings['ODH_info']['dimensions']}")

    '''
    THIS IS LOVELY BUT WILL ONLY BE ENABLED IF WELL DESERVED
    predicate_str = str(self.eligibility_info.llm_replies['OAH_indicators']) + '\n' + str(self.eligibility_info.llm_replies['relations'])
    predicate_prompt = f"""
    Format the following text as a list of comma separated ontology predicates:

    {predicate_str}
    """
    print(predicate_str)
    llm_output = self.llm_model.invoke(predicate_prompt)
    print(llm_output.content)
    findings["predicates"] = llm_output.content
    '''
    return json.dumps(findings)


####################################
# Main Data Structure for the review 

class PrismaReview():

  def __init__(self, llm_model: str):
    self.state = PRISMA_Stage.IDENTIFY
    self.queries = []
    self.papers = {}
    self.paper_counts = {stage: 0 for stage in PRISMA_Stage}  
    
    # Init the LLM we will use. One for LANGCHAIN API, other for LLAMA-INDEX API. TBD 
    ## LALALA THIS + screenerInfo should move to prisma_screener TBD
    # We will use model name tp init langchain or llama-index llm model based on phase
    if llm_model == 'Gemini' or llm_model == 'OpenAI':
      self.llm_name = llm_model
      self.llm = get_llm(llm_model)
      print(f"LLM model {self.llm_name} initiated")
    else:
      print("Invalid model type. Choose 'OpenAI' or 'Gemini'.")
      raise ValueError("Invalid model type. Choose 'OpenAI' or 'Gemini'.")

    self.embed_model = get_embed() # We download once and init per Scientific paper

    # Init Screening chain
    self.screener = None
    self.exclusion_counts = {'I1': 0,
                             'I2': 0,
                             'I3': 0,
                             'I4': 0,
                             'I5': 0, 
                            }

  def add_queries(self, queries: list):
    self.queries.extend(q for q in queries)

  def set_state(self, state: PRISMA_Stage):
    self.state = PRISMA_Stage

  def add_paper(self, paper_dict: dict) -> ScientificPaper: 
    paper = ScientificPaper(pmid = paper_dict['pmid'], 
                          title =  paper_dict['title'],
                          abstract = paper_dict['abstract'], 
                          mesh = paper_dict['mesh'], 
                          keywords = paper_dict['keywords'], 
                          authors = paper_dict['author'], 
                          lang = paper_dict['language'], 
                          source = paper_dict['source'], 
                          pmc = paper_dict['pmc'], 
                          pubilcation_date = paper_dict['pubilcation_date'], 
                          embed_model = self.embed_model, 
                          llm_model = self.llm)  #We're sending llama-index llm
    
    #print(f"add_paper: {paper.content.pmid} {paper.content.source}")
    self.papers[paper.content.pmid] = paper 

  def check_duplicates_and_add_all(self, papers):
    
    pmid_set = set()
    
    # Add all exisitng pmids to the set
    for key in self.papers.keys():
      pmid_set.add(key)

    count = len(pmid_set)
    #unique = {}
    
    for paper in papers:
      if paper['pmid'] in pmid_set:
        continue
        #print(f"Duplicate found! {paper['pmid']}")       
      else:
        pmid_set.add(paper['pmid'])
        self.add_paper(paper)
        
        #unique.append(paper)
        #unique[paper.content.pmid] = paper

    #self.papers = {pmid: {} for pmid in unique.keys()}
    #self.papers.extend(unique)
    unique_count = len(pmid_set) - count
    print(f"Added {unique_count} unique records")
    return unique_count

  ## We write the same data to two files: .txt for the application use, and csv for output
  def screen_papers(self, output_path: str):
    
    # Init Screening chain
    self.screener = PrismaScreener(self.llm) 

    txt_output_file = output_path + '_' + self.llm_name + '.txt'
    csv_output_file = output_path + '_' + self.llm_name + '.csv'
    print(txt_output_file)

    i = 0
    screen_counts = 0
    start_time = time.time()
    token_count = 0
    with open(csv_output_file, 'a') as csv:  # Open the file in append mode
      csv.write('PMID,Title,PubDate,Source,Keywords,I1,I2,I3,I4,I5,ScreenStatus\n')
      with open(txt_output_file, 'a') as txt:  # Open the file in append mode
        txt.write(f'LLM Engine: {self.llm_name}\n\n')
        for pmid in self.papers.keys():
          print(f"Screening {pmid}")
          self.papers[pmid].status = 'Screened'
          info, counts = self.screener.screen_paper(self.papers[pmid])
          token_count += counts
          self.papers[pmid].update_screen_info(info)
          info_str = self.papers[pmid].screen_info.get_info()
          status_message = f"\nPMID: {self.papers[pmid].content.pmid}\n" + f"Title: {self.papers[pmid].content.title}\n" + f"PubDate: {self.papers[pmid].content.pubilcation_date}\n" + f"Source: {self.papers[pmid].content.source}\n" + info_str
          print('Done:\n', status_message)
          txt.write(status_message)  # Append the status message to the text file
          
          csv_str = f'{self.papers[pmid].content.pmid},\"{self.papers[pmid].content.title}\",\"{self.papers[pmid].content.pubilcation_date}\",\"{self.papers[pmid].content.source}\",\"{self.papers[pmid].content.keywords}\",{self.papers[pmid].screen_info.I1},{self.papers[pmid].screen_info.I2},{self.papers[pmid].screen_info.I3},{self.papers[pmid].screen_info.I4},{self.papers[pmid].screen_info.I5},{self.papers[pmid].screen_info.status}\n'
          #print(csv_str)
          csv.write(csv_str)
          if self.papers[pmid].screen_info.status == 'Screened':
              screen_counts += 1

          ## OAH_AI_POC SCOPE BREAK AT 1,000 PAPERS 
          i += 1
          if i == 1000:
      
            end_time = time.time()
            elapsed = end_time - start_time
            exec_time_str = f'\n###################\nTotal Execution Time: {elapsed}\n\n'
            txt.write(exec_time_str)
            print(exec_time_str)
            print(f"\n\nPapers total token count: {token_count}.\nAverage token count per paper: {token_count/i}")
            break
          
    txt.close()
    csv.close()
    return screen_counts
    
  def update_state_count(self, counts: int):
    self.paper_counts[self.state] = counts

  def update_exclusion_count(self):
    rejected_count = 0
    i = 0
    for pmid in self.papers.keys():
      if self.papers[pmid].status == 'Screened':
        #print(f"{i} update_exclusion_count {pmid} Screened. screen info status: {self.papers[pmid].screen_info.status} ")

        if self.papers[pmid].screen_info.status == 'Rejected':
          #print(f"{i} update_exclusion_count {pmid} rejected ")
          rejected_count += 1
          if self.papers[pmid].screen_info.I1 == False:
            self.exclusion_counts['I1'] += 1
          if self.papers[pmid].screen_info.I2 == False:
            self.exclusion_counts['I2'] += 1
          if self.papers[pmid].screen_info.I3 == False:
            self.exclusion_counts['I3'] += 1
          if self.papers[pmid].screen_info.I4 == False:
            self.exclusion_counts['I4'] += 1
          if self.papers[pmid].screen_info.I5 == False:
            self.exclusion_counts['I5'] += 1
          i += 1
    return rejected_count

  # This method prints screened papers' metadata to screen,
  # and saves {pmid, pmc, source} to file
  def print_screened_papers(self):

    i = 1
    print("#####################################################################################\n")
    print("The following papers passed the Screening stage and are up for eligibility assessment:")
    print("\n#####################################################################################\n")
    
    with open(screened_records_pmid_path, 'a') as f:
      
      f.write(f'LLM Engine: {self.llm_name}\n\n#################\n\n')
      
      for pmid in self.papers.keys():
        if self.papers[pmid].status == 'Screened':
          if self.papers[pmid].screen_info.status == 'Screened':
            print(f"\n({i}): {str(self.papers[pmid])}")
            f.write(f'{pmid}\n')
            i+=1 
    
  # this methods recieves the final screened list of papers
  def update_screened_info_from_csv(self, file_path):
    count = 0
    print(file_path)

    with open(file_path, 'r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        pmid = row['PMID'].strip()    # WHERE TF DID THESE CHARS COME FROM?!?!?
        if pmid not in self.papers:
            print("WTFFFF ", pmid)
            return -1
        
        # Update screening information
        self.papers[pmid].screen_info.I1 = row['I1'].strip().lower() == 'true'
        self.papers[pmid].screen_info.I2 = row['I2'].strip().lower() == 'true'
        self.papers[pmid].screen_info.I3 = row['I3'].strip().lower() == 'true'
        self.papers[pmid].screen_info.I4 = row['I4'].strip().lower() == 'true'
        self.papers[pmid].screen_info.I5 = row['I5'].strip().lower() == 'true'
        
        # Update status
        status = row['ScreenStatus'].strip()
        self.papers[pmid].status = status
        self.papers[pmid].screen_info.status = status
        
        if status == 'Screened':
            count += 1

    return count


  '''
  Screened papers file is formatted as follows:
  PMID: 38232590
  Title: Response of a pan-European fish index (EFI+) to multiple pressures in rivers across Spain.
  PubDate: 2024 Feb 14
  Source: source 
  I1: true
  I2: true
  I3: false
  I4: true
  I5: false
  Status: Rejected
  '''
  def update_screened_info_from_file__OLD(self, file_path):
    count = 0
    end_paper_block = 0
    with open(file_path, 'r') as f:
      for line in f:
        if 'PMID' in line:
          pmid = line.split(': ')[1].strip()
          #print(f"loading {pmid} from file")
          if pmid not in self.papers.keys():
            print("WTFFFF ", pmid)
            return -1
        if "I1" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I1 = True
        if "I2" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I2 = True
        if "I3" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I3 = True
        if "I4" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I4 = True
        if "I5" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I5 = True
        if "Status" in line:
          #print(f"PMID {pmid}: {self.papers[pmid].status}")
          self.papers[pmid].status = 'Screened'            
          if "Screened" in line:
            self.papers[pmid].screen_info.status = 'Screened'
            count += 1
          elif "Rejected" in line:
            self.papers[pmid].screen_info.status = 'Rejected'

          #print(f"updated {pmid} - screen status: {self.papers[pmid].screen_info.status} paper status: {self.papers[pmid].status}")
          #screen_info.print_info()
  
    return count


  def update_eligibility_info_from_json(self, json_file):
    
    findings_list = []
    count = 0
    with open(json_file, 'r') as file:
        for line in file:
            # Handle potential JSONDecodeError for individual lines
            try:
                findings = json.loads(line.strip())
                if findings['PMID'] in self.papers.keys():
                  self.papers[findings['PMID']].status = "Eligible"
                  self.papers[findings['PMID']].eligibility_info.domain = findings["domain"]
                  self.papers[findings['PMID']].eligibility_info.OAH_score = findings["OAH_score"]["score"]
                  self.papers[findings['PMID']].eligibility_info.OAH_score_rational = findings["OAH_score"]["rationale"]
                  self.papers[findings['PMID']].eligibility_info.ODH_score = findings["ODH_score"]["score"]
                  self.papers[findings['PMID']].eligibility_info.ODH_score_rational = findings["ODH_score"]["rationale"]
                  count += 1
                  print(f"{findings['PMID']} Screened for Eligibility, #{count}")
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}")
                print(e)  # Print the specific error for debugging
    return count


  ## Eligibility assessment methods.
  ## Inits after 'Screened' papers MANUALLY downloaded to Path
  ## Goes throgh each pdf in folder, finds matching screeend paper
  ## TBD WHAT TO DO IF NO MATCH IN DB
  ## classifies to HUMAN/ENV/OTHER and evaluates OAH and ODH relevance.
  ## Output is saved to csv and jsonl
  def eligibility_assessment(self, folder_path: str):

    csv_records_file = eligible_records_path + today() + '_' + self.llm_name + '_relevance.csv'
    eligible_records_file = eligible_records_path + today() + '_' + self.llm_name + '.jsonl'
    
    start_time = 0
    end_time = 0
    i = 0 #temp debug break
    with open(csv_records_file, 'a') as csv_file:
      with open(eligible_records_file, 'a') as json_file:
      
        csv_file.write('PMID,title,source,domain,OAH score,ODH score,decision,time\n')
        for pdf in os.listdir(folder_path):
          text = ''
          if pdf.endswith('.pdf') and os.path.isfile(os.path.join(folder_path, pdf)):
            paper = None
            pmid = pdf.split('.')[0]
            print(f" Eligibility scan for {pmid}")
            
            # WE FIND the matching paper and get
            if pmid in self.papers.keys():
              paper = self.papers[pmid]
              print(f"Found {pmid} in papers, status: {self.papers[pmid].status}")
            # !!!! THIS IS MAINLY FOR DEBUG FULL PROCESS SHOULD NOT ALLOW THIS and RAISE ERROR
            else: # !!!! THIS IS MAINLY FOR DEBUG FULL PROCESS SHOULD NOT ALLOW THIS
              print(f"WARNING {pmid} NOT found in papers")
              paper = ScientificPaper(pmid=pmid, title='', abstract='', mesh='', keywords='',
              authors='', lang='', source=pdf, pmc='', pubilcation_date='', 
              embed_model=self.embed_model, llm_model=self.llm)
              paper.screen_info.I1 = True
              paper.screen_info.I2 = True
              paper.screen_info.I3 = True
              paper.screen_info.I4 = True
              paper.screen_info.status = 'Screened'
              paper.status = 'Screened'
              self.papers[pmid] = paper
            
            if self.papers[pmid] and self.papers[pmid].status == 'Eligible':
              print(f"{pmid} already screened for Eligibility status: {self.papers[pmid].status}")
              continue
            
            if self.papers[pmid] and self.papers[pmid].status == 'Screened':
              
              start_time = time.time()

              self.papers[pmid].eligibility_info.init_query_engine(os.path.join(folder_path, pdf))
              self.papers[pmid].eligibility_info.domain = paper.eligibility_info.classify_paper()
              
              print(f"{i+1}: PMID {pmid} was classified under domain: {self.papers[pmid].eligibility_info.domain}")
                
              if self.papers[pmid].eligibility_info.domain == 'N/A':
                end_time = time.time()
                print(f"PMID {pmid} is irrelevant and excluded, ODH Score: {self.papers[pmid].eligibility_info.ODH_score}, OAH Score: {self.papers[pmid].eligibility_info.OAH_score}")
                self.papers[pmid].status = 'Exclude'
              else:
                self.papers[pmid].status = 'Eligible'
                
                # This method exracts data from answers into our DS and returns a jsonl 
                json_line = self.papers[pmid].extract_findings(full_ass = False)
                json_file.write(json_line + '\n')
            
                end_time = time.time()
                print(f"PMID {pmid} was classified under domain: {self.papers[pmid].eligibility_info.domain}")
        
              elapsed_time = end_time - start_time
              csv_file.write(f"{pmid},\"{self.papers[pmid].content.title}\",\"{self.papers[pmid].content.source}\",{self.papers[pmid].eligibility_info.domain},{str(self.papers[pmid].eligibility_info.OAH_score)},{str(self.papers[pmid].eligibility_info.ODH_score)},{self.papers[pmid].status},{elapsed_time}\n")

            ## TEMP DEBUG              
            #if (i == 3):
            #  break
            i += 1
              
  ## Full text assessment method.
  ## Reads from DB AFTER loading it from json file of eligible papers, 
  ## Chats to pdf and gets answers on full text.
  ## Output is saved to jsonl, txt files.
  
  def assess_full_text(self, input_path, output_path):
    print("assess_full_text")

    eligible_records_file = output_path + today() + '_' + self.llm_name + '.jsonl'
    processed_records_file = output_path + today() + '_' + self.llm_name + '_processed.txt'

    pdf =  ''
    text = ''
    start_time = 0
    end_time = 0
    i = 1

    with open(eligible_records_file, 'a') as json_file:
      with open(processed_records_file, 'a') as txt_file:
        for pmid in self.papers.keys():
          if self.papers[pmid].status == 'Eligible':
            pdf = input_path + pmid + '.pdf'
            print(f"assess_full_text - Eligible: {pdf}")
            self.papers[pmid].eligibility_info.init_query_engine(pdf)
            
            start_time = time.time()
            self.papers[pmid].eligibility_info.analyze_paper()

            # This method exracts data from answers into our DS and returns a jsonl 
            json_line = self.papers[pmid].extract_findings(full_ass = True)
            end_time = time.time()
            text = str(self.papers[pmid])
            text = text + '\nDomain: ' + self.papers[pmid].eligibility_info.domain + '\n' 
            summary = str(self.papers[pmid].eligibility_info.summary)
            if summary == '':
              print('somethings worng, breaking')
              break
            text = text + summary
            elapsed_time = end_time - start_time
          
            txt_file.write(f'*****\n{i}\tPMID {pmid}\n*****\n')
            txt_file.write(f'LLM: {self.llm_name}\tToken count: {self.papers[pmid].eligibility_info.token_count}\tProcessing time: {elapsed_time}\n')
            txt_file.write(text + '\n')
            
            json_file.write(json_line + '\n')
            i = i+1

    

  def update_eligibility_data_from_file(self, file_path):
    count = 0
    end_paper_block = 0
    with open(file_path, 'r') as f:
      for line in f:
        if 'PMID' in line:
          pmid = line.split(': ')[1].strip()
          #print(f"loading {pmid} from file")
          if pmid not in self.papers.keys():
            print("WTFFFF ", pmid)
            return -1
        if "I1" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I1 = True
        if "I2" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I2 = True
        if "I3" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I3 = True
        if "I4" in line:
          if "true" in line.lower():
            self.papers[pmid].screen_info.I4 = True
        if "Status" in line:
          #print(f"PMID {pmid}: {self.papers[pmid].status}")
          self.papers[pmid].status = 'Screened'            
          if "Screened" in line:
            self.papers[pmid].screen_info.status = 'Screened'
            count += 1
          elif "Rejected" in line:
            self.papers[pmid].screen_info.status = 'Rejected'

          #print(f"updated {pmid} - screen status: {self.papers[pmid].screen_info.status} paper status: {self.papers[pmid].status}")
          #screen_info.print_info()
  
    return count

  def update_fullscreen_data_from_file(self, file_path):
    return

  ### Here's the interesting part. Once we've analysed all eligible papers, 
  ### We move on to drawing insights from the combined dataset.
  ### agent brainstorming?? loop through papers? provide support evidence or new insight
  ### keep in mind that eligible_records can be quite a big dataset so we need to be clever about digesting it
  ### keep in mind are we doing it per domain or starting with paper:domain and looking at other domains for links??

  ### Indicator Identification: identify key indicators  related to {domain}. Suggest to group
  ### Relationship Extraction and Trend Analysis: Discover relationships between indicators, researches, and concepts in papers.
  ### Anomaly Detection: Identify indicators or relationships that may point to abnormal or alarming findings in the data.
  ### Insight Generation: Generate novel hypotheses or research directions based on the data.
  ### Critic, domain expert(s), knowledge engineer, reference searcher, indicator identifier, relationship  
  ### some agents should have tools - for searching online and access to query papers

  def draw_insights(self, eligible_papers_path):

    #TBD read file into ds 
    if eligible_papers_path != '':
      self.update_fullscreen_data_from_file(eligible_papers_path)

    text = ''








