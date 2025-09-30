# prisma_insights.py

### Here's the interesting part. Once we've analysed all eligible papers, 
### We move on to drawing insights from the combined dataset.
### keep in mind that eligible_records can be quite a big dataset so we need to be clever about digesting it
### are we doing it per domain or starting with paper:domain and looking at other domains for links??
### Indicator Identification: identify key indicators  related to {domain}. Suggest to group
### Relationship Extraction and Trend Analysis: Discover relationships between indicators, researches, and concepts in papers.
### Anomaly Detection: Identify indicators or relationships that may point to abnormal or alarming findings in the data.
### Insight Generation: Generate novel hypotheses or research directions based on the data.
### Critic, domain expert(s), knowledge engineer, reference searcher, indicator identifier, relationship  
### some agents should have tools - for searching online and access to query papers


import os
import autogen
from google.colab import userdata
import json
import tiktoken
import random
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from utils import get_llm, today
from prisma_eligibility import ODH_eligibility_qs
from ODH_definitions import OneAquaHealth

##############################
# Paths and files

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/PRISMA'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
ideas_folder_path = GOOGLE_DRIVE_PATH + '/Ideas/'
ouputs_folder_path = GOOGLE_DRIVE_PATH + '/Outputs/OAH-AI/'

# TBD limit report count sent to summarize
CONTEXT_WINDOW_MAX_TOKENS = 30000

def get_autogen_llm(model_name='OpenAI'):

  llm_config = {}
  if model_name == 'Gemini':
    llm_config = [
      {
          "model": "gemini-1.5-flash",
          "api_key": userdata.get('GOOGLE_KEY'),
          "api_type": "google"
      }
    ]

  elif model_name == 'OpenAI':
    llm_config =  [
      {
          #'model': 'gpt-4',
          #'model': "gpt-3.5-turbo",
          'model': "gpt-4o-mini",
          "api_key": userdata.get('OPENAI_KEY'),
      }
    ]
  else:
    raise ValueError("Invalid model type. Choose 'OpenAI' or 'Gemini'.")
  
  return llm_config

# token estimation - Returns the number of tokens in a text string
def num_tokens_from_string(string) -> int:
  enc = tiktoken.encoding_for_model("gpt-4o")
  return len(enc.encode(string))


class PaperFindings():
  def __init__(self, findings_dict):
    
    self.pmid = findings_dict['PMID']
    self.title = findings_dict['Title']
    self.domain = findings_dict['domain']
    self.metadata = {
      "Source": findings_dict["metadata"]["Source"],
      "PMC": findings_dict["metadata"]["PMC"],
      "Authors": findings_dict["metadata"]["Authors"],
      "PubDate": findings_dict["metadata"]["PubDate"],
      "Keywords": findings_dict["metadata"]["Keywords"],
      "MeSH": findings_dict["metadata"]["MeSH"],
      "Language": findings_dict["metadata"]["Language"],
      "Abstract": findings_dict["metadata"]["Abstract"],
    }
    self.ODH_score = {
      "score": findings_dict["ODH_score"]["score"],
      "rationale": findings_dict["ODH_score"]["rationale"],
    }
    self.OAH_score = {
      "score": findings_dict["OAH_score"]["score"],
      "rationale": findings_dict["OAH_score"]["rationale"],
    }
    self.questions = {}
    for q in findings_dict["questions"].keys():
        self.questions[q] = findings_dict["questions"][q]
    
    #self.predicates = findings_dict["predicates"]

    self.ODH_info = {
      "keys": findings_dict["ODH_info"]["keys"],
      "perspectives": findings_dict["ODH_info"]["perspectives"],
      "dimensions": findings_dict["ODH_info"]["dimensions"],
    }

    self.dump = json.dumps(findings_dict, indent=2)
    self.paper_str = self.prepare_text()
    self.token_count = num_tokens_from_string(self.paper_str)
    print(self.pmid, 'token count: ', self.token_count)


  def prepare_text(self) :
    text = '\n*****\n'
    text = text + 'PMID:\t' + self.pmid + '\n'
    text = text + 'Title:\t' + self.title + '\n'
    text = text + 'Keywords:\t' + self.metadata['Keywords'] + '\n'

    for q in ODH_eligibility_qs.keys():
      if q == 'ODH_keys':
        break
      text = text + q +':\t' + str(self.questions[q])
    #text = text + 'Predicates:\t' + self.predicates
    return text



# TBD LALALALAL
def load_findings_from_ds(papers):
  return []


def load_findings_from_file(filename):
    findings_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Handle potential JSONDecodeError for individual lines
            try:
                findings = json.loads(line.strip())
                paper_findings = PaperFindings(findings)
                findings_list.append(paper_findings)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}")
                print(e)  # Print the specific error for debugging
    return findings_list


# TBD add tools
def ask_biomed(papers: str, llm):

  prompt = f"""{OneAquaHealth}
  You are a biomed scientist and a domain expert in human health and wellbeing, researcher in OneAquaHealth.
  Below is a collection of scientific papers related to OneAquaHealth. Within your field of expertise, you are tasked with:
  1. Identify patterns, correlations and recurring themes between indicators, researches, and concepts in papers.
  2. Identify indicators that support the detection of disease vectors or environmental health proxies. 
  3. Identify early warning indicators for ecosystem degradation
  4. Generate potential hypotheses based on the text.
  Always provide reference from text - state paper pmid. Answer tasks in the given order.

  {papers}
  """

  biomed_prompt = PromptTemplate(
      input_variables=["papers"],
      template=prompt
  )

  biomed_chain = LLMChain(llm = llm, prompt=biomed_prompt)
  res = biomed_chain.run(papers=papers)
  #llm_output = llm.invoke(prompt)
  #print(llm_output.content)
  #print(res)
  #return llm_output.content
  return res


def consolidate_hypotheses(hypotheses, output_file, llm):

  print("consolidate_hypotheses. ", num_tokens_from_string(hypotheses))
  consolidation_prompt = PromptTemplate(
    input_variables=["hypotheses"],
    template=f"""
    {OneAquaHealth}
    Evaluating our scientific papers corpus, our domain expert identified: (task 1) Patterns, Correlations, and Recurring Themes, (task 2) Disease vectors or environmental health proxies, and (task 3) Early warning indicators for ecosystem degradation.
    They also prepared (task 4) Potential Hypotheses based on that corpus.
    We repeated this process a few times. The below text is the results of these runs.
    {hypotheses}

    Please consolidate these information by removing duplicates and merging similar ones into concise statements. 
    Always provide reference from text - state all supporting paper PMIDs.  
    """
  )

  consolidate_chain = LLMChain(
      llm=llm,
      prompt=consolidation_prompt
  )

  #hypotheses_text = "\n".join(hypotheses)
  consolidated_hypotheses = consolidate_chain.run(hypotheses=hypotheses)
    
  print(consolidated_hypotheses)
  return consolidated_hypotheses

def add_references(text, llm) -> str:
  print("add_references. ", num_tokens_from_string(text))
  ref_prompt = PromptTemplate(
    input_variables=["text"],
    template=f"""
    The following text contains multiple PMID references. Your task is to prepare a numbered, sorted PMID list.
    
    {text}
    """
  )

  ref_chain = LLMChain(
      llm=llm,
      prompt=ref_prompt
  )

  ref_list = ref_chain.run(text=text)
  
  print(ref_list)
  return ref_list

def initialize_cons_chain(llm, report):
  prompt = PromptTemplate(
    input_variables=["reports"],
    template=f"""
    {OneAquaHealth}
    You are building a consolidated report from batch of reports prepared by our domain experrts. 
    In each report OAH domain experts identified: (task 1) Patterns, Correlations, and Recurring Themes, (task 2) Disease vectors or environmental health proxies, and (task 3) Early warning indicators for ecosystem degradation.
    They also prepared (task 4) Potential Hypotheses based on that corpus. However, each scientist was only given a different subset of our scientific papers corpus.
    
    Please summarize the following reports into a single, concise and clear report. 
    ---
    {report}

    Always keep all references from reports - state all supporting paper PMIDs inline.  
    """
)
  chain = LLMChain(llm=llm, prompt=prompt)
  return chain


def consolidate_ideas(reports, output_file, llm) -> str:

  print("consolidate_ideas, we have report count ", len(reports))

  # check token count 


  # Init final report with the first report
  cons_rep = reports[0]
  cons_chain = initialize_cons_chain(llm=llm, current_report='', new_report=cons_rep)
  token_count = 0
  i=0
  first_entry_flag = 1
  with open(output_file, 'a') as out_file:
    for rep in reports:   
      #print(i, rep)   
      if first_entry_flag:
        first_entry_flag = 0
        continue
      
      cons_rep = cons_chain.run(current_report=cons_rep, new_report=rep)
      print(f"Iteration {i}: REPORT: {rep[:200]}")
      print(f"Iteration {i}: CONS REPORT: {cons_rep[:200]}")
      i += 1
      #print("consolidate_ideas rep #", i, cons_rep)
      if i ==5:  #debug
        break
      
      out_file.write('\n\n +*+*+*+*+*+*++*+*+*+*+*\n')
      out_file.write(cons_rep)
      
      token_count = num_tokens_from_string(cons_rep)
      print(token_count)
      if token_count >= 30000: #LALA DEBUG
        print("Hmmm we running short on context window.. ", token_count)
    #ref = add_references(cons_rep, llm)
  
  return cons_rep

# Step 1: generate a collection of ideas from a subset of the papers. 
# at each iteration we randomly select X papers from the corpus, and ask the domain expert to find themes, trends etc
def brainstorm_subset(corpus, iterations, subset_size, ideas_file, llm):

  text = ''
  views = []
  with open(ideas_file, 'a') as file:
    for iteration in range (iterations): 
      papers_numbers = random.sample(range(0,(len(corpus)-1)),subset_size)
      for num in papers_numbers:
        text = text + corpus[num].paper_str
      biomed_view = ask_biomed(text, llm)
      views.append(biomed_view)
      file.write(f'\n\nIteration:\t{iteration}\t Paper numbers:\t {papers_numbers}\n')
      file.write(biomed_view)
    print(iteration, "reply length:", num_tokens_from_string(biomed_view))
  return views
    

def extract_reports_from_folder(prefix):
    print("extract_reports_from_folder", ideas_folder_path)
    report_list = []
    
    for filename in os.listdir(ideas_folder_path):
      file_path = os.path.join(ideas_folder_path, filename)
      
      if os.path.isfile(file_path) and prefix in filename:
        print("Reading file name ", file_path)
        with open(file_path, 'r') as f:
            text = f.read().strip()
            # Debugging to check the content before split
            #print(f"Content of {filename}: {text[:200]}...")  # Print first 200 chars for inspection
            
            # Splitting and removing empty strings (in case there are unwanted splits)
            reports = [report.strip() for report in text.split('Iteration:') if report.strip()]
            print(f"{filename} has {len(reports)} reports.")
            
            report_list.extend(reports)

    print(f"Total reports: {len(report_list)}")
    
    if report_list:
        print(f"First report: {report_list[0][:200]}...")  # Print first 200 chars of the first report
        print(f"Last report: {report_list[-1][:200]}...")  # Print first 200 chars of the last report
    return report_list

# We ask a biomed (and an ecologist??) to generate hypothesys based on corpus. Random batches are sent 
# max tokens observed. After X iterations we go to autogen to brainstorm about all generated, validate
#provide support, and summarize all
# We load our corpus either from json file OR from PrismaReview
def brainstorm(generate_reports_flag, prisma_review, file_path, prefix, llm_name):

  llm = get_llm(llm_name, temperature = 0.8)

  corpus = []
  ideas = []
  
  print("BRAINSTORM")

  # Step 1: generate a collection of reports from subsets of the corpus. 
  # If flag True, we generate new reports. Otherwise, we load exisiting ones from path
  if generate_reports_flag == True:
    # Load from the DS
    if prisma_review != None:
      corpus = load_findings_from_ds(prisma_review.papers)
    else:
      corpus = load_findings_from_file(file_path)
    print(f"Corpus contains {len(corpus)} papers.")

    # at each iteration we randomly select X papers from the corpus, and ask the domain expert to find themes, trends etc
    ideas_file = ideas_folder_path + llm_name + '_Ideas_' + today() + '.txt'
    ideas = brainstorm_subset(corpus=corpus, iterations=10, subset_size=7, ideas_file=ideas_file, llm=llm)
  else:
    ideas = extract_reports_from_folder(prefix)

  # Step 2: compile all ideas together into one coherent list. 
  # We watch MAX_TOKENS
  # But not to worry - GPT-4o-mini - 128K; Gemini-flash - 1M.. 
  output_file = ouputs_folder_path  + llm_name + '_Outputs_' + today() + '.txt'

  ideas_str = ("\n").join(ideas)
  token_count = num_tokens_from_string(ideas_str)
  print(f"Token count of entire IDEAS: {token_count}")
  # TBD 
  if token_count < CONTEXT_WINDOW_MAX_TOKENS:
    print("WARNING our reports token count is ", token_count)
  
  with open(output_file, 'a') as out_file:
  
    cons_rep = consolidate_hypotheses(ideas_str, output_file, llm)
    out_file.write(cons_rep)
    ref = add_references(cons_rep, llm)
    out_file.write('\n\n')
    out_file.write(ref)
  
  return ref

#######################################################################
## pyautogen TBD
#######################################################################

PRISMA_biomed_message = f"""
You are a biomed scientist and a domain expert in human health and wellbeing. 
Your task is to identify key terms and indicators related to your domain from the provide text and suggest to the group.
Always provide reference from text. You can query the papers using your tool.
"""

PRISMA_environmental_message = f"""
You are a enviromental scientist and a domain expert specializing in urban aquatic ecosystems.
Your task is to identify key terms and indicators related to your domain from the provide text and suggest to the group.
Always provide reference from text. You can query the papers using your tool.
"""

PRISMA_data_analyst_message = f"""
You are a data analyst, proficient in extracting indicators, relations and trends from data.
Your task is to evaluate the responses from the domain expert, and structure it so we can create a knowledge graph from it.
Use this data and discover relationships between indicators, researches, and concepts.
Also, identify any indicators or relationships that may point to abnormal or alarming findings in the data.
You can query the papers using your tool.
"""

PRISMA_critic_message = f"""
You are a critic, known for your thoroughness and commitment to science.
Your task is to evaluate the output of each agent, double check and provide feedback.
Draw insights from your collegues' output and present your broad, holistic perspective.
You can use your tool to search online for support and references. 
"""
def autogen_group_chat():

  model_config = {
      "cache_seed": 783,  # change the cache_seed for different trials
      "temperature": 0,
      "config_list": get_autogen_llm('Gemini'),
      "timeout": 120,
  }

  user_proxy = autogen.UserProxyAgent(
      name="Admin",
      system_message="A human admin. Interact with the group to discuss the task. ",
      code_execution_config=False,
  )

  biomed = autogen.AssistantAgent(
      name="Biomed",
      llm_config=model_config,
      system_message=PRISMA_biomed_message,
  )

  ecologist = autogen.AssistantAgent(
      name="Ecologist",
      llm_config=model_config,
      system_message=PRISMA_environmental_message,
  )

  analyst = autogen.AssistantAgent(
      name="Analyst",
      system_message=PRISMA_data_analyst_message,
      llm_config=model_config,
  )

  critic = autogen.AssistantAgent(
      name="Critic",
      system_message=PRISMA_critic_message,
      llm_config=model_config,
  )
  groupchat = autogen.GroupChat(
      agents=[user_proxy, biomed, ecologist, analyst, critic], messages=[], max_round=20
  )
  manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=model_config)

    
  message = f"""{OneAquaHealth}. The following  is a collection of scientific papers related to OneAquaHealth.
  1. Within your field of expertise, identify patterns, correlations and recurring themes from the text.
  2. Generate potential hypotheses based on the text.

  We are assessing the novelty and relevance of the following papers to {OneAquaHealth}
  """

  user_proxy.initiate_chat(manager, message=message,)


