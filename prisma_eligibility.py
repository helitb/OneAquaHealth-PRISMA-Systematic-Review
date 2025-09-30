# prisma_eligibility.py
# full text assessment based on llama-index engine queries

import time
#from annotated_types import IsDigit
import tiktoken
from google.colab import userdata

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from ODH_definitions import OneDigitalHealth, OneAquaHealth, OneAquaHealth_indicators

import nest_asyncio

nest_asyncio.apply()

###########################################
## Here we define the list of questions 
## we ask each paper in order to decide 
## if it is eligible for the PRISMA review
## We first confirm all general requirements 
## for paper inclusion, then we categorize the paper into:
## 1. Human health or wellbeing indicators (WP3)
## 2. Ecological or biological health indicators (WP2)
## 3. Citizens engagement (WP4)
## And ask specific questions for each branch.
## The replies are used to decide if we need 
## to extract full findings from paper (open ended questions for llm)
###########################################

# Route to followup questions based on domain, or reject paper based on these Qs.
gen_eligibility_router = {
  'Q1': "Does this paper focus on impact on human health or well-being (e.g., quality of life, mental health)? Reply yes or no",
  'Q2': "Does this paper focus on impact on environmental health (ecology, animals, plants)? Reply yes or no",
  'Q3': "Does this paper focus on impact on species other than human? Reply yes or no",
  'Q4': f"""Blue spaces are dominated by a watery element, such as a lakeside, river or coast.
  Green spaces may include a watery element but are characterised by predominantly 'green' elements such as grass or trees.
  Does this paper focus on blue spaces or green spaces? Reply green/blue/other""",
  'Q5': f"""{OneDigitalHealth}. In a scale of 1 to 10, how relevant is this paper to OneDigitalHealth? Explain your reply and end it with final score in this format: <<score>>""",
  'Q6': f"""{OneAquaHealth}. In a scale of 1 to 10, how relevant is this paper to OneAquaHealth? Explain your reply and end it with final score in this format: <<score>>""",
}

'''
gen_eligibility_router = {
	'Q3': "Does this paper investigate the impact on ecosystems or biodiversity? Reply yes/no",    	
  'Q5': "Does this paper involve citizen science, public participation, policy implications, or education? Reply yes/no. If yes, which one? ",
	'Q7': "Does this paper deal with Natural Environments (Forests, Grasslands, Deserts, Mountains, Polar Regions, Coastal Areas, Islands), Built Environments (Urban Areas, Suburban Areas, Rural Areas), Transitional Environments (Agricultural Lands, Industrial Areas)? Reply Natural Environments (Forests, Grasslands, Deserts, Mountains, Polar Regions, Coastal Areas, Islands)/Built Environments (Urban Areas, Suburban Areas, Rural Areas)/Transitional Environments (Agricultural Lands, Industrial Areas)/A combination of a few environments
}

ODH_eligibility_qs = {
	'Q11': "Considering all the previous questions, classify this paper in one or more of the following classes 'Ecological Health', 'Human Health', 'Citizen Engagement', 'Digital Technologies'. If this paper can be assign to more than one class give for each class the probability to be related to said class considering that the sum of said probabilities must equal 1",
	'Q12_bonus': "If there are some indicator-related values, list them by location, species, dates and more when relevant"
    }
 

'''

ODH_eligibility_qs = {
    'paper_aim': "What is the aim of the paper? Reply in  plain text",
    'OAH_indicators': f"""{OneAquaHealth_indicators}. What indicators are mentioned in the paper? How were they measured? Reply in  plain text""",
    'geolocation': "What is the geographical location the paper focuses on? Include continent in your reply. Reply in plain text",
    'methodology': "What methodology/design was used in the paper? Reply in  plain text",
    'key_findings': "What are the key findings in the paper? Reply in  plain text",
    'population': "What is the studied population's characteristics? number, species, age etc. Reply in  plain text",
    'relations': "Express the key findings of the paper in terms of the relationship between the indicators that are discussed in the paper. Reply in  plain text",
    'ODH_keys': f"""{OneDigitalHealth}. What OneDigitalHealth keys are addressed in this paper? Place the keys between double stars in your reply""",
    'ODH_perspectives': f"""{OneDigitalHealth}. What OneDigitalHealth perspectives are addressed in this paper? Place the perspectives between double stars in your reply""",
    'ODH_dimensions': f"""{OneDigitalHealth}. What OneDigitalHealth dimensions are addressed in this paper? Place the dimensions between double stars in your reply""",
    }


# get_router_query_engine: create a querry engine for one paper
#   Args:
#   file_path (str): Path to the document.
#   model_type (str, optional): The LLM and embedding model to use. Defaults to 'Gemini'.
#
# Source code:
# https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/2/router-query-engine

## Get router query engine for file
def get_router_query_engine(file_path: str, llm, embedings):

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    #print(f" hello got llm {llm.model_name}")
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    #print(f" heelo {len(nodes)}")
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embedings)
    
    #print(f" created indexed")
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    
    #print(f" created summary_query_engine")
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    #print(f" created vector_query_engine")
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to paper"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the paper."
        ),
    )
    print(f" created query engines")
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine


'''
def get_llamahub_llm(llm_model: str):

  Settings.llm = get_langchain_llm(llm_model)
  return Settings.llm

  if llm_model == 'OpenAI':
    print("Loading OpenAI")
    openai_api_key = userdata.get('OPENAI_KEY')

    Settings.llm = OpenAI(model="gpt-4o-2024-08-06", api_key=openai_api_key)
  elif llm_model == 'Gemini':
    print("Loading Gemini")
    google_api_key=userdata.get('GOOGLE_KEY')
    Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
  else:
      raise ValueError("Invalid model type. Choose 'OpenAI' or 'Gemini'.")
  return Settings.llm
  '''

## This class holds all the information extracted from full text screening of one paper.
## It creates a query_engine, classifies the paper to WP2/3/(4?)/Reject, based on that 
## classification it runs through another queation and answer chain, then reduces the output
## into a single summary  
## llm here must be initialialized via llama-index api, not langchain. we pass the llm_str and init here
class EligibilityInfo():  
  def __init__(self, pmid: str, title: str, embed_model, llm_model):
    self.path = ''
    self.chain = None
    self.PMID = pmid
    self.title = title

    self.query_engine = None
    self.embed_model = embed_model
    self.model = llm_model
    Settings.llm = llm_model

    self.token_count = 0

    self.domain = 'TBD'
    self.ODH_score = 0
    self.ODH_score_rational = ''
    self.OAH_score = 0
    self.OAH_score_rational = ''
    self.llm_replies = {}
    self.summary = ''
    
  def init_query_engine(self, path: str):
    self.path = path
    self.query_engine = get_router_query_engine(self.path, self.model, self.embed_model)

  def get_score(self, text: str):

    score = text.split('<<')[1]
    score = score.split('>>')[0]

    print(score)
    for word in score.split():
      if word.isdigit():
        return int(word)

    return -1

  ## We analyze the responses to gen_eligibility_router questions and make a decision
  def classify_paper(self):
    responses = {}
    for q in gen_eligibility_router.keys():
      responses[q] = self.query_engine.query(gen_eligibility_router[q])
      # Gemini 1.5 call limit TBD
      time.sleep(1)

      #print(gen_eligibility_qs[q])
      print(f"\t{q}: {str(responses[q])}\n")

    # If OAH and ODH scale is above 7, we categorize and include paper. 
    self.ODH_score_rational = str(responses['Q5'])
    self.OAH_score_rational = str(responses['Q6'])

    self.ODH_score = int(self.get_score(str(responses['Q5'])))
    self.OAH_score = int(self.get_score(str(responses['Q6'])))

    if (self.OAH_score > 6):
      if 'yes' in str(responses['Q1']).lower() and 'other' not in str(responses['Q4']).lower():
        return 'Human Health'
      elif 'yes' in str(responses['Q2']).lower() and 'other' not in str(responses['Q4']).lower():
        return 'Environment'
      elif 'yes' in str(responses['Q3']).lower() and 'other' not in str(responses['Q4']).lower():
        return 'Animal Health'
      return "TBD"

    return 'N/A' 

  """TEMP Returns the number of tokens in a text string."""
  def num_tokens_from_string(self, string) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(string))

  def prepare_text(self):
    text = ''
    text = text + 'ODH Score: ' + self.ODH_score_rational + '\n'
    text = text + 'OAH Score: ' + self.OAH_score_rational + '\n'

    for q in ODH_eligibility_qs.keys():
      text = text + q +':\t' + str(self.llm_replies[q]) + '\n'
    
    self.token_count = self.num_tokens_from_string(text)
    print(f"Tokens: {self.token_count}\n Text: \n {text}") 
    return text


  # This method does two things:
  # 1. It asks the paper questions based on the relevant domain Q&A pipline 
  # 2. It runs the replies through langchain map reduce pipline NOT HERE IN DS
  def analyze_paper(self):
    for q in ODH_eligibility_qs.keys():
      self.llm_replies[q] = self.query_engine.query(ODH_eligibility_qs[q])
      time.sleep(1)
      print(f"\t{q}: {str(self.llm_replies[q])}\n")
    
    self.summary = self.prepare_text()
  
