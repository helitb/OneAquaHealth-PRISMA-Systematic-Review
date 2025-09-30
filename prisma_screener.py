# prisma_screener.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import openai
from google.colab import userdata
import json
import tiktoken
from sentence_transformers.SentenceTransformer import Dict

#from utils import get_llm, search_arxiv, search_pubmed
#from prisma_ds import ScreenInfo

from utils import *

class ScreenInfo():
  def __init__(self):
    self.I1 = False
    self.I2 = False
    self.I3 = False
    self.I4 = False
    self.I5 = False
    self.status = 'Rejected'
  
  def print_info(self):
    print(f"I1: {self.I1}\nI2: {self.I2}\nI3: {self.I3}\nI4: {self.I4}\nI5: {self.I5}\nStatus: {self.status}\n")

  def get_info(self) -> str:
    info = f"I1: {self.I1}\nI2: {self.I2}\nI3: {self.I3}\nI4: {self.I4}\nI5: {self.I5}\nStatus: {self.status}\n"
    #print(info)
    return info

  def get_status(self) -> str:
    return self.status






screener_message = """Your task is to assess the relevance of the following paper to PRISMA review about Urban Freshwater Ecosystems.
If the paper does not meet the inclusion criteria, reject it.
The record inclusion criteria are as follows:
I1. Must be in English. Reply true or false.
I2. Publication date is after January 2013 and before May 2024. Reply true or false.
I3. Source includes doi, OR PMC information included 
I4. Check papers' keywords and MeSH terms before replying. Focus on Human Health Indicators in Urban Freshwater Environments. Reply true or false.
I5. Check papers' keywords and MeSH terms before replying. Focus on Ecological Indicators in Urban Freshwater Environments. Reply true or false.
Reply with a dictionary:
"I1": true or false
"I2": true or false
"I3": true or false
"I4": true or false
"I5": true or false
"Status": "Screened" if (I1, I2, I3) is true AND (I4 or I5) is true, or "Rejected" otherwise. 

{paper}
"""

# Define a prompt template for identifying papers
screener_prompt = PromptTemplate(
    input_variables=["paper"],
    template=screener_message
)


# Define the LLMChain with the LLM and the prompt template
#screener_chain = LLMChain(
#    llm=llm,
#    prompt=screener_prompt
#)

class PrismaScreener:
    def __init__(self, llm):      
      self.chain = LLMChain(llm=llm, prompt=screener_prompt)
      self.paper = ''

    def update_paper(self, paper):
      self.paper = paper

    def update_info(self, str_info: str) -> ScreenInfo:
      info = ScreenInfo()
      for line in str_info.splitlines():
        if "I1" in line:
          if "true" in line.lower():
            info.I1 = True
        if "I2" in line:
          if "true" in line.lower():
            info.I2 = True
        if "I3" in line:
          if "true" in line.lower():
            info.I3 = True
        if "I4" in line:
          if "true" in line.lower():
            info.I4 = True
        if "I5" in line:
          if "true" in line.lower():
            info.I5 = True
        if "Status" in line:
          if "Screened" in line:
            info.status = "Screened"

      info.print_info()  
      return info

    def num_tokens_from_string(self, string) -> int:
      enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
      return len(enc.encode(string))

    # method updates new paper, runs screening chain, updates paper info, 
    def screen_paper(self, paper) -> ScreenInfo:
      #print ('Hello ', paper.content.pmid)
      self.update_paper(str(paper))
      response = self.chain.invoke({"paper": self.paper})
      token_count = self.num_tokens_from_string(response['paper'])
      print(token_count)
      info = self.update_info(response['text'])
      return info, token_count


##################################################

##################################################




