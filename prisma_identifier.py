# prisma_identifier.py

import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import openai
from google.colab import userdata
import json

#from utils import get_langc_llm, search_pubmed

def get_queries() -> str:
  queries = [
    'Urban Aquatic Engagement',
  ]
  '''
    'Urban Freshwater Environments',
    'Human Health AND Urban Aquatic Ecosystems',
    '\'Citizen Engagement\' AND Urban Aquatic Ecosystems',
    'Public participation AND Urban Aquatic Ecosystems',
  ]'''
  return queries


##################################################




