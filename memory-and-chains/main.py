import os
import openai
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
# from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

# Memory in LLMs
llm = OpenAI(temperature=0.3,
             openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate.from_template("What is the name of the e commerce store that sells {product}")
chain1 = LLMChain(llm=llm,prompt=prompt, memory=ConversationBufferMemory())
output = chain1.run("sneaker shoes");

import time
print("Loading in 10 seconds");
time.sleep(10)
print(chain1.memory)
print(chain1.memory.buffer);
print(output);