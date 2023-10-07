import os
import openai
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# llm to get the name of a ecommerce store from a product name

prompt = PromptTemplate.from_template(
    "What is the name of the e-commerce store that sells {product}")

llm = OpenAI(temperature=0.3,
             openai_api_key=os.environ.get("OPENAI_API_KEY"))
chain1 = LLMChain(llm=llm, prompt=prompt)

# product = "iphone";
# output = chain.run(product);
# print(output);

# LLM to get comma seperated names of products from an e-commerce store name
prompt = PromptTemplate.from_template(
    "What are the names of the products at {store}")
llm = OpenAI(temperature=0.3,
             openai_api_key=os.environ.get("OPENAI_API_KEY"))
chain2 = LLMChain(llm=llm, prompt=prompt)
# store = "amazon"
# output = chain.run(store);
# print(output);

# create a overall chain from a simple sequential chain
# by setting verbose to true , it will notify us about what it is doing in terminal , like giving message about store name, and products
chain = SimpleSequentialChain(
    chains=[chain1, chain2], verbose=True
)

# chain.run("candles")

# without verbose
# chain = SimpleSequentialChain(
#     chains=[chain1, chain2]
# )
# output = chain.run("candles");
# print(output)


# an example of sequential chain
llm = OpenAI(temperature=0.7,
             openai_api_key=os.environ.get("OPENAI_API_KEY"))
template = """you are a playwright, given the title of play and the era it is set in,
it is your job to write a synopsis for that title,

Title : {title}
Era : {era}
Playwright : This is a synopsis for the above play : 
"""
prompt_template = PromptTemplate(input_variables=['title', 'era'], template=template);
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis");

# this is a an LLMChain to write a review of a play given a synopsis
llm = OpenAI(temperature=0.7, openai_api_key=os.environ.get("OPENAI_API_KEY"));
template="""you are a play critic from New York Times, Given the sysnopsis of play,
it is your job to review for that play

Play Synopsis : {synopsis}
Review from a New York Times play critic of the above play : 
"""

prompt_template = PromptTemplate(input_variables=["synopsis"], template=template);
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review");

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era","title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True
)

# print(overall_chain({
#     "era" : "Renaissance", "title" : "The Tempest"
# }))

# Agent Demo
llm = OpenAI(temperature=0.7)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
output = agent.run("When was Battle of Malazgirt fought?")
print(output);