import openai
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

# defining our LLM
# you can set the temperature , if u set it to 1 then our llm will generate random things
# but if you want the same things repeatedly then choose 0.3-0.5 or something like this
llm = OpenAI(temperature=0.3, openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate.from_template("what is the capital of {place}")
chain = LLMChain(llm=llm, prompt=prompt)

# suppose we have a list of cities and we have to know the capital of those cities , what can we do to do that
cities = ['bangladesh', 'india', 'japan', 'pakistan']

for city in cities:
    output = chain.run(city)
    print(output)
    import time
    time.sleep(2)