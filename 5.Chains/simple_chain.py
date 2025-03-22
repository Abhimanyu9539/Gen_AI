from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template= "Generate 5 ineteresting facts about {topic}",
    input_variables= ["topic"]
)

model = ChatOpenAI(model = "gpt-4o")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "blackhole"})

print(result)

chain.get_graph().print_ascii()

