from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model = "gpt-4o")

prompt1 = PromptTemplate(
    template= "Generate a detailed report on {topic}",
    input_variables= ["topic"]
)


prompt2 = PromptTemplate(
    template= "Generate a 5 point summary on the following text. \n {text}",
    input_variables= ["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({"topic":"cricket"})

print(result)

chain.get_graph().print_ascii()