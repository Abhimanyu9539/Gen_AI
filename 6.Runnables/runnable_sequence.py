from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence


load_dotenv()

model = ChatOpenAI(model = "gpt-4o")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Write a mail about {topic}.",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "Explain the following. \n {mail}",
    input_variables= ["mail"]
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic":"AI"})

print(result)

chain.get_graph().print_ascii()
