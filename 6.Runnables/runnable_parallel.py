from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence


load_dotenv()

model1 = ChatOpenAI(model = "gpt-4o")
model2 = ChatOpenAI(model = "o1")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Write a tweet about {topic}.",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "Write a linkedin post about {topic}.",
    input_variables= ["topic"]
)

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model1, parser),
    "linkedin": RunnableSequence(prompt2, model2, parser)
})

result = parallel_chain.invoke({'topic':"AI"})
print(result)

parallel_chain.get_graph().print_ascii()
