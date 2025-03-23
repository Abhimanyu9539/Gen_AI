from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough


load_dotenv()


model1 = ChatOpenAI(model = "gpt-4o")
model2 = ChatOpenAI(model = "o1")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Write a joke about {topic}.",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "Explain the following joke. \n {joke}",
    input_variables= ["joke"]
)

joke_gen_chain = RunnableSequence(prompt1, model1, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(), 
    "explanation" : RunnableSequence(prompt2, model2, parser)
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

result = final_chain.invoke({"topic":"AI"})

print(result)

final_chain.get_graph().print_ascii()


