from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch


load_dotenv()


model1 = ChatOpenAI(model = "gpt-4o")
model2 = ChatOpenAI(model = "o1")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Write a detailed about {topic}.",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "Summarize the following text. \n {text}",
    input_variables= ["text"]
)

report_generation_chain = RunnableSequence(
    prompt1, model1, parser
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model2, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

result = final_chain.invoke({"topic":"AI VS Human"})

print(result)

final_chain.get_graph().print_ascii()