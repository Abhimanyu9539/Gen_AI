from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatOpenAI(model = "gpt-4o")

model2 = ChatOpenAI(model = "o1-mini")

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal["Postive", "Negative"] = Field(description= "Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template= "Classify the sentiment of following feedback as positive or negative. \n {feedback} \n {format_instruction}",
    input_variables= ["feedback"],
    partial_variables= {'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model2 | parser2

#print(classifier_chain.invoke({"feedback":"This is a terrifying smartphone"}))

prompt2 = PromptTemplate(
    template= "Write an appropriate resposne for this positive feedback \n {feedback}" ,
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template= "Write an appropriate response for this negative feedback \n {feedback}" ,
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == "Positive", prompt2 | model1 | parser),
    (lambda x : x.sentiment == "Negative", prompt3 | model1 | parser),
    RunnableLambda(lambda x : "Could not find sentiment.")
)


chain = classifier_chain | branch_chain

result = chain.invoke({"feedback":"This is a terrible phone"})

print(result)

chain.get_graph().print_ascii()
