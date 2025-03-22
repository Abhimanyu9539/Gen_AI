from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model = "gpt-4o")

model2 = ChatOpenAI(model = "o1-mini")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Generate short and simple on the {text}", 
    input_variables= ["text"]
)

prompt2 = PromptTemplate(
    template= "Generate 5 short question and answers from the following text. \n {text}",
    input_variables= ["text"]
)

prompt3 = PromptTemplate(
    template= "Merge the provided notes and quiz into a single document . \n notes --> {notes} \n quiz--> {quiz}",
    input_variables= ["notes","quiz"]
)


parallel_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser, 
    "quiz" : prompt2 | model2 | parser
})

merged_chain = prompt3 | model2 | parser 

chain = parallel_chain | merged_chain

text = """
Gen AI, or generative artificial intelligence, is a type of AI that can create new content like images, videos, and music. It can also learn human language, programming languages, and other complex subjects. 
How does Gen AI work?
Uses machine learning techniques to learn from and create new data 
Reuses what it knows to solve new problems 
Can be trained to generate new content based on natural language input 
Examples of Gen AI applications: 
Chatbots, Media creation, Product development, Design, Summarization, Q&A, and Classification.
Benefits of Gen AI 
Can help people analyze information and insights at work
Can help employees and leaders make better decisions
Can help companies streamline processes
Can help companies personalize offerings
Potential drawbacks of Gen AI Can lead to the growth of data sweatshops, Can strain electric grid infrastructure, Can create new security threats, and Can create new social engineering attacks. 
"""


result = chain.invoke({"text": text})

#print(result)

chain.get_graph().print_ascii()