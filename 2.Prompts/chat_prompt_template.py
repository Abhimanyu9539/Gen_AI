from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert."),
    ('hcuman', "Explain in simple terms what is {topic}")
])


prompt = chat_template.invoke({'domain':'cricket', 'topic':'googly'})

print(prompt)