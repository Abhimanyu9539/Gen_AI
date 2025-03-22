from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt 
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables= ['topic']
)

# 2nd Prompt
template2 = PromptTemplate(
    template="Write a 5 line summar on following text. /n {text}",
    input_variables= ['text']
)

prompt1 = template1.invoke({'topic':'blackholes'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result = model.invoke(prompt2)

print(result.content)