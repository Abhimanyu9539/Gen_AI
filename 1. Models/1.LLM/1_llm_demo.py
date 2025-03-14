from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model = "gpt-3.5-turbo-instruct")

result = llm.invoke("What is Capital on India?")

print(result)


