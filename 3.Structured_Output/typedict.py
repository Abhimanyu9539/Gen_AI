from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

class Person(TypedDict):
    name: str
    age: int

new_person : Person = {'name':'abhi', 'age': 29}
print(new_person)

