""" this is the file that contains the code for the OPENAI-based chatbot """
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load the environment variable with the API key
load_dotenv()

# choose the model to use
model = ChatOpenAI(model='gpt-4o-mini')

print(model.invoke("What is the capital of New Mexico, USA?"))