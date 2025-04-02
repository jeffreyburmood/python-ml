""" this is the file that contains the code for the OPENAI-based chatbot """
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver

# load the environment variable with the API key
load_dotenv()

# choose the model to use
model = ChatOpenAI(model='gpt-4o-mini')

# print(model.invoke("What is the capital of New Mexico, USA?"))

# set up mode memory
workflow = StateGraph(MessagesState)

def call_model(state: MessagesState):
    updated_messages = model.invoke(state['messages'])
    return {'messages': updated_messages}

workflow.add_node('model_node', call_model)
workflow.add_edge(START, 'model_node')

# specify a memory checkpointer (this example is in-memory but can use a DB)
memory = MemorySaver()

# now build an invokable model
app = workflow.compile(memory)

# set up a conversation thread
chat1 = {'configurable': {'thread_id': 1}} # thread_id can be any integer or string
print(app.invoke({'messages': 'My favorite state is New York'}, chat1))

# now add another message to the thread
print(app.invoke({'messages': 'What is my favorite state?'}, chat1))

# make output more readable
output = app.invoke(None, chat1) # no new message is added to the conversation thread
print([message.pretty_print() for message in output['messages']])

# supporting multiple independent conversation threads
chat2 = {'configurable': {'thread_id': 2}}
# select just the last message response
print(app.invoke({'messages': 'What is my favorite state?'}, chat2)['messages'][-1].content)

# now build the chatbot function
def chatbot(chat_id: int):
    config = {'configurable': {'thread_id': chat_id}}

    while True:
        user_input = input('User:')

        if user_input in ['exit', 'quit']:
            print('AI: See you later!')
            break

        else:
            chatbot_output = app.invoke({'messages': user_input}, config)
            print('AI:', chatbot_output['messages'][-1].content, end='\n\n')
