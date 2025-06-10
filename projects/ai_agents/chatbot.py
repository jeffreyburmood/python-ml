""" this is the file that contains the code for the OPENAI-based chatbot """
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

# Build the workflow ***************************
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

# build a prompt template using langchain System messagess
prompt = ChatPromptTemplate(
    [
        ("system", "Limit all of your responses to two sentences."),
        ("placeholder", "{messages}")
    ]
)

# first question to the model
state = {"messages": ["What is the history of the United States?"]}


# now build the chatbot function - streaming version
def chatbot(chat_id: int):
    config = {'configurable': {'thread_id': chat_id}}

    while True:
        user_input = input('User:')

        if user_input in ['exit', 'quit']:
            print('AI: See you later!')
            break

        else:
            print('AI: ', end='')
            for chunk, metadata in app.stream({'messages': user_input}, config, stream_mode='messages'):
                print(chunk.content, end='', flush=True)
            print('\n')


# chatbot(1)

# print(prompt.invoke(state))

print(model.invoke(prompt.invoke(state)))

