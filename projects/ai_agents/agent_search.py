""" this is the file that contains the code for the OPENAI-based chatbot """
from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage

# load the environment variable with the API key
load_dotenv()

# set up external search tool
search = TavilySearchResults(max_results=5)

# create the list of model tools
tools_list = [search]

# choose the model to use
model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools_list)

# print(model.invoke("What is the capital of New Mexico, USA?"))

def call_model(state: MessagesState):
    updated_messages = model.invoke(state['messages'])
    return {'messages': updated_messages}

call_tool = ToolNode(tools_list)

# set up mode memory
workflow = StateGraph(MessagesState)

workflow.add_node('model_node', call_model)
workflow.add_node("tools", call_tool)
workflow.add_edge(START, 'model_node')
workflow.add_conditional_edges("model_node", tools_condition)
workflow.add_edge("tools", "model_node")

# specify a memory checkpointer (this example is in-memory but can use a DB)
memory = MemorySaver()

# now build an invokable model
app = workflow.compile(memory)

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
                if isinstance(chunk, AIMessage):
                    print(chunk.content, end="", flush=True)
                if isinstance(chunk, ToolMessage):
                    result_list = json.loads(chunk.content)
                    urls = [result["url"] for result in result_list]
                    print(urls, end="\n\n")
            print("\n")

def examine_chatbot(chat_id: int):
    config = {'configurable': {'thread_id': chat_id}}

    while True:
        user_input = input('User:')

        if user_input in ['exit', 'quit']:
            print('AI: See you later!')
            break

        else:
            for state in app.stream({'messages': user_input}, config, stream_mode='values'):
                state["messages"][-1].pretty_print()
            print("\n")

examine_chatbot(6)