""" this is the file that contains the code for the OPENAI-based chatbot """
from dotenv import load_dotenv
import json

from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the environment variable with the API key
load_dotenv()

# set up document loader for RAG
loader = WebBaseLoader("https://docs.python.org/3/whatsnew/3.13.html")
docs = loader.load()
# build the text chunks from the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
# add the embedding model and the vector store
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore.from_documents(chunks, embedding)
# now add the vector store retriever and associated retriever tool
retriever = vector_store.as_retriever()
python_tool = create_retriever_tool(retriever, "python_3_13_update_retriever",
            "A retriever that returns relavant documents from the What's New page of the Python 3.13 documentation" 
                        "Useful when you need to answer questions about the features, enhancements, removals, deprecations and other changes introduced in this version of python"
        )
# set up external search tool
search = TavilySearchResults(max_results=5)

# create the list of model tools
tools_list = [python_tool, search]

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
                    if chunk.name == "tavily_search_results_json":
                        result_list = json.loads(chunk.content)
                        urls = [result["url"] for result in result_list]
                        print(urls, end="\n\n")
                    if chunk.name == "python_3_13_update_retriever":
                        print("Checking Python 3.13 documentation...", end="\n\n")
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

chatbot(7)