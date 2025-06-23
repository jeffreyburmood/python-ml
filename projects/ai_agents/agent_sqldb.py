""" this is the file that contains the code for the OPENAI-based chatbot """
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

# load the environment variable with the API key
load_dotenv()

# this allows connection to the database to be used for LLM queries
database_file_path = '../../data/Chinook.sqlite'
db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
# get the tool list from the toolkit
sql_model = ChatOpenAI(model="gpt-4o-mini")
toolkit = SQLDatabaseToolkit(db=db, llm=sql_model)
tools_list = toolkit.get_tools()

# choose the model to use
model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools_list)

# print(model.invoke("What is the capital of New Mexico, USA?"))
system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most 5 results. You can order the results by a relevant column to
return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.
You have access to tools for interacting with the database.
You must double check your query before executing it. If you encounter and error while executing a query, rewrite the 
query and try again.
Do not make any DML statement (INSERT, UPDATE, DELETE, DROP, etc) to the database.
To start you should always looks at the tables in the database to see what you can query.
Then you should query the schema of the most relevant tables.
"""

prompt = ChatPromptTemplate(
    [("system", system_message),
     ("placeholder", "{messages}")]
)

def call_model(state: MessagesState):
    chain = prompt | model
    updated_messages = chain.invoke(state)
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

#examine_chatbot(12)

chatbot(13)

# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))
