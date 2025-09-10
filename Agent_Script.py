#%% initial imports
from smolagents import CodeAgent, LiteLLMModel
from smolagents.default_tools import DuckDuckGoSearchTool
from langfuse import Langfuse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv
from smolagents.tools import BaseTool

#%% Api Key Setup
load_dotenv("enter your env file path")
ApiKey = os.getenv('KEY')
Skey = os.getenv("LANGFUSE_SECRET_KEY")
pKey = os.getenv("LANGFUSE_PUBLIC_KEY")
Host = os.getenv("LANGFUSE_HOST")
Gkey = os.getenv("GKey")

# Validate API keys
if not ApiKey:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

#%% Langfuse Setup 
if Skey and pKey and Host:
    langfuse = Langfuse(
        secret_key=Skey,
        public_key=pKey,
        host=Host
    )
    
    if langfuse.auth_check():
        print("Connected to Langfuse")
    else:
        print("Not connected to Langfuse")
else:
    print("Langfuse credentials not found, skipping setup")
    langfuse = None

#%% Instrumentor (optional - only if you want tracing)
try:
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    SmolagentsInstrumentor().instrument()
    print("SmolagentsInstrumentor enabled")
except ImportError:
    print("SmolagentsInstrumentor not available, skipping instrumentation")

#%% RAG Setup 
# Check if files exist, Enter your files
file_paths = [
    r"D:\PROGRAMMING\rag D1.txt",
    r"D:\PROGRAMMING\rag D2.txt",
    r"D:\PROGRAMMING\rag_D3.txt"
]

existing_files = [f for f in file_paths if os.path.exists(f)]
if not existing_files:
    raise FileNotFoundError(f"None of the specified files exist: {file_paths}")

reader = SimpleDirectoryReader(input_files=existing_files)
documents = reader.load_data()
print(f"{len(documents)} document(s) loaded")

# Node parsing
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} nodes")

# Settings configuration - use gpt-4o-mini instead of gpt-5-mini
Settings.llm = OpenAI(model="gpt-5-mini", api_key=ApiKey)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=ApiKey)

# Create indexes
vector_index = VectorStoreIndex(nodes)
summary_index = SummaryIndex(nodes)

# Create query engines
vector_query_engine = vector_index.as_query_engine()
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# Create tools for router
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from the files"
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to the files"
)

# Create router query engine
query_engine = RouterQueryEngine.from_defaults(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool],
    verbose=True
)

# Custom RAG Tool for Smolagents
class RagTool(BaseTool):
    name = "rag_engine"
    description = "Answers questions based on the retrieved data from uploaded documents"
    
    def __init__(self, query_engine):
        super().__init__()
        self.query_engine = query_engine
    
    def __call__(self, input_text: str) -> str:
        """Query the RAG system and return response as string"""
        try:
            response = self.query_engine.query(input_text)
            return str(response)
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"
    
    def to_code_prompt(self) -> str:
        """Generate a code prompt for this tool"""
        return f"""
Tool: {self.name}
Description: {self.description}
Usage: {self.name}("your query here")
Returns: String response with retrieved information from the documents
"""

# Create RAG tool instance
rag_tool = RagTool(query_engine)

#%% Model Setup
# Use proper model names
try:
    vision_model = LiteLLMModel(
        model_id="openrouter/google/gemini-2.0-flash-exp",  # Next version will include this model
        api_key=Gkey
    ) if Gkey else None
except Exception as e:
    print(f"Vision model setup failed: {e}")
    vision_model = None

text_model = LiteLLMModel(
    model_id="gpt-5-mini",  # Use correct model name
    api_key=ApiKey
)

#%% Agent Setup
try:
    # Create tools list
    tools = [rag_tool]
    
    agent = CodeAgent(
        tools=tools,
        model=text_model,
        planning_interval=2,
        max_steps=10  # Add max steps to prevent infinite loops
    )
    print("Agent created successfully")
    
except Exception as e:
    print(f"Error creating agent: {e}")
    raise

#%% Agent Query
if __name__ == "__main__":

    query = "Enter Your Query" #Enter Your Query here

    try:
        print("Running agent query...")
        agent_response = agent.run(query )
        print("Agent Response:")
        print(agent_response)
        
    except Exception as e:
        print(f"Error during agent execution: {e}")
        
        # Fallback: try direct RAG query
        print("\nTrying direct RAG query as fallback...")
        try:
            direct_response = rag_tool(query)
            print("Direct RAG Response:")
            print(direct_response)
        except Exception as e2:
            print(f"Direct RAG query also failed: {e2}")
