🤖 Simple RAG Agent Project
This is a minimal repository demonstrating how to build a Retrieval-Augmented Generation (RAG) system using: - Smolagents (for agent orchestration) - LlamaIndex (for document indexing & querying) - Langfuse (optional) for tracing & observability
It includes a Jupyter notebook and a simple Python script showing how to: 1. Load and split documents. 2. Create vector + summary indexes. 3. Expose them as a tool (RagTool). 4. Run a CodeAgent with tool usage.

________________________________________
⚙️ Requirements
Install the minimal dependencies:
smolagents
llama-index
Optional:langfuse
openinference-instrumentation-smolagents
Install them with:
pip install -r requirements.txt
________________________________________
🔑 Environment Variables
Create a .env file. Example:
KEY=your-openai-api-key,
LANGFUSE_SECRET_KEY=your-langfuse-secret,
LANGFUSE_PUBLIC_KEY=your-langfuse-public,
LANGFUSE_HOST=https://us.cloud.langfuse.com,
GKey=your-gemini-api-key,
________________________________________
▶️ Usage
Run the notebook
Open Notebook_Test.ipynb in Jupyter and execute step by step.
Run the script: run agent_script.py in your vs code or any ide

Example query inside the agent:
agent.run("How to create material in Blender using Python? Use the RAG tool.")
Or directly use the tool:
print(rag_tool("Explain how to apply materials in Blender"))
________________________________________
🛡️ Notes
•	Minimal dependencies are included, Langfuse is optional.
•	Works with OpenAI/Gemini models (configurable in .env).
•	You can extend by adding more tools or retrievers.
________________________________________
📜 License
MIT License — free to use, modify, and share.
