from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
import boto3
import json
from langchain.prompts import PromptTemplate
from QASystem.ingestion import get_vector_store
from QASystem.ingestion import data_ingestion
from langchain_community.embeddings import BedrockEmbeddings

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""


prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Define the LLM initialization function
def get_llama2_llm():
    # Create a Bedrock LLM instance
    llm = Bedrock(model_id="amazon.titan-embed-text-v1", client=bedrock)
    return llm

# Define the response generation function
def get_response_llm(llm, vectorstore_faiss, query):
    # Create a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    # Get the answer from the QA chain
    answer = qa({"query": query})
    return answer["result"]

# Main execution block
if __name__ == '__main__':
    # Load the FAISS index
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    
    # Define the query
    query = "Who are the new owners?"
    
    # Initialize the LLM
    llm = get_llama2_llm()
    
    # Get the response from the LLM
    print(get_response_llm(llm, faiss_index, query))
