# Deploy rag application usng AWS(aws lmbda, aws ECR), langchain, huggungface, docker

RAG - Data Ingestion ---> Retrieval ---> Generation

Data(csv,tsv,text,pdf) ---> Fetching ---> chunking ---> embedding ---> knowledge base(vectorbd, chromadb, FAISS, can be anything to store the data)  ---> INGESTION

If query is given by the user then with similarity serach it will go to the knowledge base and extracts the relevant docs from the kb ---->  RETRIEVAL


If input query/prompt is given to the llm then it is going to generate the response. if the model(llm) don't answer to the given prompt then we need to follow some prompt mechanics.
1) few shot prompting  - who will be the next PM?(based on the surveys provided in few shots the response will be given)
2) zero shot prompting  - who is the PM of india?   ---> GENERATION 


COMMANDS
1) aws configure
2) create a conda env - conda create -p env python=3.10 -y
3) source activate ./env
4) pip install -r requirements.txt
5) to create a package - python setup.py install
6) ingestion.py
7) retrievalandgeneration.py




Application is dockerised into docker image and that is pushed to ECR and then they can be consumed in EC2/lambda func/apprunner/ECS