# Deploy rag application usng AWS(aws lmbda, aws ECR), langchain, huggungface, docker

RAG - Data Ingestion ---> Retrieval ---> Generation

Data(csv,tsv,text,pdf) ---> Fetching ---> chunking ---> embedding ---> knowledge base(vectorbd, chromadb, FAISS, can be anything to store the data)  ---> INGESTION

If query is given by the user then with similarity serach it will go to the knowledge base and extracts the relevant docs from the kb ---->  RETRIEVAL


If input query/prompt is given to the llm then it is going to generate the response. if the model(llm) don't answer to the given prompt then we need to follow some prompt mechanics.
1) few shot prompting  - who will be the next PM?(based on the surveys provided in few shots the response will be given)
2) zero shot prompting  - example question - who is the PM of india?   ---> GENERATION 


COMMANDS
1) aws configure - create IAM user and specify the access key, secret access key, region and format
2) create a conda env - conda create -p env python=3.10 -y
3) source activate ./env
4) pip install -r requirements.txt
5) to create a package - python setup.py install
6) python ingestion.py
7) python retrievalandgeneration.py
8) create streamlit app for the application.
9) push the code to github.


Application is dockerised into docker image and that is pushed to ECR and then they can be consumed in EC2/lambda func/apprunner/ECS:
1) create the ECR repository.
2) write dockerfile for the entire application.
3) go to the secrets in the settings of the repository and add the aws access key id, aws default region,aws ecr repo uri, aws secret access key to the repository secrets.
4) now write workflows main.yml file for the ci/cd pipeline.
5) no when you push the code to the repo then automatically cicd flow will execute.
6) as it is mentioned in the main.yml, the image will be pushed to the ecr repository.


APPRUNNER-
1) create the app runner with the ecr image created and then we can access the application once it is deployed.

configure aws access key id-
1) create IAM user to access this particular resources.
2) update the access key and secret key id in the secrets of the github.
3) and then do aws configure and update the credentials in the vs code also.

Amazon Sagemaker-
run llm for different applications to generate text, image, audio and for generation and summarization.
pass input to llm and generate output.
and we want to deploy the llm, the llm is given in huggung face and fetch it and deploy it in the amazon sagemaker studio.

the llm is passed to the hugging face hub and deploy in sagemaker studio.

1) create domain.
