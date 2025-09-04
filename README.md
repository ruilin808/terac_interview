# Task 2

Included in the folder synthetic_system are three files data_generation.py, rag.py, and routing.py.
This task is mainly powered by Gemini 2.5 Flash.

data_generation.py contains all the functionalities of generating synthetic data. 
- I decided to structure all the data in json format, with the interviewee metadata and the transcripts of past interviews separated (interviewee metadata contains ID of all past participated interviews and interview transcripts contain ID of interviewee). 
- I first generated 100 different interviewees' metadata before generating the interview transcripts.
- For generating interview transcripts, I had gemini first compile me a list of random amazon products and structure them into json format as products.txt
- Before generating the full interview transcripts, I had gemini do quick web searches for each of the products in products.txt and generate 20 reviews based on actual amazon reviews and evenly distributed based on ratings
- Then, I had Gemini choose two of four most realistic reviews of each star rating and prompted it to create an interview structured in json format for each selected interview.
- prior to generation I also had Gemini match the transcript to an interviewee
- this repeats until all products from products.txt have been processed

rag.py was used for experimental purposes for chunking and querying purposes
- I used Qdrant since it's fast and scalable
- Semantic Chunking was used on interview transcripts
- Vector Embeddings (using BAAI/bge-base-en-v1.5) created vector representations of the chunks
- I used a few sample queries to test out which interviews/interviewee ID's would be returned 

routing.py was used to simulate dynamic matching of users (previous interviewees) and incoming queries in realtime.
- Basically uses the contents of rag.py and simulates real-time incoming sample queries to be processed by the system
- created several dummy agent interviewers with specializations (allowed agent duplications as well so there was no need for a queue)
- system paired agent interviewers with interviewees/users based on topic/specialization of the query

## Getting Started

Deployment is quite fast.

### Prerequisites

Before you run the program, remember to create a virtual enviroment and pip install the requirements from requirements.txt.
Ensure that Qdrant is installed through docker.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare a [Gemini api key](https://ai.google.dev/gemini-api/docs) and name it GEMINI_API_KEY in .env file

### Start

To generate the synthetic data, run
```
python data_generation.py
```
To run routing.py, first ensure docker is on before running
```
docker run -p 6333:6333 qdrant/qdrant
```
Then you are ready to run 
```
python rag.py
python router.py
```

## Authors

* **Ray** - *Initial work* - [ruilin808](https://github.com/ruilin808)


## License

This project is licensed under the MIT License
