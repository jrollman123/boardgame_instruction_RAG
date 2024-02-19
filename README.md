# Boardgame RAG

This code is now outdated as it uses the earlier langchain iteration.

```python
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModel, AutoConfig
from ctransformers import AutoModelForCausalLM
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, CTransformers
import re
import os

```

## Import Data

```python
loader = PyPDFLoader("robinson_crusoe_rulebook.pdf")
data_raw = loader.load()
print (f'{len(data_raw)} document(s) in your data')
```

40 document(s) in your data

```python
data = data_raw

for text in data:
    text.page_content = text.page_content.replace('•', ' ')
    text.page_content = re.compile(r'\,\s{2,10}\n').sub(', ',text.page_content)
    text.page_content = re.compile(r'\:\s{2,10}\n').sub(': ',text.page_content)
    text.page_content = re.compile(r'\.\n').sub('. ',text.page_content)
    text.page_content = re.compile(r'\,\n').sub(', ',text.page_content)
    text.page_content = re.compile(r'\:\n').sub(': ',text.page_content)
    text.page_content = text.page_content.replace(' . ',' ')
    text.page_content = re.compile(r'[0-9][0-9][0-9]+').sub('',text.page_content)
    
text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n","."],chunk_size=750, chunk_overlap=10)

texts = text_splitter.split_documents(data)

for text in texts:
    text.page_content = re.compile(r'\n').sub('',text.page_content)
    text.page_content = re.compile(r'\s+').sub(' ',text.page_content)

texts[30].page_content
```

'8I. EVENT PHASE You must work together to reach the scenario goal within the set number of rounds. Often, this requires building special Items or exploring specific locations (see the Scenario sheets and the appendix on pages 28-32). You must cooperate to overcome the obstacles; you only win as a group. If you do not manage to reach the scenario goal within the time limit, or if one of the characters dies, all players lose the game. A Game round comprises 6 Phases: 1. EVENT PHASEAn Event card is revealed and its Event Effect applied. The card is then placed in the right-hand Threat space, moving the other cards, which can trigger Threat Effects. Sometimes multiple cards may need to be drawn and resolved. 2. MORALE PHASE’

## Embed Context

```python
embedding_function = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

try:
    db._collection.name
except:
    print('Vector Store is Empty')
else:
    db._client.delete_collection(db._collection.name)
finally:
    db = Chroma.from_documents(texts, embedding_function)
```

```python
query = "How is the Dog played?"
docs = db.similarity_search(query)
docs[3].page_content
```

'The Hunt Action provides you with food and possibly fur, but the resolving character is at risk of suffering wounds if the Weapon level is not high enough. This Action can be resolved once per Beast card in the Hunting deck, as each Beast can only be hunted once. Once the Hunting deck is empty, this Action can no longer be taken, but as long as enough cards are available, the Action can be resolved multiple times per round. Each Hunt Action requires exactly 2 Action pawns. For each Hunt Action, the Action pawns are placed in the Action space stacked on top of each other, with the character owning the topmost Action pawn resolving the Action. When the Action is resolved, the topmost Beast card is drawn from the Hunting’

## Load Local LLM

```python
local_llm_2 = CTransformers(model='TheBloke/Llama-2-13B-chat-GGML', 
                            model_file='llama-2-13b-chat.ggmlv3.q4_K_M.bin', 
                            config={"temperature" : .1})
```

## Configure Prompts and Chains

### Stuff Method Prompt Chain

```python
prompt_template = """ Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

query = "How is the Dog played?"
docs = db.similarity_search(query)
print(docs[0:3])
```

[Document(page_content='The Dog is represented by his card and 1 purple Action pawn. In a solo game, place his card next to the board and the Action pawn on it. He is used like a neutral Action pawn in all respects. He can be used every round for either the Hunt or Explore actions only. Dog’s Action pawn need not be assigned to any Action if the players do not wish it. He cannot die. VARIANTSEASIER GAMEIf players think a scenario is too hard for them, they can make it easier by: Adding the Dog. This is especially recommended for 3 players. Adding Friday Drawing more Starting Equipment Using fewer Event cards with the book symbol and more Event cards with the adventure symbol when creating the deck (step 15 of the setup). For example, using 4', metadata={'page': 26, 'source': 'robinson_crusoe_rulebook.pdf'}), Document(page_content='2 PLAYERSIf players randomly draw characters, it is recommended they should select from the carpenter, cook and explorer. The special abilities of the soldier are less useful in the 2-player game for most scenarios. The 2-player game also adds an additional character - Friday. When setting up the game, place Friday next to the board in reach of both players and place the white Action pawn representing Friday on it. Friday is explained in greater detail on page 27. SOLOThe solo variant uses the same rules as the 2-player game. Additionally, the sole character also has the Dog available alongside Friday. The Dog’s card is placed next to the board and the purple Action pawn placed on it.', metadata={'page': 25, 'source': 'robinson_crusoe_rulebook.pdf'}), Document(page_content='2-player game, place his card next to the board and the Action pawn on it. Place a Wound marker on the square space on the left of his Wound track. Friday counts as an additional character, but has the following rules: Friday can never be the First Player. Friday is not affected by Event cards - neither immediate nor Threat Events Exception: the Argument card, see appendix. Friday can be assigned to any Action. If it is successful, it is resolved like any other character’s Action. Friday can be assigned to an Action as the only Action pawn or with neutral Action pawns (including the Dog). If Friday is assigned to an Action along with another player’s Action pawn, he may only support. In this case, the character always resolves th', metadata={'page': 26, 'source': 'robinson_crusoe_rulebook.pdf'})]

```python
chain = load_qa_chain(local_llm_2, chain_type="stuff", prompt=PROMPT)
stuff_result = chain.run(input_documents=docs{0:1}, question=query)
print(stuff_result)
```

The Dog is played like a neutral Action pawn that can be used every round for either the Hunt or Explore actions only. It cannot die and is placed next to the board with one purple Action pawn on it.

### Map-Reduce Prompt Chain

```python
question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer.

QUESTION: {question}
=========
{summaries}
=========
Answer:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

chain = load_qa_chain(local_llm_2, chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
map_reduce_result = chain.run(input_documents=docs, question=query)
print(map_reduce_result)
```

In a solo game, the Dog can be used as a neutral Action pawn like any other, and it is recommended to use it with caution due to the risk of suffering wounds if the Weapon level is not high enough when using the Hunt Action.

## Compare to OpenAI GPT 3.5

```python
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = OpenAI(temperature=.1, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

GPT_result = chain.run(input_documents=docs, question=query)
print(GPT_result)
```

The Dog is used like a neutral Action pawn in all respects. He can be used every round for either the Hunt or Explore actions only. Dog’s Action pawn need not be assigned to any Action if the players do not wish it. He cannot die.
