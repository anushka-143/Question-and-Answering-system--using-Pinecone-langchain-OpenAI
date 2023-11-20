def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print('Loading file {file}')
        loader = PyPDFLoader(file)
    elif extension =='.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading...{file}')
        loader = Docx2txtLoader(file)
    elif extension == '.csv':
        from langchain.document_loaders.csv_loader import CSVLoader
        print(f'Loading...{file}')
        loader = CSVLoader(file)
    elif extension == '.json':
        from langchain.document_loaders.json_loader import JSONLoader
        print(f'Loading...{file}')
        loader = JSONLoader(file)
    elif extension=='.html':
        from langchain.document_loaders import UnstructuredHTMLLoader
        print(f'Loading...{file}')
        loader = UnstructuredHTMLLoader(file)
    elif extension=='.md':
        from langchain.document_loaders import UnstructuredMarkdownLoader
        print(f'Loading...{file}')
        loader = UnstructuredMarkdownLoader(file)
    else :
        print('Document Format is not supported!')
        return None

    data = loader.load()
    return data
def load_from_wikipedia(query, lang = 'en', load_max_docs = 2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query= query, lang = lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def chunk_data(data,chunk_size = 256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
    print(f'Total tokens :{total_tokens}')
    print(f'Embedding Cost in USD:{total_tokens/1000*0.0004:.6f}')

def insert_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    from secretkeys import pinecone_env, pinecone_key
    pinecone.init(api_key=pinecone_key,environment=pinecone_env)

    if index_name in pinecone.list_indexes():
        print('index name already exists')
        vector_store = Pinecone.from_existing_index(index_name,embeddings)

    else:
        print(f'Creating index {index_name} and embeddings..')
        pinecone.create_index(index_name,dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings,index_name =index_name)

    return vector_store

def delete_pinecone_index(index_name = 'all'):
    import pinecone
    from secretkeys import pinecone_env, pinecone_key
    pinecone.init(api_key=pinecone_key, environment=pinecone_env)
    if index_name=='all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name}...')
        pinecone.delete_index(index_name)

#this will embed thr chunks and add both the chunks and the embeddings into pinecone index

def ask_and_get_answers(vector_store,q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature = 1)
    retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs = {'k':3})
    chain = RetrievalQA.from_chain_type(llm = llm, chain_type='stuff',retriever= retriever)
    answer = chain.run()

    return answer

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature = 1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm,retriever)
    result = crc({'question':question,'chat_history':chat_history})
    chat_history.append((question,result['answer']))
    return result, chat_history

data = load_document('GenerativeAI.pdf')
print(data[1].page_content)
print(data[1].metadata)
print(f'You have {len(data)} pages in your data')
'''data = load_from_wikipedia('GPT-4')
print(data[0].page_content)'''

chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)

print_embedding_cost(chunks)

delete_pinecone_index()

index_name = 'askadoc'
vector_store = insert_fetch_embeddings(index_name)

i = 1
while True:
    q = input(f"question {i}:")
    i = i+1
    if q.lower() in ['quit', 'exit','bye']:
        print('byee bye')
        break
    answer = ask_and_get_answers(vector_store,q)
    print(f"\nAnswer: {answer}")
    print(f"\n{'-'*50}\n")

#asking the memory:
chat_history = []
question = 'What is generative ai?'
result,chat_history = ask_with_memory(vector_store,question,chat_history)
print(result['answer'])
print(chat_history)

question = 'Why it is used?'
result,chat_history = ask_with_memory(vector_store,question,chat_history)
print(result['answer'])
print(chat_history)

