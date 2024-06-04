import io
import settings
import requests
from datetime import datetime
from fastapi import status
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient

embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(url=settings.QDRANT_URL)


def success_response(response, data):
    """
    Set the status code of the response to 200 (OK) and return the data.

    Args:
        response (HttpResponse): The HTTP response object.
        data (dict or str): The data to be returned. If it's not a dictionary, it will be wrapped in a dictionary with a "message" key.

    Returns:
        dict: The data to be returned.

    """
    response.status_code = status.HTTP_200_OK
    return data if isinstance(data, dict) else {"message": data}


def bad_request_error(response, data):
    """
    Sets the status code of the response to 400 (Bad Request) and returns a dictionary
    with a message containing the provided data.

    Args:
        response (HttpResponse): The response object to modify.
        data (str): The data to include in the response message.

    Returns:
        dict: A dictionary with a "message" key containing the provided data.

    """
    response.status_code = status.HTTP_400_BAD_REQUEST
    return {"message": data}


def get_file_data(file_data: bytes, contents) -> tuple:
    """
    Extracts the filename and raw data from the given file data.

    Args:
        file_data (bytes): The file data.
        contents: The contents of the file.

    Returns:
        tuple: A tuple containing the filename and raw data.
    """
    filename = file_data.filename

    if filename.lower().endswith(".pdf"):
        raw_data = extract_text_from_pdf(contents)
    else:
        raw_data = contents.decode("utf-8")

    return filename, raw_data


def extract_text_from_pdf(pdf_data: bytes) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_data (bytes): The PDF file data as bytes.

    Returns:
        str: The extracted text from the PDF file.
    """
    pdf_text = ""
    pdf_file = io.BytesIO(pdf_data)
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    return pdf_text


def train_data(response, collection_name: str, raw_data: str, filename: str) -> dict:
    """
    Trains the data using the provided raw data and metadata.

    Args:
        response: The response object to send the result.
        collection_name (str): The name of the collection to train the data.
        raw_data (str): The raw data to be used for training.
        filename (str): The name of the file being trained.

    Returns:
        dict: A dictionary containing the response data and status.

    """
    metadata = {"bash": filename, "date": str(datetime.date(datetime.now()))}
    documents = get_text_splitter_documents(raw_data=raw_data, metadata=metadata)
    response_data, status = train_document(collection_name, documents)
    return (
        success_response(response, response_data)
        if status
        else bad_request_error(response, response_data)
    )


from typing import List


def train_document(collection_name: str, documents: List[Document]):
    """
    Trains a document by adding it to the specified collection.

    Args:
        collection_name (str): The name of the collection to add the documents to.
        documents (List[Document]): A list of documents to be added.

    Returns:
        Tuple[str, bool]: A tuple containing the response message and the status of the operation.
            - The response message indicates whether the documents were successfully added or not.
            - The status is a boolean value indicating the success (True) or failure (False) of the operation.
    """
    try:
        if collection_exists(collection_name):
            qdrant = Qdrant(qdrant_client, collection_name, embeddings)
            qdrant.add_documents(documents)
        else:
            Qdrant.from_documents(
                documents,
                embeddings,
                url=settings.QDRANT_URL,
                prefer_grpc=True,
                collection_name=collection_name,
            )
            response, status = "Added documents to the collection.", True
    except Exception as e:
        print(e)
        response, status = (
            f"Failed to add documents to the collection. error: {str(e)}.",
            False,
        )
    return response, status


def get_text_splitter_documents(raw_data=None, metadata={}):
    """
    Splits the raw data into smaller chunks using a text splitter and creates a list of Document objects.

    Args:
        raw_data (str): The raw data to be split into smaller chunks.
        metadata (dict): Additional metadata to be associated with each Document object.

    Returns:
        list: A list of Document objects, each representing a chunk of the raw data.

    """
    data_trained = []
    if raw_data:
        data_trained = [
            Document(metadata=metadata, page_content=doc)
            for doc in RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200, add_start_index=True
            ).split_text(raw_data)
        ]
    return data_trained


def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in the QDRANT service.

    Args:
        collection_name (str): The name of the collection to check.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    url = f"{settings.QDRANT_URL}/collections/{collection_name}"
    response = requests.get(url)
    return response.status_code == 200


def chatbot_response(response, collection_name: str, query: str) -> dict:
    """
    Generates a response from the chatbot based on the given query.

    Args:
        response (dict): The response object to be returned.
        collection_name (str): The name of the collection to search for documents.
        query (str): The query string to search for similar documents.

    Returns:
        dict: The response object containing the generated chatbot response.

    Raises:
        Exception: If an error occurs while generating the response.

    """
    try:
        memory = get_memory_buffer_window()
        prompt = get_prompt()
        chain = get_chain(memory, prompt)
        qdrant = get_document(collection_name)
        input_documents = qdrant.similarity_search(query)
        response_data = chain.run(input_documents=input_documents, human_input=query)
        return success_response(response, response_data)
    except Exception as e:
        print(e)
        response_data = f"Failed to get response. error: {str(e)}."

    # Return a response based on success or error
    return bad_request_error(response, response_data)


def get_memory_buffer_window():
    """
    Creates and returns a ConversationBufferWindowMemory object.

    Returns:
        ConversationBufferWindowMemory: The created ConversationBufferWindowMemory object.
    """
    return ConversationBufferWindowMemory(
        k=1, input_key="human_input", memory_key="chat_history"
    )


def get_prompt():
    """
    Returns a template prompt for the chatbot.

    The template prompt includes placeholders for the context, human input, and chat history.
    It is used to generate prompts for the chatbot's conversation.

    Returns:
        PromptTemplate: The template prompt object.
    """
    prompt = settings.TESTBOT_PROMPT
    prompt = prompt.replace("}", "")

    template = (
        ""
        + prompt
        + """
    ```{context}```
    {chat_history}
    Human: {human_input}
    Chatbot:"""
    )

    template_prompt = PromptTemplate(
        input_variables=["context", "human_input", "chat_history"],
        template=template,
    )

    return template_prompt


def get_llm():
    """
    Returns an instance of ChatLiteLLM with the default model name, temperature, and max tokens.

    Returns:
        ChatLiteLLM: An instance of ChatLiteLLM.
    """
    return ChatLiteLLM(
        model=settings.DEFAULT_MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


def get_chain(memory, prompt):
    """
    Retrieves a question and answer chain based on the given memory and prompt.

    Args:
        memory (str): The memory to use for the question and answer chain.
        prompt (str): The prompt to use for generating the question and answer chain.

    Returns:
        QuestionAnswerChain: The generated question and answer chain.

    """
    return load_qa_chain(
        llm=get_llm(),
        chain_type="stuff",
        memory=memory,
        prompt=prompt,
        verbose=True,
    )


def get_document(collection_name: str) -> dict:
    """
    Retrieves a document from the specified collection.

    Args:
        collection_name (str): The name of the collection.

    Returns:
        dict: The document retrieved from the collection.
    """
    if collection_exists(collection_name):
        qdrant = Qdrant(qdrant_client, collection_name, embeddings)
    else:
        qdrant = Qdrant.from_documents(
            [Document(page_content="")],
            embeddings,
            url=settings.QDRANT_URL,
            prefer_grpc=True,
            collection_name=collection_name,
        )
    return qdrant
