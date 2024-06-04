import utils

from fastapi import FastAPI, Response, UploadFile

app = FastAPI(
    title="Test-Bot API",
    description="This API is used to train a test-bot model with text or file data.",
)

@app.post("/chatbot", response_model=dict)
async def chatbot(query: str, collection: str, response: Response):
    """
    Endpoint for the chatbot API.

    Parameters:
    - query (str): The user's query.
    - collection (str): The collection to search for responses.
    - response (Response): The HTTP response object.

    Returns:
    - dict: The chatbot response.

    """
    return utils.chatbot_response(response, collection, query)

@app.post("/train_file", response_model=dict)
async def train_file(file: UploadFile, collection: str, response: Response):
    """
    Trains the model using the provided file.

    Args:
        file (UploadFile): The file to be trained.
        collection (str): The collection to store the trained data.
        response (Response): The response object.

    Returns:
        dict: A dictionary containing the response data.
    """
    filename, raw_data = utils.get_file_data(file, await file.read())
    return utils.train_data(response, collection, raw_data, filename)
