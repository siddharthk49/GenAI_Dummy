import os

openAI_Key = ""

pdf_folder_path = "C:\\Global_GenAI\\test_docs"

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.chat_models import ChatOpenAI
import textwrap
# langchain_openai import OpenAIEmbeddings
#from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd



def create_embeddings(pdf_folder_path):
    persist_directory = 'db'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    all_documents = []
    raw_document = []

    # Create a loader for the current document
    for file in os.listdir(pdf_folder_path):
        pdf_path = os.path.join(pdf_folder_path, file)
        print(pdf_path)
        loader = PyPDFLoader(pdf_path)
        raw_document.extend(loader.load())
        print(raw_document)

        # Load the raw document
    #raw_document = loader.load()

        # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500, separator="\n\n")
    docs = text_splitter.split_documents(raw_document)
    all_documents.extend(docs)

        # Initialize the Chroma database
    id_list = [str(x) for x in range(len(all_documents))]
    embedding = OpenAIEmbeddings(openai_api_key = openAI_Key)
    db = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding,
        ids=id_list,
        persist_directory=persist_directory
    )
    print("DB has been successfully created!!")
    return db

def run_chain(query):
    embedding = OpenAIEmbeddings(openai_api_key = openAI_Key)

    db = Chroma(persist_directory="db", embedding_function=embedding)

    chunks_list = []

    llm = ChatOpenAI(temperature=0, model_name='gpt-4-32k', openai_api_key= openAI_Key)





    # Perform a similarity search with scores
    docs_with_score = db.similarity_search_with_score(query, 10)
    #print(docs_with_score)

    df = pd.DataFrame(data=docs_with_score, columns=["chunk", "chunk_score"])
    top_5_chunks = df[:5]
    chunks_list.append(top_5_chunks)
    extracted_chunks = [chunk.page_content for chunk in top_5_chunks['chunk']]
    context = "\n".join(extracted_chunks)

    prompt_template = f"""You are a helpful AI assistant assigned with the task of responding to the given query {query} based on the given {context}. 
       Your task is to provide the answers exactly as they are available in the context.
       Do not hallucinate, and do not make up the response unnecessarily and only provide what is factual and known from the given context.
       You should always fetch information or details from the given context.
       If the information is not explicitly stated, analyze the document thoroughly and provide the most relevant response based on the context.
       The final answer should be clear, perfect, and descriptive.
       Make sure the alignment of the final response is concise and accurate.
       Provide the response in bullet points.
       If specific details from the query are missing, explicitly state that those specific details are not available. 
       Only use the data you have in the input context. Do not give any other information.
       Do not give any extra information. Just give the information which is asked. Be crisp.
       Guardrails:
       Avoid making claims of sentience or consciousness.
       Do not express personal opinions or beliefs.
       Do not engage in emotional responses.
       If no relevant document is found:
       State clearly: "I'm sorry. I'm afraid I don't have an answer to that {query}."
       Offer a suggestion: "Please ask me something related to NYGH." (Replace NYGH with your specific document collection acronym or name)
       Helpful Answer:"""
    BULLET_POINT_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    chain = LLMChain(prompt=BULLET_POINT_PROMPT, llm=llm, verbose=False)

    # Generate the response asynchronously
    output_summary = chain.run({'context': context, 'question': query})

    wrapped_text = textwrap.fill(output_summary, width=100, break_long_words=False, replace_whitespace=False)
    return wrapped_text


def improved_response_generator(response, feedback):
    # Initialize the language model and prompt outside the loop
    llm = ChatOpenAI(temperature=0, model_name='gpt-4-1106-preview',
                     openai_api_key='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')

    prompt_template2 = f"""Based on the given feedback {feedback}, please incorporate all the changes in the generated response {response} and generate a response.
    Make sure to incorporate all the improvements mentioned in the feedback provided and address every detail mentioned in the feedback.
    Go through the feedback to improve the response and generate a new version of the response.
    Improve the response as per mentioned in the feedback and do not make up the answer.
    Do not hallucinate, and do not make up the response unnecessarily and and only provide what is factual and known from the given context.
    The final answer should be clear, perfect, and descriptive.
    Make sure the alignment of the final response is concise and accurate.
    Provide the response in bullet points and the format remains same always.

        """

    PROMPT = PromptTemplate(
        template=prompt_template2,
        input_variables=["response", "feedback"],
    )

    chain2 = LLMChain(prompt=PROMPT, llm=llm, verbose=False)

    improved_response = chain2.run({'response': response, 'feedback': feedback})

    return improved_response

if __name__ == "__main__":

    while True:
    #db = create_embeddings(pdf_folder_path)
        query = "Please provide me with details about emergency care at NYGH."
        response = run_chain(query)
        print(response)
        feedback = input("Enter your feedback for the response: \n")
        print("\n")
        user_feedback = improved_response_generator(response, feedback)
        print(user_feedback)
        print("\n")

