import os
import base64
from dotenv import load_dotenv
from pinecone_actions import load_vectorstore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(
    api_key = OPENAI_API_KEY,
    model = "gpt-4o",
    temperature=0
)

chat_streaming_model = ChatOpenAI(
    api_key = OPENAI_API_KEY,
    model = "gpt-4o",
    streaming = True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.2
)

def get_image_description(image_bytes: bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe what is in the the image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ],
    )
    
    response = chat_model.invoke([message])
    return response.content

def get_chat_chain(memory):
    prompt_template = """you are an AI assistant designed to help humans regarding any questions they have about documents.
    You are having a conversation with a Human. Strictly give answers related to the context of the document but you can use your knowledge if you think necessary. You can response to greetings and feedbacks as well.
    
    Here is your conversation history with same Human:
    {chat_history}
    ==========
    Human's New Input: {question}
    =========
    Context:
    {context}
    =========
    Answer in Markdown:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    vectorstore = load_vectorstore()
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    retrieved_docs = retriever.get_relevant_documents("what this document about?")
    if not retrieved_docs:
        print("No relevant documents found.")
    else:
        for doc in retrieved_docs:
            print("Metadata:", doc.metadata)

    chat_chain = ConversationalRetrievalChain.from_llm(llm=chat_streaming_model, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                combine_docs_chain_kwargs=chain_type_kwargs,
                                                return_source_documents=False,
                                                get_chat_history=lambda h : h,
                                                memory=memory,
                                                )
    return chat_chain
