import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from markup import doc_qa_tools_demo
from process_docs import (
    process_pdf,
    process_docx,
    process_txt,
    process_ppt
) 


def tab1():
    st.header(("DocuBot: A document QA Bot"))
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie("https://lottie.host/28845468-6375-4bbb-a5b5-40f3742acfd1/puY00pXHQP.json")
       # st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(doc_qa_tools_demo(), unsafe_allow_html=True)

def tab2():
    st.header(("Manage vectorstore"))
    st.subheader(("Upload File"))
    # upload pdf, docx, txt, ppt
    uploaded_files = st.file_uploader(("Select files"), type=["pdf", "docx", "txt", "ppt"], accept_multiple_files=True, key="upload") 

    if uploaded_files:
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1].lower()
            file_bytes = file.read()
            if file_extension == "pdf":
                upload_status = process_pdf(file_bytes)
            elif file_extension == "docx":
                upload_status = process_docx(file_bytes)
            elif file_extension == "txt":
                upload_status = process_txt(file_bytes)
            elif file_extension == "ppt":
                upload_status = process_ppt(file_bytes)
            st.write(upload_status)
        st.success(("Files uploaded successfully!"))
    else:
        st.info(("Upload files to get started."))
    
def tab3():
    """ Chat with the Vector Store. This page will have a chatbot that interacts with the vector store. It will have button option to clear the chat history and start over. """
    st.header(("Chat with the Vector Store"))
    st.write(("This page will have a chatbot that interacts with the vector store. It will have button option to clear the chat history and start over."))
    

def main():
    st.set_page_config(page_title="Document Manage Assist", page_icon="📚", layout="wide")
    st.title(("DocuManage Assist"))
    app_mode = option_menu(("Choose a page"), ["Home", "Upload & Manage", "Chat with DocuBot"])
    if app_mode == "Home":
        tab1()
    elif app_mode == "Upload & Manage":
        tab2()
    elif app_mode == "Chat with DocuBot":
        tab3()

if __name__ == "__main__":
    main()