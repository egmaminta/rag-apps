import random
import time
import ast
import os
import re
import base64
from io import BytesIO

from google import genai
from google.genai import types

from dotenv import load_dotenv

import pymupdf
from llama_index.core.node_parser import SentenceSplitter
from fastembed import TextEmbedding
from qdrant_client.models import (
    VectorParams,
    Distance,
    OptimizersConfigDiff,
    BinaryQuantization,
    BinaryQuantizationConfig,
    Prefetch,
    SearchParams,
    QuantizationSearchParams
)
from qdrant_client import QdrantClient
import streamlit as st

# 1. Create a .env file with the following content: GEMINI_API_KEY=your_api_key
load_dotenv('../api_key.env')


st.set_page_config(
    page_title="RAG with Page Annotation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RAG with Page Annotation")
st.markdown(
    """
    RAG with Page Annotation is a Streamlit web application that allows users to upload documents, extract text from them, and perform semantic search using a vector database. 
    This application uses `Qdrant` for vector database, `BAAI/bge-small-en-v1.5` for text embedding, and `gemini-2.0-flash` (via Google Gemini API) for text generation.
    """
)

# 2. Set session state variables
if st.session_state.get('context_bank') is None:
    # Context bank to store the document chunks
    st.session_state['context_bank'] = []

if st.session_state.get('uploaded_files') is None:
    st.session_state['uploaded_files'] = []

if st.session_state.get('stop_doc_upload') is None:
    st.session_state['stop_doc_upload'] = False

if st.session_state.get('vector_store') is None:
    # Qdrant Client is local, using in-memory storage for simplicity
    st.session_state['vector_store'] = QdrantClient(":memory:")

if st.session_state.get('embedding_model') is None:
    st.session_state['embedding_model'] = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

if st.session_state.get('collection_name') is None:
    st.session_state['collection_name'] = 'collection_test'

if st.session_state.get('vector_params') is None:
    st.session_state['vector_params'] = VectorParams(
        size=384,
        distance=Distance.DOT,
        on_disk=True,
    )

if st.session_state.get('optimizers_config') is None:
    st.session_state['optimizers_config'] = OptimizersConfigDiff(default_segment_number=4, indexing_threshold=0)

if st.session_state.get('optimizers_config_update') is None:
    st.session_state['optimizers_config_update'] = OptimizersConfigDiff(indexing_threshold=20000)

if st.session_state.get('quantization_config') is None:
    st.session_state['quantization_config'] = BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))

if st.session_state.get('shard_number') is None:
    st.session_state['shard_number'] = 4

if st.session_state.get('vectors_uploaded') is None:
    st.session_state['vectors_uploaded'] = False

if st.session_state.get('llama_index_sent_splitter') is None:
    st.session_state['llama_index_sent_splitter'] = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64
    )

if st.session_state.get('gemini') is None:
    # Write tool / function call declaration
    retrieve_info_declaration = {
        "name": "retrieve_info",
        "description": "Retrieve information from the context bank from your provided search prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search prompt to retrieve information from the context bank."
                },
            },
            "required": ["query"]
        }
    }

    st.session_state['tool_declarations'] = [retrieve_info_declaration]
    st.session_state['gemini'] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    st.session_state['system_prompt'] = ("You are a helpful assistant and an expert in document analysis. "
                                         "You are provided with a tool named `retrieve_info` that can be used to retrieve information from the context bank (a collection of chunks of text parsed from a document). "
                                         "You have the autonomy to use this tool when you think it is necessary. "
                                         "After retrieving the information, you will provide a final answer to the user. You may also provide a summary of the retrieved information. "
                                         "Maintain professionalism and clarity in your responses. "
                                         "If you are unable to find relevant information, please inform the user that you could not find any relevant information. "
                                         "If you think the retrieved information is irrelevant, you can ask the user to provide more context or clarify their request.")
    st.session_state['chat_session'] = st.session_state['gemini'].chats.create(
        model='gemini-2.0-flash',
        config=types.GenerateContentConfig(
            system_instruction=st.session_state['system_prompt'],
            temperature=0.5,
            tools=[types.Tool(function_declarations=st.session_state['tool_declarations'])],
            max_output_tokens=2048
        )
    )

    st.session_state['rephraser_system_prompt'] = ("You are a helpful assistant and an expert in rephrasing human prompts or requests. "
                                                   "Most of the time, human prompts are not clear and need to be rephrased. "
                                                   "Therefore, it is your job to rephrase the human prompt or request in a more clear and concise way knowing that the user is looking for information from a document and "
                                                   "your rephrased prompt will be used as the search prompt to retrieve information from the context bank (a collection of chunks of text parsed from a document). "
                                                   "Output must be in a JSON object format like the following: "
                                                   "```json\n"
                                                   "{\n"
                                                   "  'search_prompt': (string) <the rephrased prompt>\n"
                                                   "}\n"
                                                   "```")


# 3. Initialize chat history
if st.session_state.get('messages') is None:
    st.session_state['messages'] = []

# 4. Display chat messages from history on app rerun
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'], unsafe_allow_html=True)

# 5. Chat input / conversation
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state['stop_doc_upload'] = True

    # Add user message to chat history
    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        pixmap_b64 = ""
        parts = [types.Part(text=prompt)]
        response = st.session_state['chat_session'].send_message(message=parts)

        try:
            tool_call = response.candidates[0].content.parts[0].function_call
            if tool_call.name == 'retrieve_info':
                def retrieve_info(query: str) -> dict[str, str]:
                    """
                    Function to retrieve information from the context bank based on the provided query.

                    Args:
                        query (str): The search prompt to retrieve information from the context bank.
                    
                    Returns:
                        A dictionary containing the retrieved information and the corresponding pixmaps (highlighted/annotated pages).
                    """
            
                    rephrased_query = st.session_state['gemini'].models.generate_content(
                        model='gemini-2.0-flash-exp',
                        config=types.GenerateContentConfig(
                            system_instruction=st.session_state['rephraser_system_prompt'],
                            temperature=0.5,
                            max_output_tokens=512,
                        ),
                        contents=[types.Content(role='user', parts=parts)]
                    )
                    rephrased_query = rephrased_query.text
                    json_extract_pattern = r'```json\n(.*?)\n```'
                    match = re.search(json_extract_pattern, rephrased_query, re.DOTALL)[1]
                    rephrased_query = ast.literal_eval(match)['search_prompt']
                    query_vector = list(st.session_state['embedding_model'].embed([query]))
                    prefetch = Prefetch(
                        query=query_vector[0],
                        params=SearchParams(
                            quantization=QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0)
                        )
                    )
                    retrieved_ctx = st.session_state['vector_store'].query_points(
                        collection_name=st.session_state['collection_name'],
                        prefetch=prefetch, limit=5, with_payload=True
                    ).points

                    pixmaps = []
                    contents = []
                    for ctx in retrieved_ctx:
                        doc_name = ctx.payload['doc_name']
                        for uploaded_file in st.session_state['uploaded_files']:
                            if uploaded_file.name == doc_name:
                                try:
                                    uploaded_file.seek(0, 0)    # Ensure the file pointer is at the beginning
                                    document = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                                    page = document[ctx.payload['page_number']]
                                    rect = page.search_for(ctx.payload['content'])
                                    if rect:
                                        contents.append(ctx.payload['content'])
                                        for r in rect:
                                            page.add_highlight_annot(r)
                                        pixmap = page.get_pixmap(dpi=300)
                                        out = pixmap.pil_image().convert('RGB')
                                        io_buffer = BytesIO()
                                        out.save(io_buffer, format='PNG')
                                        out_str = base64.b64encode(io_buffer.getvalue()).decode('utf-8')
                                        out = f'<img src="data:image/png;base64,{out_str}" style="width:65%; height:auto;">'
                                        pixmaps.append(out)
                                except Exception as e:
                                    st.error(f"Error processing file {uploaded_file.name}: {e}", icon="❌")
                                finally:
                                    # Ensure the document is closed if it was successfully opened
                                    if 'document' in locals() and document:
                                        document.close()

                    return {
                        'pixmaps': pixmaps,
                        'contents': contents
                    }

                # Call the function to retrieve information from the context bank
                with st.spinner('Retrieving information from the context bank...'):
                    result = retrieve_info(**tool_call.args)
                    contents = result.get('contents', [])
                    pixmaps = result.get('pixmaps', [])
                    contents_str = ""

                    for rel_rank, content in enumerate(contents):
                        contents_str += f"RANK (based on relevance): {rel_rank + 1}\n"
                        contents_str += content
                        contents_str += "-" * 50 + "\n"

                    for pixmap in pixmaps:
                        st.markdown(pixmap, unsafe_allow_html=True)
                        pixmap_b64 += pixmap

                    function_response_part = types.Part.from_function_response(
                        name=tool_call.name,
                        response={'result': contents_str},
                    )
                    parts.append(function_response_part)
                    response = st.session_state['chat_session'].send_message(message=parts)
                    st.markdown(response.text)
        except Exception as e:
            st.markdown(response.text)

    content = pixmap_b64 + response.text if pixmap_b64 else response.text
    st.session_state['messages'].append({'role': 'assistant', 'content': content})


# 6. Sidebar UI to upload documents
with st.sidebar:
    st.title("Upload Documents")
    if not st.session_state['stop_doc_upload']:
        uploaded_files = st.file_uploader(
            label='Uploaded your documents here',
            type='pdf',
            accept_multiple_files=True,
            label_visibility="collapsed",
            key='doc_uploader'
        )

        if uploaded_files:
            processed_file_names_in_batch = set()

            for uploaded_file in uploaded_files:
                if uploaded_file.name in processed_file_names_in_batch:
                    st.warning(f"Ignoring duplicate file '{uploaded_file.name}' selected multiple times in this upload batch.", icon="⚠️")
                    continue
                if uploaded_file.name in [file.name for file in st.session_state['uploaded_files']]:
                    processed_file_names_in_batch.add(uploaded_file.name)
                    continue
    
                st.session_state['uploaded_files'].append(uploaded_file)
                processed_file_names_in_batch.add(uploaded_file.name)
                st.success(f"File {uploaded_file.name} uploaded successfully!", icon="✅")

                try:
                    uploaded_file.seek(0, 0)
                    document = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in document:
                        text = page.get_text()
                        sent_chunks = st.session_state['llama_index_sent_splitter'].split_text(text)
                        for sent_chunk in sent_chunks:
                            text = sent_chunk.strip()
                            payload_item = {
                                'doc_name': uploaded_file.name,
                                'content': text,
                                'page_number': page.number
                            }
                            st.session_state['context_bank'].append(payload_item)
                except Exception as e:
                    st.error(f'Error processing file {uploaded_file.name}: {e}', icon="❌")
                    st.session_state['uploaded_files'] = [file for file in st.session_state['uploaded_files'] if file.name != uploaded_file.name]
                finally:
                    if 'document' in locals() and document:
                        document.close()

    st.button('Process Documents', key='process_docs', disabled=not st.session_state['uploaded_files'], on_click=lambda: st.session_state.update({'vectors_uploaded': True, 'vectors_updated': True}))

    if st.session_state['vectors_uploaded']:
        with st.spinner('Processing documents...'):
            if not st.session_state['vector_store'].collection_exists(collection_name=st.session_state['collection_name']):
                st.session_state['vector_store'].create_collection(
                    collection_name=st.session_state['collection_name'],
                    vectors_config=st.session_state['vector_params'],
                    optimizers_config=st.session_state['optimizers_config'],
                    quantization_config=st.session_state['quantization_config'],
                    shard_number=st.session_state['shard_number']
                )
                st.session_state['vector_store'].upload_collection(
                    ids=[i for i in range(len(st.session_state['context_bank']))],
                    collection_name=st.session_state['collection_name'],
                    vectors=list(st.session_state['embedding_model'].embed([item['content'] for item in st.session_state['context_bank']], batch_size=64)),
                    payload=[item for item in st.session_state['context_bank']]
                )
                st.session_state['vector_store'].update_collection(
                    collection_name=st.session_state['collection_name'],
                    optimizers_config=st.session_state['optimizers_config_update'],
                )
                st.session_state['vectors_uploaded'] = False
                st.session_state['vectors_updated'] = True
                st.success('Documents processed and vectors uploaded successfully!', icon="✅")

    st.divider()
    st.markdown(
        """
        ### Context Bank
        This is a collection of all the uploaded documents. You can use this context to search for specific information within the documents.
        """
    )
    st.markdown(f"**Total documents in context bank: {len(st.session_state['context_bank'])}**")
    st.markdown(f"**No. of files uploaded: {len(st.session_state['uploaded_files'])}**")
    st.markdown(f"**Files in context bank:**")
    if st.session_state['uploaded_files']:
        for file in st.session_state['uploaded_files']:
            st.sidebar.text(f"📄 {file.name}")
    else:
        st.sidebar.text("No files uploaded yet.")
    st.divider()
    st.markdown(
        """
        ### Instructions
        1. Upload your document in PDF format.
        2. The app will extract the text from the document and store it in the context bank.
        3. You can then search for specific information within the documents using the context bank.
        """
    )