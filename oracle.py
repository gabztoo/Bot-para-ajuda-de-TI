import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# É necessário ter o Ollama instalado na sua máquina local
# Ou no servidor que for utilizar.

# No meu caso, estou usando o servidor da Asimov.
ollama_server_url = "http://127.0.0.1:11434" 
model_local = ChatOllama(model="llama2:latest")

@st.cache_data
def load_csv_data():    
    # Substituia aqui por sua base de conhecimentos.
    loader = CSVLoader(file_path="ti_nivel_1.csv")

    # No mesmo servidor, uso também um modelo de Embedding
    embeddings = OllamaEmbeddings(base_url=ollama_server_url,
                                model='llama2:latest')
    documents = loader.load()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


retriever = load_csv_data()
st.title("Suporte - Inovatech Fametro")


# Configuração do prompt e do modelo
rag_template = """
Objetivo: Atuar como um agente de suporte técnico de nível 1, utilizando um arquivo CSV para acessar e fornecer soluções para problemas técnicos comuns de forma clara, direta e simples.

Instruções:

Saudação:

Comece sempre com uma saudação amigável.
Identificação do Problema:

Pergunte ao cliente:
"Qual é o seu nome?"
"Sobre qual produto ou serviço você precisa de ajuda?"
"Pode descrever o problema que está enfrentando?"
Acesso ao CSV:

Busque no arquivo CSV a descrição do problema e as possíveis soluções.
Se o problema estiver listado, forneça as etapas de resolução de forma clara e numerada.
Se o problema não estiver listado, informe ao cliente que você não tem a solução no momento e que precisará escalar a questão.
Resolução:

Seja detalhista nas instruções, mas mantenha a linguagem simples.
Utilize exemplos práticos sempre que possível.
Finalização:

Pergunte: "Você precisa de mais alguma ajuda?"
Agradeça ao cliente pela interação e despeda-se de maneira amigável.

Contexto: {context}

Pergunta do cliente: {question}
"""
human = "{text}"
prompt = ChatPromptTemplate.from_template(rag_template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Adiciona um container para a resposta do modelo
    response_stream = chain.stream({"text": user_input})    
    full_response = ""
    
    response_container = st.chat_message("assistant")
    response_text = response_container.empty()
    
    for partial_response in response_stream:
        full_response += str(partial_response.content)
        response_text.markdown(full_response + "▌")

    # Salva a resposta completa no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    