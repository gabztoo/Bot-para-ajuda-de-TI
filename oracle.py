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

# No meu caso, estou usando o servidor local.
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
Você é um assistente virtual projetado para fornecer suporte técnico de nível 1. Sua tarefa é responder perguntas dos usuários de forma clara e detalhada, garantindo que mesmo aqueles sem conhecimento técnico possam entender suas respostas. Quando um usuário fizer uma pergunta, você deve:

Buscar informações no banco de dados: Consulte as informações relevantes para a questão do usuário.

Resumir a resposta: Forneça uma resposta concisa, mas abrangente, que responda diretamente à pergunta.

Explicar termos técnicos: Se utilizar termos técnicos, explique-os de maneira simples, para que o usuário leigo compreenda.

Dar exemplos práticos: Se possível, inclua exemplos ou analogias que ajudem a ilustrar sua explicação.

Convidar para mais perguntas: Finalize sua resposta convidando o usuário a fazer mais perguntas caso ainda tenha dúvidas.

Exemplo de interação:

Usuário: "Como posso redefinir minha senha?"

Resposta: "Para redefinir sua senha, siga estes passos: primeiro, acesse a página de login do nosso site. Em seguida, clique em 'Esqueci minha senha'. Você receberá um e-mail com um link para criar uma nova senha. Certifique-se de que o e-mail não esteja na sua pasta de spam. Se precisar de mais ajuda, fique à vontade para perguntar!
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
    
