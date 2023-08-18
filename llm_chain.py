import logging
import sys

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# ドキュメントの読み込み
with open("source.txt") as f:
    test_all = f.read()

# チャンクの分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # チャンクの最大文字数
    chunk_overlap=20,  # オーバーラップの最大文字数
)
texts = text_splitter.split_text(test_all)

# チャンクの確認
print(len(texts))
for text in texts:
    print(text[:10].replace("\n", "\\n"), ":", len(text))


# インデックスの作成
# index = FAISS.from_texts(
#     texts=texts,
#     embedding=HuggingFaceEmbeddings(
#         model_name="intfloat/multilingual-e5-large",
#         cache_folder="./model"
#     ),
# )
# index.save_local("storage")

# インデックスの読み込み
index = FAISS.load_local(
    "storage",
    HuggingFaceEmbeddings(model_name="./model/intfloat_multilingual-e5-large"),
)

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# LLMの準備
# llm = OpenAI(temperature=0, verbose=True)
llm = LlamaCpp(
    model_path="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

# 質問応答チェーンの作成
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(search_kwargs={"k": 4}),
    verbose=True,
)

# 質問応答チェーンの実行
print("A1:", qa_chain.run("What team is Arsenal?"))
print("A2:", qa_chain.run("Where is Arsenal based?"))
print("A3:", qa_chain.run("What is Arsenal's motto?"))
