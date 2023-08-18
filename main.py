from llama_cpp import Llama

# LLMの準備
llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q4_0.bin")

# プロンプトの準備
prompt = """### Instruction: Translate the Input text from English to Japanese.
### Input: China's State Developers Warn of Major Losses as Crisis Spreads
### Response:"""

# 推論の実行
output = llm(
    prompt,
    temperature=0.1,
    stop=["Instruction:", "Input:", "Response:", "\n"],
    echo=True,
)
print(output["choices"][0]["text"])
