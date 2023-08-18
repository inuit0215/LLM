import numpy as np
from matplotlib import pyplot as plt

from llama_cpp import Llama
from memory_profiler import memory_usage


# プロットする関数
def plot(x, y):
    # ここからグラフ描画
    # フォントの種類とサイズを設定する。
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Times New Roman"

    # 目盛を内側にする。
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position("both")
    ax1.xaxis.set_ticks_position("both")

    # 軸のラベルを設定する。
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Memory [MB]")
    ax1.plot(x, y, lw=1, color="red", label="Memory usage")

    # レイアウトと凡例の設定
    fig.tight_layout()
    plt.legend()

    # グラフを表示する。
    # plt.show()
    plt.savefig("./img/memory_usage.png")
    # plt.close()
    return


def func():
    # LLMの準備
    llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q4_0.bin")

    # プロンプトの準備
    prompt = """### Instruction: Translate the Input text from English to Japanese.Input text is the title of a news article.
    Input: China's State Developers Warn of Major Losses as Crisis Spreads
    ### Response:"""

    # 推論の実行
    output = llm(
        prompt,
        temperature=0.1,
        stop=["Instruction:", "Input:", "Response:", "\n"],
        echo=True,
    )
    print(output["choices"][0]["text"])


# memory_profilerを使う時はこの形で書かないといけない
if __name__ == "__main__":
    dt = 0.01
    memory = memory_usage((func, ()), interval=dt)

    # matplotlibでプロットする
    x = np.arange(0, len(memory), 1) * dt
    plot(x, memory)
