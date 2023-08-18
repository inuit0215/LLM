# Llama を実行してみた

実行には二つある。一つはローカル、もう一つは Google Colab である。Google Colab でないと、GPU 用のモデルは動かない（ノートパソコンでは動かせない？ ← 動かせるのかも）。なので、Google Colab は無料で、GPU を動かせるので動かしてみる。

- [Llama を実行してみた](#llama-を実行してみた)
  - [Google Colab で動かす](#google-colab-で動かす)
    - [LangChain を使った実装](#langchain-を使った実装)
  - [ローカルで動かす](#ローカルで動かす)
    - [普通に Llama2 cpp を動かす](#普通に-llama2-cpp-を動かす)
    - [LangChain を使った複雑なプロンプトの実行](#langchain-を使った複雑なプロンプトの実行)

## Google Colab で動かす

### LangChain を使った実装

[参考資料：npaka さんのテックブログ RetrievalQA](https://note.com/npaka/n/n6d33c2181050)
[参考資料：npaka さんのテックブログ LlamaIndex](https://note.com/npaka/n/n3e1b59d1ac9e)
[Github：LlamaIndex](https://github.com/jerryjliu/llama_index)

## ローカルで動かす

### 普通に Llama2 cpp を動かす

cpp は、Llama2 が MacBook で動くように C 言語に変換してくれているアプリケーション。だから、GPU 上ではなくて、CPU で動く。すごい。

モデルは[HuggingFace の Llama2 7B GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)にある。GGML は GPU なしでチャット AI を動作させる機械学習用の tensor ライブラリのこと。GGML は次の特徴を持つ。

- C で書かれている
- 16 ビット浮動小数点をサポート
- 整数量子化をサポート（例：4bit、5bit、8bit）
- 自動微分
- 組み込みの最適化アルゴリズム「ADAM」「L-BFGS」などを搭載
- Apple シリコン用に最適化
- x86 アーキテクチャでは AVX/AVX2 組み込み関数を利用
- WebAssembly および WASM SIMD による Web サポート
- サードパーティへの依存関係なし
- 実行時にメモリ割り当てなし
- ガイド付き言語出力のサポート

よって、CPU でも動く。

コードとしてはかなりシンプルで、商用もできる。

[参考資料：npaka さんのテックブログ](https://note.com/npaka/n/n0ad63134fbe2)

### LangChain を使った複雑なプロンプトの実行

Index 機能を搭載させたいと考え、LangChain を使った実装も行った。

[Text Embedding：intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)

[参考資料：npaka さんのテックブログ](https://note.com/npaka/n/n3164e8b24539)
