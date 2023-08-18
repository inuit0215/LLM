# LINE の日本語 LLM を試す

LINE の日本語 LLM を試す。試すのは、Google Colab とローカルで試す。将来的に、サーバー上で動かすことを考えているため、メモリや CPU の使用状況を調べる。

- [LINE の日本語 LLM を試す](#line-の日本語-llm-を試す)
  - [Google Colab で実行](#google-colab-で実行)
    - [疑問点](#疑問点)
  - [ローカルで実行](#ローカルで実行)
    - [CUDA って何？](#cuda-って何)

## Google Colab で実行

[Google Colab でのコード](https://colab.research.google.com/drive/15TtMHCmo5vsQ4pzIx9Paq6e4c_yD3QBn?usp=sharing)にて、実行可能。

### 疑問点

1. **AutoModelForCausalLM の使い方がよくわからない**  
   AutoModelForCausalLM の使い方によって、Google Colab で RAM 不足が解消したり、色々動作が異なる。AutoModelForCausalLM が何かを調べる。
2. **aa**  
   aaa

## ローカルで実行

CUDA が必要できなそう。CUDA ってそもそもなんなのかわからなくなった。GPU が使えないってことなんだけど、そうなると実行が大変そう。かなりの計算資源を食うし、常時動かそうと思うとお金もかかる気がする。CUDA について少し勉強するとともに、どうすれば良いか考えたいと思う。一旦、ChatGPT の API を叩くのがいいのかな。

### CUDA って何？
