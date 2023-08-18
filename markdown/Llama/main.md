# Llama の概要

Llama は、Meta 社が出したオープンソースの LLM である。商用利用可能。

- [Llama の概要](#llama-の概要)
  - [実行コード](#実行コード)
  - [Llama2](#llama2)
  - [Llama2.c](#llama2c)
  - [Llama2.cpp](#llama2cpp)
  - [Llama の論文](#llama-の論文)
    - [概要](#概要)
    - [参考](#参考)

## 実行コード

- [Llama2](https://github.com/facebookresearch/llama)
- [Llama2.cpp](https://github.com/ggerganov/llama.cpp)
- [Llama2.c](https://github.com/karpathy/llama2.c)

## Llama2

一番ベーシックな Llama2 LLM アーキテクチャのコード。

## Llama2.c

PyTorch を使用して Llama 2 LLM アーキテクチャを学習し、シンプルな C ファイルで推論する。Llama 2 LLM は小さなモデルでもドメインを狭くすることで強力な性能を発揮することができることが示されています（[TinyStories 論文](https://arxiv.org/abs/2305.07759)参照）。レポートはミニマリズムとシンプルさを重視し、Meta の Llama 2 モデルをロードして推論する手法も紹介していますが、現在のコードは fp32 のモデルのみをサポートしているため、大きなモデルには対応できません。モデルの量子化に関する作業も進行中です。このプロジェクトは楽しい週末のプロジェクトとして始まり、Llama 2 アーキテクチャをシンプルな C の推論ファイルに組み込むことに焦点を当てています。

- [tinyllamas](https://huggingface.co/karpathy/tinyllamas)

## Llama2.cpp

llama.cpp の主な目的は、MacBook 上で 4 ビット整数量子化を用いて LLaMA モデルを実行すること。

- 依存関係のないプレーンな C/C++実装
- Apple シリコン M1 - ARM NEON、Accelerate、Metal フレームワークを介して最適化
- x86 アーキテクチャの AVX、AVX2、AVX512 をサポート
- F16/F32 の混合精度
- 4 ビット、5 ビット、8 ビット整数量子化のサポート
- BLAS で OpenBLAS/Apple BLAS/ARM Performance Lib/ATLAS/BLIS/Intel MKL/NVHPC/ACML/SCSL/SGIMATH などをサポート
- cuBLAS と CLBlast のサポート

## Llama の論文

### 概要

この研究では、70 億から 700 億のパラメータを持つ大規模言語モデル（LLM）の事前学習と微調整のコレクションである Llama2 を開発し、リリースする。Llama2 Chat と呼ばれる我々のファインチューニングされた LLM は、対話のユースケースに最適化されています。我々のモデルは、我々がテストしたほとんどのベンチマークにおいて、オープンソースのチャットモデルを凌駕しており、有用性と安全性に関する人間による評価に基づいて、クローズドソースのモデルの適切な代替となり得る。我々は、Llama 2-Chat のファインチューニングと安全性向上のための我々のアプローチの詳細な説明を提供し、コミュニティが我々の研究を基に、LLM の責任ある開発に貢献できるようにする。

### 参考

- [Llama の論文](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=04ReMOti9ikAX-CQWPh&_nc_ht=scontent-nrt1-1.xx&oh=00_AfD1nFYEdzfF-8yNBTAh0MIj8PIiz_t-QUEaM5YCbTIe_g&oe=64E1FF7F)
