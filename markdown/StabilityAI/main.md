# Stability AI

## 基本情報

モデルの説明
japanese-stablelm-base-alpha-7b は、日本語と英語の多様なデータセットを用いて事前学習された 7B パラメータのデコーダのみの言語モデルです。

### 参照

- [Hugging Face の HP](https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b)
- [試してみた例](https://colab.research.google.com/drive/1f8lIJbz-8MWrynrS_YtvUjjCl5pTci7J#scrollTo=Jz2bT7eKkQaG)

## 使用方法

まず、requirements.txt にある依存関係をインストールする

```sh
pip install sentencepiece einops
```

次に、以下のコード・スニペットを使って、japanese-stablelm-base-alpha-7b によるテキスト生成を開始する：

```python
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-base-alpha-7b",
    trust_remote_code=True,
)
model.half()
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

prompt = """
AI で科学研究を加速するには、
""".strip()

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

# this is for reproducibility.
# feel free to change to get different result
seed = 23
torch.manual_seed(seed)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)
"""
AI で科学研究を加速するには、データ駆動型文化が必要であることも明らかになってきています。研究のあらゆる側面で、データがより重要になっているのです。
20 世紀の科学は、研究者が直接研究を行うことで、研究データを活用してきました。その後、多くの科学分野ではデータは手動で分析されるようになったものの、これらの方法には多大なコストと労力がかかることが分かりました。 そこで、多くの研究者や研究者グループは、より効率的な手法を開発し、研究の規模を拡大してきました。21 世紀になると、研究者が手動で実施する必要のある研究は、その大部分を研究者が自動化できるようになりました。
"""
```

様々な生成設定（top_p、repetition_penalty など）を試して、あなたのタスクに最適な設定を見つけることをお勧めします。例えば、ロールプレイのタスクには高い温度を使い、推論には低い温度を使う。

### モデルの詳細

- モデルタイプ : japanese-stablelm-base-alpha-7b モデルは NeoX 変換アーキテクチャに基づいた自動回帰言語モデルです。
- 対応言語 : 日本語
- ライブラリ : GPT-NeoX
- ライセンス : このモデルのライセンスは Apache License, Version 2.0

### トレーニング

| パラメータ | 隠されたサイズ | レイヤー | ヘッズ | シーケンス長 |
| ---------- | -------------- | -------- | ------ | ------------ |
| 7B         | 4096           | 32       | 32     | 2048         |

### 学習データセット

japanese-stablelm-base-alpha-7b は、以下のコーパスを混合した約 750B のトークンで事前学習されている。

- [Japanese/English Wikipedia](https://dumps.wikimedia.org/other/cirrussearch/)
- [Japanese mc4](https://huggingface.co/datasets/mc4)
- [Japanese CC-100](http://data.statmt.org/cc-100/ja.txt.xz)
- [Japanese OSCAR](https://oscar-project.github.io/documentation/)
- [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)

## 使用と制限

### 使用目的

このモデルは、商業的な使用を厳密に制限することなく、用途に応じた微調整を行うための基礎モデルとして、すべての個人が使用することを意図している。

### 限界とバイアス

事前学習データセットには、データクレンジングフィルターを適用した後でも、攻撃的または不適切なコンテンツが含まれている可能性があります。本番システムでこれらのモデルを使用する際には、相応の注意を払うことをお勧めします。個人や集団に危害や苦痛を与える可能性のある用途には、このモデルを使用しないでください。

- [参考資料](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b)
