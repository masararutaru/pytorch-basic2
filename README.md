# PyTorch Basics (MNIST/CIFAR-10)

手で学習ループを書き、MNIST(MLP)/CIFAR-10(CNN)を学習する最小構成。

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
```

> GPU版PyTorchは環境に合わせて公式手順でインストールしてください。

## Run

```bash
# MNIST (MLP)
python -m src.train --cfg configs/default.yaml

# CIFAR-10 (CNN)
# configs/default.yaml の task を `cifar10-cnn` に変更してから実行
```

## Structure

* `src/datasets.py`: torchvisionでMNIST/CIFAR-10を取得しDataLoader化
* `src/models.py`: MLP / シンプル3層CNN
* `src/train.py`: 学習・評価ループ、StepLR対応
* `configs/default.yaml`: 学習設定

## Notes

* データは `./data/` に自動ダウンロード
* チェックポイントは `./ckpts/` に保存
