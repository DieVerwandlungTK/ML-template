使い方
# デフォルト (ResNet + CIFAR-10)
python train.py

# 実験ごとの差分だけ上書き
python train.py experiment=exp001

# コマンドラインから個別上書き
python train.py model.lr=1e-4 trainer.max_epochs=50