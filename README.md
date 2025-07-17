課題..py

＜Kaggleの何のデータ（URLとデータの名前）を用いたのか。＞

薬の処方のデータ　https://www.kaggle.com/datasets/prathamtripathi/drug-classification

＜何を入力として何を推定するのか。＞

多クラス分類（5つ）

入力　Age、Sex、Blood Pressure Levels (BP)、Cholesterol Levels、Na to Potassium Ration

出力　Drug type

＜ニューラルネットワークの構成（ニューロン数，層数など）＞

入力層、中間層2つ、出力層からなるニューラルネットワーク。1つ目の中間層のニューロン数は64、2つ目の中間層のニューロン数は32

損失関数はCrossEntropy

オプティマイザはAdam

学習率は0.001

＜結果と考察＞

Epoch [10/100], Loss: 0.7863

Epoch [20/100], Loss: 0.4239

Epoch [30/100], Loss: 0.1854

Epoch [40/100], Loss: 0.0924

Epoch [50/100], Loss: 0.0563

Epoch [60/100], Loss: 0.0387

Epoch [70/100], Loss: 0.0275

Epoch [80/100], Loss: 0.0214

Epoch [90/100], Loss: 0.0170

Epoch [100/100], Loss: 0.0136

<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/c937eba6-456b-4186-bf47-461994103484" />

Test Accuracy: 97.50 %

課題2..py

＜Kaggleの何のデータ（URLとデータの名前）を用いたのか。＞

スポーツカーの価格　[https://www.kaggle.com/datasets/prathamtripathi/drug-classification](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset)

＜何を入力として何を推定するのか。＞

回帰

入力　Engine Size (L)、Horsepower、Torque (lb-ft)、0-60 MPH Time (seconds)

出力　price

＜ニューラルネットワークの構成（ニューロン数，層数など）＞

入力層、中間層2つ、出力層からなるニューラルネットワーク。1つ目の中間層のニューロン数は128、2つ目の中間層のニューロン数は64

損失関数はMSE

オプティマイザはAdam

学習率は0.001

＜結果と考察＞

Epoch [20/100], Loss: 0.1618

Epoch [40/100], Loss: 0.1235

Epoch [60/100], Loss: 0.1141

Epoch [80/100], Loss: 0.0912

Epoch [100/100], Loss: 0.0691

<img width="1000" height="800" alt="価格" src="https://github.com/user-attachments/assets/44ef02ca-40ef-49b8-8649-f40ce9109f6e" />


課題1.py

＜Kaggleの何のデータ（URLとデータの名前）を用いたのか。＞

卒業研究で使用したデータ（ボートレースのデータ484,006レース）

＜何を入力として何を推定するのか。＞

選手の勝率やコース、天候など152の説明変数を主成分分析し66に圧縮。１号艇が１着になるかならないかを推定

＜ニューラルネットワークの構成（ニューロン数，層数など）＞

入力層、中間層1つ、出力層からなるニューラルネットワーク。中間層のニューロン数は100

＜結果と考察＞

Epoch: 100/100, Train_Loss: 0.6092, Train_Acc: 0.6677, Test Loss: 0.6088, Test_Acc: 0.6667

