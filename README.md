## Out-of-Distribution Detection in Classifiers via Generation

### 실험 과정
#### 1. Variational AutoEncoder를 In-distribution dataset으로 학습한다.
<pre>
  - Data : In-distribution data (Train&Test)
  - Data setting : ./data/train/In-distribution data/train
                                                          / class 0 / *.png
                                                          / class 1 / *.png
                                                                 ...
                                                          / class n / *.png
                   ./data/train/In-distribution data/test
                                                          / class 0 / *.png
                                                          / class 1 / *.png
                                                                 ...
                                                          / class n / *.png
  - python train_vae.py --in_data './data/train/In-distribution'
  - output : vae model
</pre>
 
#### 2. 잘 학습된 Variational AutoEncoder를 이용해 OOD sample을 만든다.
<pre>
  - Data : 위와 동일
  - Data setting : 위와 동일
  - model : 1 번에서 학습된 VAE
  - python gen_ood.py --data './data/train/In-distribution' --dataset train --model_path './result'
  - output : OOD sample
</pre>

#### 3. 만들어진 OOD sample과 In-distribution dataset을 이용해 classifier을 학습한다.
<pre>
  - Data : In-distribution data (Train&Test) / Generated OOD samples (Train&Test)
  - Data setting : x
  - python train_classi.py --in_data './data/train/In-distribution'
  - output : classifier model
</pre>

#### 4. 앞서 학습된 classifier가 새로운 Out of distribution dataset도 잘 구분해 내는지 test 한다. 
<pre>
  - Data : In-distribution data (Test) / Out-of-distribution data (Test)
  - Data setting : ./data/test/Out-of-distribution data/test
                                                            / class 0 / *.png
                                                            / class 1 / *.png
                                                                   ...
                                                            / class n / *.png
  - model : 3 번에서 학습된 classifier
  - python test.py --in_data './data/train/In-distribution' --out_data './data/test/Out-of-distribution data' 
                   --model_path './result'
  - output : AUROC
</pre>
-----------------------------------------------------------------------------------------------------------
### Example
#### Train a vae model
<pre>
<code>
python train_vae.py --in_data './data/train/mnist' --save_path './result' --epochs 100 --batch_size 128 --opt rmsprop --lr 0.001 --momentum 0.0 --gpu_id 0 --seed False
</code>

- 저장파일 : ./result/vae/model.pth
                         /loss_curve.png
                         /train_loss.log
                         /vali_loss.log
                         /configuration.json
</pre>


#### Generate OOD Sample
<pre>
<code>
python gen_ood.py --data './data/train/mnist' --dataset train --model_epoch 100 --model_path './result' --batch_size 1000 --gpu_id 0
</code>

- 저장파일 : ./data/train/mnist/ood/train/*.png
</pre>


#### Train a Classifier model
<pre>
<code>
python train_classi.py --in_data './data/train/mnist' --save_path './result' --epochs 200 --batch_size 256 --opt ada --lr 0.05 --momentum 0.0 --gpu_id 0 --seed False
</code>

- 저장파일 : ./result/classifer/model.pth
                               /loss_curve.png
                               /accuracy_curve.png
                               /train_loss(accuracy).log
                               /vali_loss(accuracy).log
                               /configuration.json
</pre>

#### Test a model
<pre>
<code>
python test.py --in_data './data/train/mnist' --out_data './data/test/fmnist' --model_path './result' --model_epoch 200 --save_path './result' --gpu_id 0 --seed False
</code>

- 저장파일 : ./result/auroc.log
</pre>


-----------------------------------------------------------------------------------------------------

### In-Distribution Datasets
- MNIST
- FMNIST

### Out-Of-Distribution Datasets
- EMNIST
- FMNIST
- MNIST
- NotMNIST
- Gaussian-Noise
- Uniform-Noise
- Shpere-OOD

### Architecture
- Variational autoencoder
- Classifier

### Metric
- FPR at 95% TPR
- FPR at 80% TPR
- AUROC
- Detection Error
- AUPR(In, Out)

### Parameters
- latent dimension

### Training Setup
|Architecture|Variational autoencoder|Classifier|
|------|---|---|
|epochs|100|200|
|batch size|128|128|
|loss|Reconstruction loss + KL loss|CrossEntropy|
|optimizer|RMSprop|Adadelta|
|learning rate|0.001|0.01|
|weight_decay||0.001|

### Official Code
- <https://github.com/sverneka/OODGen>
