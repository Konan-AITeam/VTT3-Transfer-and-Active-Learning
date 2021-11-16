# Transfer Learning

### Datasets Preparation
[UODB benchmark](http://www.svcl.ucsd.edu/projects/universal-detection/)에서 데이터셋 설치 후 data 폴더로 이동

### prerequisites

* Python 3.6
* Pytorch 1.0.0
* CUDA 8.0 or higher

### Pretrained Model
Pretrained Model를 아래의 링크에서 설치 후 data/pretrained_model 폴더에 넣는다

* DA-50: [Dropbox](https://drive.google.com/file/d/1kddC55_eByFfMZqDTM9cLj0j1BiHBq9D/view?usp=sharing)
* ResNet50: [Dropbox](https://drive.google.com/file/d/1_0wFe2soxLkyP5DCCpOJddp1k_xcowv-/view?usp=sharing)


### Compilation

1. 가상환경을 만들고 활성화

```
conda create -n uodb python=3.6 -y
conda activate uodb
```

2. pip로 필요한 패키지 설치
```
pip install -r requirements.txt --user
```

3. 콘다로 pytorch 1.0.0 버전 설치
```
conda install pytorch=1.0.0 torchvision cuda100 -c pytorch
```


4. CUDA 관련 코드를 아래의 스크립트로 컴파일

```
cd lib
python setup.py build develop
python setup_tools.py build_ext --inplace
```

