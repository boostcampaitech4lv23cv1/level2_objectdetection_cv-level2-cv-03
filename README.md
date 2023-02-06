![header](https://capsule-render.vercel.app/api?type=rect&color=gradient&text=재활용%20품목%20분류를%20위한%20Object%20Detection&fontSize=30)

# Members
김도윤 - 2 stage model(detector, feature extractor 위주의 실험 진행_faster, cascade, htc, etc.), ensemble(wbf), Hyperparameter tuning
김윤호 - Augmentation(Auto-augmentation, Mosaic, Multi-Scale) 실험, 2 stage model(ATSSDyhead, cascade rcnn), StratifiedGroupKfold 구현
김종해 - 1 stage model (RetinaNet, yolov7) 실험, StratifiedGroupKfold 구현, WeightedBoxesFusion 실험
조재효 - Augmentation(TTA, albumentation), 1stage model (yolov7), 2 stage model(cascade_swin_b), hyperparameter tuning, k-fold, ensemble(wbf)
허진녕 - EDA, 1 stage model (yolov3, yolof, yolox) 실험, hyperparameter tuning(atss_dyhead), kfold, ensemble(wbf)

# 프로젝트 개요
![image](https://user-images.githubusercontent.com/39187226/216992364-4e56b8aa-f99c-402a-be29-db7ca5b35313.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나이다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 한다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋을 사용한다.

이를 이용하여 학습 시킨 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것이다.

우리는 수많은 쓰레기를 배출하면서 지구의 환경파괴, 야생동물의 생계 위협 등 여러 문제를 겪고 있습니다. 이러한 문제는 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능합니다.

제공되는 이 데이터셋은 위의 기술을 뒷받침하는 쓰레기를 판별하는 모델을 학습할 수 있게 해줍니다.

전체 이미지 개수 : 9754장 (train: 4883장 / test: 4871장)
Class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
Image size : (1024, 1024)

&nbsp;

# 프로젝트 수행 절차
<h3> 1. 데이터 EDA  </h3>
<h3> 2. 모델 탐색  </h3>
<h3> 3. Augmentation 기법 탐색 & K-fold validation을 위한 Cross Validation 전략 수립/h3>
<h3> 4. Ensemble & Hyperparameter Tuning  </h3>


# 문제정의
<h3> 1. 클래스의 분포의 불균형 </h3>  
- EDA 결과, 각각의 클래스 별 분포가 균일하지 않아, 전체 데이터셋에서 train set과 validation set 분리 시 클래스 분포에 맞게 나눌 필요가 있다.
  
<h3> 2. 성능 향상을 위한 Model Ensemble 전략 수립 </h3>
- 2개 이상의 모델을 통해 예측한 bounding box 중 더 ground truth에 맞는 bbox를 만드는 방법이 필요하다. mAP를 계산함에 있어서는 정답과 유사한 bbox가 많을 수록 좋은 점수를 도출해내므로 정답에 가까운 bbox를 더 정확하게 만드는 ensemble 전략이 필요했다.
  
&nbsp;

# 모델 및 Data Augmentation
- Cascade_RCNN_SwinT
  - Resize (size=[(256, 256), (512, 512), (768, 768), (1024, 1024)])
  - HorizontalFlip
  - VerticalFlip
  - ShiftScaleRotate
  - Blur or MedianBlur
  - RGBShift or HueSaturationValue or RandomBrightnessContrast or ChannelShuffle
  - Normalize
  - Pad (size=(512, 512), pad_val=0)
   
- ATSS_Dyhead_SwinT
  - Resize (size=[(256, 256), (512, 512), (768, 768), (1024, 1024)])
  - HorizontalFlip
  - VerticalFlip
  - ShiftScaleRotate
  - Blur or MedianBlur
  - RGBShift or HueSaturationValue or RandomBrightnessContrast or ChannelShuffle
  - Normalize
  - Pad (size=(512, 512), pad_val=0)

- YOLOv7
  - Mosaic
  - Mixup
  - random_perspective
  - augment_hsv
  - cutout
  
	
&nbsp;  
  
# Advanced Techniques
<h3> 1. Stratified Group K-fold</h3>  

- train set과 validation set을 나눌때 단순히 annotation을 기준으로 Stratified K-fold를 적용할 경우 한 이미지에 있는 여러 개의 bbox들이 나누어 지는 현상이 발생하여 학습 시 누락되는 현상이 발생하였다. 따라서 이미지 단위로 중복되지 않도록 하기 위해 Stratified Group K-fold를 적용하여 해당 문제를 해결하였다.

<h3> 2. Weighted Box Fusion   </h3>  

- 전반적인 성능이 비슷한 3개의 모델을 바탕으로 가장 정확한 예측 bbox를 만들기 위해 각 모델 별 confusion matrix를 통해 예측이 어려운 클래스를 잘 맞추는 모델에 더 가중치를 두어 WBF를 적용하여 mAP50 기준 기존 성능 대비 약 8%의 성능향상을 확인하였다.

	
&nbsp;
# Reference
<a name="footnote_1">[1]</a>  : AIstage
