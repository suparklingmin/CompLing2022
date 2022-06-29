# 컴퓨터언어학 (2022학년도 1학기, 서울대학교 인문대학 언어학과)

## 개요

+ 강좌명: 컴퓨터언어학
+ 강좌번호: 108.413A (001 & 002)
+ 교수자: 박수지 (mam3b@snu.ac.kr, sut.i.palatalised@gmail.com)
+ 교재: Jurafsky & Martin, *[Speech and Language Processing 3rd edition](https://web.stanford.edu/~jurafsky/slp3/)*
+ 선수과목: [언어와 컴퓨터(2021년)](https://github.com/suzisuti/lecture/tree/master/2021/LC)

## 목표

컴퓨터언어학(108.413A)에서는 심층학습(딥러닝, 인공신경망)을 사용한 자연어처리 기법을 소개한다. 이 과목을 이수함으로써 수강생들은 여러 가지 인공신경망 모형들의 개략적인 원리를 이해하고 자연어처리 분야의 구체적인 과제를 해결하는 데 이를 활용할 수 있게 될 것이다. 이 강좌의 전반부에서는 로지스틱 회귀분석에 대한 이해를 바탕으로 순방향신경망(FFNN)과 합성곱신경망(CNN)에 대해 배우며, 이를 통해 주어진 영화평이 긍정적인지 부정적인지를 자동으로 분류하는 모형을 pytorch로 구현한다. 후반부에서는 순환신경망(RNN)을 응용하여 단어 사이의 문맥 정보를 포착하는 기법을 알아보고, 현재 다양한 자연어처리 과제에서 좋은 성능을 보이는 BERT와 GPT-2,3 등의 사전학습 언어모형(PLM)을 활용하는 방법을 익힌다.

## 일정

|회차|날짜|제목|슬라이드|실습|읽기 자료|
|--|--|--|--|--|--|
|1강|2022-03-02(수)|강의 소개|-|-|-|
|2강|2022-03-07(월)|NumPy 실습: 행렬과 벡터|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/02-20220307.pdf)]|-|[[밑바닥부터 시작하는 데이터 사이언스: 선형대수](https://github.com/insight-book/data-science-from-scratch/blob/master/scratch/linear_algebra.py)]|
|3강|2022-03-14(월)|로지스틱 회귀분석 (1)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/03-20220314.pdf)]|-|[[SLP3 Ch. 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf)]|
|4강|2022-03-16(수)|로지스틱 회귀분석 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/04-20220316.pdf)]|-|[[SLP3 Ch. 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf)]|
|5강|2022-03-21(월)|벡터의미론과 임베딩 (1)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/05-20220321.pdf)]|-|[[SLP3 Ch. 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf)]|
|6강|2022-03-23(수)|벡터의미론과 임베딩 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/06-20220323.pdf)]|-|[[SLP3 Ch. 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf)]<br>[[Demo: Korean Word2Vec](https://word2vec.kr/search/)]<br>[[Social impacts & bias of AI](https://kyunghyuncho.me/social-impacts-bias-of-ai/)]|
|7강|2022-03-28(월)|신경망 언어 모형 (1)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/07-20220328.pdf)]|-|[[SLP3 Ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)]|
|8강|2022-03-30(수)|신경망 언어 모형 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/08-20220330.pdf)]|-|[[SLP3 Ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)]|
|9강|2022-04-04(월)|신경망 언어 모형 (3)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/09-20220404.pdf)]|-|[[SLP3 Ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)]|
|10강|2022-04-06(수)|신경망 언어 모형 (4)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/10-20220406.pdf)]|-|[[SLP3 Ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)]|
|11강|2022-04-11(월)|합성곱 신경망 (1)|-|-|[[CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)]|
|12강|2022-04-13(수)|합성곱 신경망 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/12-20220413.pdf)]|-|[[Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)]|
|13강|2022-04-20(수)|PyTorch 실습: 순방향 신경망 훈련|-|[[Colab](https://colab.research.google.com/drive/1EYMWmo3oeRIsBdAXUYJ7itgUgPykeYlk?usp=sharing)]|-|
|14강|2022-04-25(월)|시퀀스 처리를 위한 딥러닝 (1)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/14-20220425.pdf)]|-|[[SLP3 Ch. 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf)]|
|15강|2022-04-27(수)|시퀀스 처리를 위한 딥러닝 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/15-20220427.pdf)]|-|[[SLP3 Ch. 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf)]|
|16강|2022-05-02(월)|시퀀스 처리를 위한 딥러닝 (3)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/16-20220502.pdf)]|-|[[SLP3 Ch. 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf)]|
|17강|2022-05-04(수)|PyTorch 실습: 순환 신경망 훈련|-|[[Colab](https://colab.research.google.com/drive/18heoB0yM4zMovZNDwxhGgKg9_HTCgPZV?usp=sharing)]|-|
|18강|2022-05-09(월)|기계번역과 부호화기-복호화기 모형 (1)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/18-20220509.pdf)]|-|[[SLP3 Ch. 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf)]|
|19강|2022-05-16(월)|기계번역과 부호화기-복호화기 모형 (2)|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/19-20220511.pdf)]|[[Colab](https://colab.research.google.com/drive/1wyTKSU3IbDkjB5aeInQYKfnCZiuBsp8d?usp=sharing)]|[[SLP3 Ch. 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf)]<br>[[KoNLPy: 형태소 분석 및 품사 태깅](https://konlpy.org/ko/latest/morph/)]|
|20강|2022-05-18(수)|PyTorch 실습: 부호화기-복호화기 모형 훈련|-|[[Colab](https://colab.research.google.com/drive/1T-WMtRWcfmMwEtD8B6QkJJGJQRhrJ35E?usp=sharing)]|[[NLP FROM SCRATCH](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)]|
|21강|2022-05-23(월)|PyTorch 실습: 부호화기-복호화기 모형 훈련 (계속)|-|[[Colab](https://colab.research.google.com/drive/1T-WMtRWcfmMwEtD8B6QkJJGJQRhrJ35E?usp=sharing)]|-|
|22강|2022-05-25(수)|셀프 어텐션 계층과 트랜스포머 소개|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/22-20220525.pdf)]|-|[[Attention Is All You Need](https://arxiv.org/abs/1706.03762)]|
|23강|2022-05-30(월)|BERT와 문맥 임베딩|[[Slides](https://github.com/suzisuti/CompLing2022/blob/main/slides/23-20220530.pdf)]|-|[[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)]|
|24강|2022-06-08(수)|Hugging Face 실습: transformers|-|[[Colab](https://colab.research.google.com/drive/1_XpQ5HoGnfs9Ikyd_1jKf0s8-sJ4dQ9C?usp=sharing)]|[[Hugging Face: Transformers: Pipelines for inference](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial)]<br>[[Hugging Face: Models](https://huggingface.co/models)]|
