# LegalBot: 법률 특화 RAG 기반 이중 언어 하이브리드 트랜스포머 챗봇
> **직접 구현한 Transformer 아키텍처와 Pre-trained 임베딩을 결합한 법률 도메인 특화 QA 시스템**

---

## 📌 프로젝트 개요 (Overview)
본 프로젝트는 '근로기준법' 및 '저작권법' 분야의 복잡한 법률 조문과 대법원 판례 데이터를 바탕으로 신뢰성 높은 답변을 제공하는 **법률 특화 AI 챗봇 서비스**입니다. 

단순히 외부 LLM API를 호출하는 방식에서 벗어나, **Transformer 전체 구조(Encoder, Decoder, Multi-Head Attention)를 PyTorch로 직접 스크래치 구현**하고, 소규모 데이터셋의 한계를 극복하기 위해 **Pre-trained 모델(KoGPT2 / DialoGPT)의 임베딩 레이어를 결합한 하이브리드 구조**를 채택했습니다. 

더불어, 학습 데이터 외의 질의에도 유연하고 정확하게 대응하기 위해 **FAISS 기반의 RAG(Retrieval-Augmented Generation) 파이프라인**과 **이중 언어(한국어/영어) 자동 감지 시스템**을 구축하고, **Streamlit**을 통해 직관적인 UI/UX로 서비스화했습니다.

---

## 🔧 주요 특징 (Key Features)

- **하이브리드 트랜스포머 직접 구현 (Scratch-built Transformer)**
  - Encoder, Decoder, MultiHeadAttention, Positional Encoding 등 핵심 레이어 직접 빌드.
  - 사전 학습된 `skt/kogpt2-base-v2`(한국어) 및 `microsoft/DialoGPT-medium`(영어)의 단어 의미 공간(Embedding)을 이식하여 540여 쌍의 소규모 도메인 데이터로도 고품질 문장 생성 가능.
- **활성화 함수(Activation Function) 이원화 실험**
  - `ReLU` 기반 모델(`relu_model.py`)과 `GELU` 기반 모델(`gelu_model.py`)을 각각 구축하여 학습 수렴 속도 및 생성 텍스트의 자연스러움 비교 분석.
- **2단계 하이브리드 RAG 검색 파이프라인**
  - **1단계 (Exact/Semantic Match):** 구축된 고품질 Q&A 데이터셋에서 쿼리 매칭 시도.
  - **2단계 (RAG Fallback):** 매칭 데이터가 없을 경우, `FAISS` 벡터 DB에서 정규식 기반 청킹된 법령/판례 원문을 실시간 검색하여 컨텍스트로 주입.
- **이중 언어(Bilingual) 지원**
  - 입력 문장의 유니코드 및 텍스트 패턴을 분석하여 한국어/영어를 자동 감지(`detect_language`). 해당 언어의 맞춤형 모델 및 토크나이저 바인딩 플로우 자동 전환.
- **사용자 중심 UI/UX (Streamlit)**
  - RAG 작동 시 실시간 검색된 법률 원문의 '핵심 조문' 카드 시각화 및 '참조 조문 전체 보기(Expander)' 기능 제공으로 법적 근거 투명성 확보.

---

## 🧩 개발 환경 및 핵심 라이브러리 (Environment & Tools)

- **OS**: Windows 10 
- **Language & Environment**: Python 3.8+ / COLAB
- **Core Libraries**:
  - `torch==2.x` (Transformer 아키텍처 구현 및 학습)
  - `transformers` (KoGPT2, DialoGPT 토크나이저 및 임베딩 로드)
  - `faiss-cpu` (고속 코사인 유사도 검색 기반 벡터 인덱싱)
  - `sentence-transformers` (jhgan/ko-sroberta-multitask 임베딩 활용)
  - `streamlit` (웹 애플리케이션 UI 구현)
  - `pickle`, `numpy`, `pandas`

---

## 🚀 [최종 파이프라인] 데이터 수집부터 서비스까지의 과정

### Step 1. 도메인 특화 데이터 수집 및 전처리
- **데이터 소스 확보:** 국가법령정보센터 등을 통해 '근로기준법', '저작권법' 핵심 법령 조문과 관련 '대법원 판례' 데이터를 수집.
- **데이터 정제(Cleansing):** Raw 텍스트에서 불필요한 공백, 특수문자, 비정상 인코딩을 제거하는 `clean_text` 파이프라인 구축.

### Step 2. RAG 전용 벡터 DB 구축 및 청킹 최적화
- **의미 기반 조항 청킹(Semantic Chunking):** 정규식(`(?=제\s*\d+조)`)을 활용하여 기계적 글자 수 분할이 아닌 의미론적 '법 조항 단위'로 분리. 조항이 길 경우 `textwrap`을 활용해 문장 단절을 방지(최대 300~400자 제한).
- **출처 메타데이터 주입:** 분할된 청크 전면에 `[LAW_근로기준법 제O조]`, `[PREC_total]` 등 명시적 태그를 부착하여 모델이 명확한 근거를 인지하도록 직렬화.
- **벡터 인덱싱:** `ko-sroberta-multitask` 모델로 문장을 벡터화한 후 코사인 유사도(`FAISS IndexFlatIP`) 기반 인덱스(`law_knowledge.index`)와 메타데이터(`metadata.pkl`) 매핑 데이터 생성.

### Step 3. 맞춤형 QA 데이터셋 구축 (Custom QA Dataset)
- **시나리오 기반 데이터 생성:** 실무 및 일상에서 발생할 수 있는 주요 시나리오("연차 유급휴가 조건", "부당해고 기준", "폰트 파일 무단 배포" 등)를 상정하여 한국어(`legal_qa_dataset.txt`) 및 영어(`legal_qa_dataset_en.txt`) 각 540여 쌍의 고품질 데이터셋 직접 구축.
- **역할:** 모델 학습용 지도 데이터(Fine-tuning Target) 겸 RAG 파이프라인 무결성을 검증하는 Ground Truth 세트로 활용.

### Step 4. 하이브리드 트랜스포머 모델 학습 (Model Training)
- Pre-trained 임베딩 고정 후 직접 구현한 Transformer Layer 학습 진행.
- 과적합 방지를 위해 `EarlyStopping` (Patience, Min Delta 관리) 및 `Warmup 레이트 스케줄러` 적용.
- 활성화 함수 실험을 통해 최종 수렴도와 자연스러운 문장 생성이 검증된 가중치 파일을 `models_v3_stable/`에 저장.
#### Activation Function 개선 실험 결과
<img width="940" height="335" alt="image" src="https://github.com/user-attachments/assets/696ad7c0-431f-42ac-9d09-0bcb9b7e0758" />
- Korean / English Model Train Loss 분석: Baseline(ReLU) 대비 Proposed(GELU) 구조가 초기 에포크에서 더 안정적이고 빠른 Loss 감소 추세를 보임을 확인.


### Step 5. Streamlit 통합 및 UI/UX 구현
- `app.py`를 실행하여 통합 웹 인터페이스 구동.
- 사용자 입력에 대응하는 언어 자동 판별 ➡️ 하이브리드 검색(QA 셋 검증 후 RAG 폴백) ➡️ 직접 구현한 트랜스포머 디코더를 통한 답변 생성 및 근거 조문 시각화 유기적 연동.

---

## 📂 프로젝트 디렉토리 구조 (Directory Structure)

```text
├── program/
│   ├── relu_model.py          # ReLU 활성화 함수 기반 스크래치 트랜스포머 및 학습 코드
│   └── gelu_model.py          # GELU 활성화 함수 기반 스크래치 트랜스포머 및 학습 코드
├── data/
│   ├── legal_qa_dataset.txt      # 구축된 한국어 법률 QA 데이터셋 (540쌍)
│   └── legal_qa_dataset_en.txt   # 구축된 영어 법률 QA 데이터셋
├── index/
│   ├── law_knowledge.index    # 법령 및 판례 원문 FAISS 벡터 인덱스 파일
│   └── metadata.pkl           # 인덱스 매칭용 법령 조문 메타데이터
├── models_v3_stable/
│   └── 2_bundle_[lang].pkl    # relu 모델로 학습 완료된 하이브리드 트랜스포머 모델 번들 
├── app.py                     # Streamlit 기반 실시간 웹 서비스 구동 스크립
└── README.md                  # 프로젝트 설명 문서
```
---


## 💻 실행 방법 (How to Run)
### 1. 의존성 라이브러리 설치
```bash
pip install torch transformers sentence-transformers faiss-cpu streamlit pandas numpy
```


### 2. 모델 학습
```
# ReLU 모델 또는 GELU 모델 선택 학습
python program/relu_model.py
```

### 3. Streamlit 웹 서비스 구동
```
streamlit run app.py
```

---


## 서비스 UI/UX 및 주요 기능 데모 (Service Interface)
<img width="600" height="339" alt="image" src="https://github.com/user-attachments/assets/05bdb476-14b3-400c-a18b-e053f98072fc" />
<img width="600" height="339" alt="image" src="https://github.com/user-attachments/assets/a7c94ce7-1a95-494b-a416-b52638af9796" />
<img width="600" height="339" alt="image" src="https://github.com/user-attachments/assets/0a6ef4d5-4b2b-4075-ba88-cb5b487dfa30" />


### 1) 메인 화면 및 다국어 감지
- 서비스 접속 시 깔끔한 챗봇 인터페이스와 함께 사용자가 입력하는 언어(국문/영문)를 자동 감지하는 배너가 활성화됩니다.

### 2) 2단계 하이브리드 답변 및 RAG 구동 (핵심 기능)
- **QA 데이터셋 매칭:** 구축된 540쌍의 데이터셋에 존재하는 질문 입력 시, 트랜스포머 모델이 정제된 정답 답변을 즉시 출력합니다.
- **RAG Fallback 및 조문 시각화:** 데이터셋에 없는 새로운 법률 질문이 들어오면, 벡터 DB(FAISS) 연동을 통해 실시간으로 RAG 지식을 검색합니다. 
  - 상단에 RAG 구동 안내 메시지가 노출됩니다.
  - **[핵심 조문]** 카드가 상단에 시각화됩니다.
  - **[참조 조문 전체 보기]** 익스팬더(Expander) 버튼을 누르면 실시간 검색된 전체 RAG 컨텍스트가 펼쳐지며 투명하게 법적 근거를 제시합니다.

### 3) 스마트 멀티 대화방 관리 (사이드바 UI)
- **자동 넘버링 및 활성화 표시:** '새 대화 만들기' 버튼 클릭 시 자동으로 다음 숫자 대화방(예: 대화 1, 대화 2...)이 생성되며, 현재 선택된 방은 검은 동그라미(●) 기호로 활성화 상태가 표시됩니다.
- **대화방 삭제 기능:** 사이드바 목록의 대화방 항목에 마우스를 호버(Hover)하면 삭제 버튼(X)이 활성화되어 사용하지 않는 채팅방을 유연하게 정리할 수 있습니다.

