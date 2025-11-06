# **DocuMate**  
> **Docu**ment + **Mate** - 공식 문서를 기반으로 학습을 돕는 AI 챗봇 / **프로젝트 기간:** 2025.10.24 ~ 2025.11.06


<img width="800" alt="스크린샷 2025-11-06 오후 4 04 32" src="https://github.com/user-attachments/assets/6d0714aa-b086-49e6-aee6-e819eeabbd59" />

---

## 🔷 1. 서비스 구성 요소  

### 🔸 1.1 주요 기능  

| 기능 및 도구명 | 설명 |
|---|---|
| 멀티턴 | 한 세션 안에서의 대화 내용을 기억해 대화의 흐름을 자연스럽게 도움 |
| 웹 검색 | 공식 문서 사이트 기반으로 웹 검색을 통해 답변 생성 |
| 로컬 및 업로드 <br> 파일 검색 | 로컬 노트북이나 사용자가 업로드 한 파일에서 <br>유사한 내용을 검색하고 응답에 반영하여 답변 생성 |
| 파일 저장 | 답변을 파일로 저장하여 사용자에게 제공하고, 저장 결과를 응답에 반영하여 답변 생성 |
| Slack 연동 | LLM이 생성한 답변을 슬랙으로 전송 |

### 🔸 1.2 프로젝트 디렉토리 구조
```
├── graph.png
├── uv.lock
├── pyproject.toml
├── README.md
├── script
│   ├── start_services.sh
│   └── stop_services.sh
├── src
│   ├── main.py
│   ├── agent_manager.py
│   ├── agent_state.py
│   ├── edge.py
│   ├── graph_builder.py
│   ├── make_graph.py
│   ├── llm.py
│   ├── node.py
│   ├── prompts.py
│   ├── rag_build.py
│   ├── tools.py
│   ├── upload_helpers.py
│   ├── baseline_code.py
│   ├── util
│   │   └── util.py
│   └── web
│       ├── main.py
│       ├── schemas.py
│       └── streamlit_app.py
└── uploads
```

### 🔸 1.3 사용자 흐름  

**Streamlit > 빠른 시작 예시 제공**  
&emsp; `pandas merge 사용법 알려줘. 공식 문서 기준으로 설명해줘.`&nbsp;&nbsp;&nbsp;`matplotlib에서 pie 차트 옵션 정리해줘.`  
&emsp; `이전 노트북에서 matplotlib histplot을 어떻게 썼는지 예제 코드 보여줘.`&nbsp;&nbsp;&nbsp;`업로드한 .ipynb 안에 있는 pandas concat 예제를 찾아줘.`  
&emsp; `pandas concat 기본 사용법을 설명하고, 내 노트북에서 실제로 사용한 예제도 함께 보여줘.`

**사용자 시나리오 예시**  
| **case 1. 특정 기술 스택에 관련된 내용 질의** | **case 2. 업로드 파일 기반 검색 질의** | **case 3. 이전 답변 내용 파일 저장 요청** |
|---|---|---|
| <img width="400" alt="스크린샷 2025-11-06 오후 3 38 50" src="https://github.com/user-attachments/assets/8f042bb8-da01-4486-8b7e-c6e6c94db5d0" /> | <img width="400" alt="스크린샷 2025-11-06 오후 3 39 28" src="https://github.com/user-attachments/assets/83b76397-4b91-462b-8b6a-71738db074f8" /> | <img width="400" alt="스크린샷 2025-11-06 오후 3 49 06" src="https://github.com/user-attachments/assets/7f8780be-434a-4259-bbab-bedaa04e4439" /> |


---

## 🔷 2. 협업 툴 

### 🔸 협업 툴 
- **소스 관리:** GitHub  
- **커뮤니케이션:** Slack  
- **버전 관리:** Git  

---

## 🔷 3. 서비스 아키텍처 
### 🔸 Agent 구조도  
<img width="800" alt="agent" src="https://github.com/user-attachments/assets/1c2bba5d-0c48-4223-ba63-d91bb6919a0d" />

---

## 🔷 4. 사용 기술 스택  
| 서비스 영역 | 기술 스택 |
|---|---|
| Agent | Langchain, LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## 🔷 5. 팀원 소개  

| 이름      | 역할              | GitHub                               | 담당 기능                                 |
|----------|------------------|-------------------------------------|-----------------------------------------|
| **장윤정** | 팀장/웹 서비스 개발 | [GitHub 링크](https://github.com/yjjang06)             | 서버 구축, Backend/Frontend 개발            |
| **송인섭** | 웹 서비스 개발  | [GitHub 링크](https://github.com/SongInseob)             | Backend/Frontend 개발             |
| **김명철** | Agent 개발    | [GitHub 링크](https://github.com/qpwpep)             | LangGraph 기반 Agent 구축, 웹검색 및 슬랙 메시지 발송 도구 개발 |
| **김상윤** | Agent 개발    | [GitHub 링크](https://github.com/94KSY)             | 멀티턴 기능 개발   |
| **정소현** | Agent 개발    | [GitHub 링크](https://github.com/soniajhung)             | 파일 저장 및 노트북 파일 기반 RAG 검색 도구 개발   |

---

## 🔷 6. Appendix  
### 🔸 6.1 참고 자료  
- [LangChain 공식 문서](https://docs.langchain.com/oss/python/langchain/overview)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/ko/)


### 🔸 6.2 설치 및 실행 방법  
**1. 필수 라이브러리 설치:**  
- ```bash
  uv pip install -e .  # pyproject.toml 기반 설치
  ```

**2. 서버 실행/종료:**  
- **CLI**
  ```bash
  uv run python -m new_src.main
  ```
    
- **WebService**
  ```bash
  uv run python -m src.main --mode startweb
  uv run python -m src.main --mode stopweb
  ```

**3. 웹페이지 접속:**  
- ```
  http://localhost:8501
  ```

### 🔸 6.3 발표 자료
[발표 자료 링크](https://github.com/AIBootcamp14/langchainproject-new-langchainproject_3/blob/main/docs/Langchain_Project_Team_3.pdf)

