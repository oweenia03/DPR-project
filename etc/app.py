import streamlit as st
import torch as _torch
import os
import sys
import pickle
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ==========================================
# 0. 경로 설정 및 모듈 주입
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
program_dir = os.path.join(os.path.dirname(current_dir), "program")
if program_dir not in sys.path:
    sys.path.append(program_dir)

from relu_model import (
    TransformerScratch, PretrainedVocab, LegalRetriever,
    prepare_tokenizer, PRETRAINED_KO, PRETRAINED_EN,
    clean_text, detect_language, retrieve_answer
)

# ==========================================
# 1. torch.load 패치
# ==========================================
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    if not _torch.cuda.is_available():
        kwargs["map_location"] = _torch.device('cpu')
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

# ==========================================
# 2. RAG 파싱
# ==========================================
def generate_response_only(model, vocab, question, retriever, max_len=256, dev='cpu', lang='ko'):
    model.eval()
    return retriever.retrieve(question, max_len=1000) if retriever else ""

def parse_rag_context(context):
    if not context:
        return "", ""
    tag_matches = list(re.finditer(r'(\[(?:LAW|PREC)_[^\]]+\])', context))
    if len(tag_matches) >= 2:
        primary = context[:tag_matches[1].start()].strip()
        return primary, context.strip()
    return context.strip(), context.strip()

# ==========================================
# 3. 모델 로드
# ==========================================
@st.cache_resource
def load_all_resources():
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(current_dir)
    retriever = LegalRetriever(
        index_path=os.path.join(base_dir, "law_knowledge.index"),
        metadata_path=os.path.join(base_dir, "metadata.pkl")
    )
    models, vocabs = {}, {}
    bundle_paths = {
        'ko': os.path.join(base_dir, "transformer", "models_v3_stable", "2_bundle_ko_ep00200_acc043.pkl"),
        'en': os.path.join(base_dir, "transformer", "models_v3_stable", "2_bundle_en_ep00200_acc036.pkl")
    }
    for suffix, lang in [("ko_ep01000_acc043", "ko"), ("en_ep01000_acc036", "en")]:
        p = os.path.join(base_dir, "transformer", "models_v3_stable", f"2_bundle_{suffix}.pkl")
        if os.path.exists(p): bundle_paths[lang] = p
    for lang in ['ko', 'en']:
        path = bundle_paths[lang]
        if not os.path.exists(path):
            continue
        tokenizer = prepare_tokenizer(PRETRAINED_KO if lang == 'ko' else PRETRAINED_EN)
        vocab = PretrainedVocab(tokenizer, lang)
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        model = TransformerScratch(
            vocab_size=len(vocab), src_pad_idx=vocab.pad_token_id,
            emb_size=768, n_layers=4, heads=8, ff_exp=4, dropout=0.1,
            max_seq_len=bundle['max_seq'], device=device
        ).to(device)
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        models[lang] = model
        vocabs[lang] = vocab
    return models, vocabs, retriever, device

@st.cache_resource
def load_qa_pairs():
    base_dir = os.path.dirname(current_dir)
    paths = {
        'ko': os.path.join(base_dir, "transformer", "models_v3_stable", "2_bundle_ko_ep00200_acc043.pkl"),
        'en': os.path.join(base_dir, "transformer", "models_v3_stable", "2_bundle_en_ep00200_acc036.pkl"),
    }
    for suffix, lang in [("ko_ep01000_acc043", "ko"), ("en_ep01000_acc036", "en")]:
        p = os.path.join(base_dir, "transformer", "models_v3_stable", f"2_bundle_{suffix}.pkl")
        if os.path.exists(p): paths[lang] = p
    qa = {}
    for lang, path in paths.items():
        if not os.path.exists(path):
            qa[lang] = []; continue
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        qa[lang] = bundle.get('pairs', [])
    return qa

# ==========================================
# 4. 페이지 설정 + CSS
# ==========================================
st.set_page_config(page_title="Legal AI", layout="wide", page_icon="⚖️",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Noto+Sans+KR:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans KR', sans-serif !important;
}
.stApp { background: #F8F8F6; color: #1A1A1A; }

/* ── Header (Deploy bar) ── */
[data-testid="stHeader"] {
    background: #F8F8F6 !important;
    border-bottom: 1px solid #E8E8E4 !important;
}

/* ── Sidebar close/open 버튼 — SVG 아이콘 교체 ── */
/* open 버튼: stSidebarCollapsedControl (사이드바 닫혔을 때 헤더 영역) */
/* close 버튼: stSidebar 내부 button[kind="header"] / stBaseButton-headerNoPadding */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] > button,
button[data-testid="stBaseButton-headerNoPadding"],
[aria-label="Close sidebar"],
[aria-label="Open sidebar"],
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stHeader"] button {
    opacity: 1 !important;
    visibility: visible !important;
    background: transparent !important;
    border-radius: 6px !important;
    transition: background 0.15s !important;
    z-index: 999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
}

/* Material Icons 텍스트 (keyboard_double_arrow_right/left) 완전 숨김 */
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stSidebarCollapsedControl"] > button span,
button[data-testid="stBaseButton-headerNoPadding"] span,
[aria-label="Close sidebar"] span,
[aria-label="Open sidebar"] span,
[data-testid="stSidebar"] button[kind="header"] span,
[data-testid="stHeader"] button span {
    font-size: 0 !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    display: none !important;
}

/* SVG 아이콘 주입 — open/close 동일 패널 아이콘 */
[data-testid="stSidebarCollapsedControl"]::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="stSidebarCollapsedControl"] > button::after,
button[data-testid="stBaseButton-headerNoPadding"]::after,
[aria-label="Close sidebar"]::after,
[aria-label="Open sidebar"]::after,
[data-testid="stSidebar"] button[kind="header"]::after,
[data-testid="stHeader"] button::after {
    content: "" !important;
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUExQTFBIiBzdHJva2Utd2lkdGg9IjEuOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4KICA8cmVjdCB4PSIzIiB5PSIzIiB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHJ4PSIyIi8+CiAgPGxpbmUgeDE9IjkiIHkxPSIzIiB4Mj0iOSIgeTI9IjIxIi8+Cjwvc3ZnPg==") !important;
    background-repeat: no-repeat !important;
    background-size: contain !important;
    background-position: center !important;
    opacity: 0.5 !important;
    visibility: visible !important;
    transition: opacity 0.15s !important;
    flex-shrink: 0 !important;
}

/* hover 시 선명하게 */
[data-testid="stSidebarCollapsedControl"]:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] > button:hover::after,
button[data-testid="stBaseButton-headerNoPadding"]:hover::after,
[aria-label="Close sidebar"]:hover::after,
[aria-label="Open sidebar"]:hover::after,
[data-testid="stSidebar"] button[kind="header"]:hover::after,
[data-testid="stHeader"] button:hover::after {
    opacity: 0.9 !important;
}

/* ── Sidebar — 이미지 추출 색상 #F7F8F3 (웜 그린 톤) ── */
[data-testid="stSidebar"] {
    background: #F7F8F3 !important;
    border-right: 1px solid #E4E4E0 !important;
}

/* ★ 중요: 아이콘(Material Icons)은 제외하고 일반 텍스트에만 폰트 적용 */
[data-testid="stSidebar"] div, 
[data-testid="stSidebar"] span:not([class*="Icon"]),
[data-testid="stSidebar"] button:not([kind="header"]) { 
    font-family: 'Inter', 'Noto Sans KR', sans-serif !important; 
}

[data-testid="stSidebarNav"] { display: none; }

/* sidebar 내부 패딩 제거 */
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] .block-container { padding: 0 !important; max-width: 100% !important; }

/* 새 대화 버튼 */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: -0.1px !important;
    padding: 9px 14px !important;
    margin: 0 4px !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: #333 !important;
}

/* 대화방 선택 버튼 */
[data-testid="stSidebar"] .stHorizontalBlock .stButton > button {
    background: transparent !important;
    color: #555 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 7px 10px !important;
    text-align: left !important;
    transition: background 0.12s, color 0.12s !important;
    box-shadow: none !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stHorizontalBlock .stButton > button:hover {
    background: #EBEBEA !important;
    color: #1A1A1A !important;
}

/* 대화방 row — 좌우 패딩 확보 */
[data-testid="stSidebar"] .stHorizontalBlock {
    padding: 0 10px !important;
    gap: 4px !important;
    align-items: center !important;
}

/* 삭제 버튼 — 우측 정렬, 평소엔 투명 hover 시 표시 */
[data-testid="stSidebar"] .stHorizontalBlock > div:last-child {
    flex: 0 0 auto !important;
    width: 28px !important;
}
[data-testid="stSidebar"] .stHorizontalBlock > div:last-child .stButton > button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 26px !important;
    height: 26px !important;
    min-height: unset !important;
    padding: 0 !important;
    border-radius: 5px !important;
    color: transparent !important;
    font-size: 11px !important;
    background: transparent !important;
    transition: color 0.12s, background 0.12s !important;
}
[data-testid="stSidebar"] .stHorizontalBlock:hover > div:last-child .stButton > button {
    color: #AAAAAA !important;
}
[data-testid="stSidebar"] .stHorizontalBlock > div:last-child .stButton > button:hover {
    color: #1A1A1A !important;
    background: #E2E2DE !important;
}

/* ── Main area ── */
.block-container {
    padding-top: 0 !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 760px !important;
}

/* 채팅 입력창 너비 제한 — 사이드바 닫혀도 늘어나지 않게 */
[data-testid="stChatInputContainer"] {
    max-width: 700px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    padding: 2rem 1rem;
}
.welcome-icon {
    font-size: 32px;
    margin-bottom: 20px;
    opacity: 0.85;
}
.welcome-title {
    font-size: 26px;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: -0.6px;
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.welcome-sub {
    font-size: 14px;
    color: #999;
    margin: 0 0 32px 0;
    font-weight: 400;
    letter-spacing: 0;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 8px;
}
.chip {
    font-size: 12.5px;
    padding: 7px 14px;
    border-radius: 99px;
    border: 1px solid #E0E0DC;
    color: #555;
    background: #E8E8E8;
    cursor: default;
    letter-spacing: -0.1px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    max-width: 100% !important;
}
/* hide avatars */
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
    display: none !important;
}
/* user bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #EBEBEA;
    border-radius: 16px 16px 4px 16px;
    padding: 11px 16px;
    font-size: 14px;
    line-height: 1.65;
    color: #1A1A1A;
    max-width: 72%;
    margin-left: auto;
}
/* assistant text */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    font-size: 14px;
    line-height: 1.7;
    color: #2A2A2A;
}

/* ── RAG card ── */
.rag-label {
    font-size: 10.5px;
    font-weight: 600;
    color: #AAAAAA;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin: 12px 0 6px 0;
}
.rag-primary-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-left: 2.5px solid #1A1A1A;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 13.5px;
    line-height: 1.78;
    color: #2A2A2A;
    white-space: pre-wrap;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #EBEBEA !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12.5px !important;
    color: #888 !important;
    font-weight: 400 !important;
    padding: 9px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: #1A1A1A !important; }

.rag-full-context {
    max-height: 220px;
    overflow-y: auto;
    background: #F4F4F2;
    padding: 13px 15px;
    border-radius: 6px;
    font-size: 12.5px;
    color: #555;
    line-height: 1.75;
    white-space: pre-wrap;
}
.rag-full-context::-webkit-scrollbar { width: 3px; }
.rag-full-context::-webkit-scrollbar-thumb { background: #D8D8D4; border-radius: 3px; }

/* ── Chat input — 칩 버튼 색상과 동일한 톤 ── */
[data-testid="stChatInputContainer"] {
    background: #F8F8F6 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] {
    border: 1px solid #D8D8D4 !important;
    border-radius: 14px !important;
    background: #E8E8E8 !important;
    box-shadow: none !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #BBBBBA !important;
    background: #E4E4E4 !important;
    box-shadow: none !important;
}
/* 전송 버튼 */
[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    transition: background 0.15s !important;
}
[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputSubmitButton"]:hover {
    background: #333333 !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; border: 1px solid #E8E8E4 !important; font-size: 13px !important; }
div[data-testid="stAlert"] { background: #F4F4F2 !important; color: #444 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-size: 13px; color: #999; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #EBEBEA; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. 세션 상태 초기화
# ==========================================
if "chat_rooms" not in st.session_state:
    st.session_state.chat_rooms = {"01": []}
if "current_room" not in st.session_state:
    st.session_state.current_room = "01"
if "room_counter" not in st.session_state:
    st.session_state.room_counter = 1

with st.spinner("모델을 불러오는 중..."):
    models, vocabs, retriever, device = load_all_resources()
qa_pairs_dict = load_qa_pairs()

# ==========================================
# 6. 사이드바
# ==========================================
with st.sidebar:
    # 로고 + 설명
    st.markdown("""
    <div style="padding: 20px 16px 12px 16px; border-bottom: 1px solid #E4E4E0; margin-bottom: 10px;">
        <div style="font-size:15px; font-weight:600; color:#1A1A1A; letter-spacing:-0.3px;">⚖ Legal AI</div>
        <div style="font-size:11.5px; color:#BABAB6; margin-top:3px; letter-spacing:0;">법률 질의응답 시스템</div>
    </div>
    """, unsafe_allow_html=True)

    # 새 대화 버튼
    if st.button("＋  새 대화", use_container_width=True):
        st.session_state.room_counter += 1
        new_id = f"{st.session_state.room_counter:02d}"
        st.session_state.chat_rooms[new_id] = []
        st.session_state.current_room = new_id
        st.rerun()

    st.markdown("""
    <div style="padding: 12px 16px 4px 16px;">
        <span style="font-size:10.5px; font-weight:600; color:#BBBBBA; letter-spacing:0.6px; text-transform:uppercase;">대화 목록</span>
    </div>
    """, unsafe_allow_html=True)

    # 대화방 목록
    for rid in sorted(st.session_state.chat_rooms.keys()):
        is_active = rid == st.session_state.current_room
        col_sel, col_del = st.columns([5, 1])
        with col_sel:
            label = f"{'●' if is_active else '·'}  대화 {rid}"
            btn_style = "font-weight:600; color:#1A1A1A;" if is_active else ""
            if st.button(label, key=f"sel_{rid}", use_container_width=True):
                st.session_state.current_room = rid
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{rid}", use_container_width=True):
                if len(st.session_state.chat_rooms) <= 1:
                    st.toast("마지막 대화방은 삭제할 수 없습니다.")
                else:
                    del st.session_state.chat_rooms[rid]
                    if st.session_state.current_room == rid:
                        remaining = sorted(st.session_state.chat_rooms.keys())
                        st.session_state.current_room = remaining[0]
                    st.rerun()

# ==========================================
# 7. 메인 영역
# ==========================================
active_messages = st.session_state.chat_rooms[st.session_state.current_room]
is_empty = len(active_messages) == 0

# ── 시작 화면 (대화 없을 때) ──
if is_empty:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">⚖</div>
        <p class="welcome-title">무엇을 도와드릴까요?</p>
        <p class="welcome-sub">법률 관련 질문을 자유롭게 입력하세요</p>
        <div class="chip-row">
            <span class="chip">근로기준법 연장근무 한도</span>
            <span class="chip">임대차계약 해지 조건</span>
            <span class="chip">퇴직금 지급 기준</span>
            <span class="chip">개인정보 침해 구제</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── 채팅 기록 렌더링 ──
else:
    # 상단 작은 헤더
    st.markdown(f"""
    <div style="padding: 28px 0 18px 0; border-bottom: 1px solid #EBEBEA; margin-bottom: 20px;">
        <span style="font-size:15px; font-weight:500; color:#1A1A1A; letter-spacing:-0.3px;">대화 {st.session_state.current_room}</span>
        <span style="font-size:12px; color:#BBBBB8; margin-left:10px;">Hybrid Transformer · RAG</span>
    </div>
    """, unsafe_allow_html=True)

    for message in active_messages:
        with st.chat_message(message["role"]):
            if message.get("is_rag"):
                st.markdown('<div style="font-size:13px; color:#666; margin-bottom:8px;">💡 질문에 대한 정답 데이터가 존재하지 않아, 실시간 법률 지식 원본(RAG) 검색 결과를 가져왔습니다.</div>', unsafe_allow_html=True)
                st.markdown('<div class="rag-label">핵심 조문</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="rag-primary-card">{message["primary_knowledge"]}</div>',
                            unsafe_allow_html=True)
                with st.expander("참조 조문 전체 보기"):
                    st.markdown(f'<div class="rag-full-context">{message["full_context"]}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

# ── 채팅 입력 ──
if prompt := st.chat_input("Ask LAI — 법률 관련 질문을 입력하세요"):
    # 첫 메시지면 헤더 표시를 위해 rerun 전 처리
    active_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        lang = detect_language(prompt)
        if lang not in models:
            err = f"해당 언어({lang.upper()}) 모델을 찾을 수 없습니다."
            st.error(err)
        else:
            fast_answer = retrieve_answer(prompt, qa_pairs_dict.get(lang, []), lang)
            if fast_answer is not None:
                st.markdown(fast_answer)
                active_messages.append({"role": "assistant", "content": fast_answer})
            else:
                with st.spinner("관련 법령을 검색하는 중..."):
                    full_context = generate_response_only(
                        models[lang], vocabs[lang], prompt, retriever,
                        max_len=256, dev=device, lang=lang
                    )
                if full_context:
                                    primary, total = parse_rag_context(full_context)

                                    # 안내 문구 상단에 추가
                                    st.markdown('<div style="font-size:13px; color:#666; margin-bottom:8px;">💡 질문에 대한 정답 데이터가 존재하지 않아, 실시간 법률 지식 원본(RAG) 검색 결과를 가져왔습니다.</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="rag-label">핵심 조문</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="rag-primary-card">{primary}</div>', unsafe_allow_html=True)
                                    with st.expander("참조 조문 전체 보기"):
                                        st.markdown(f'<div class="rag-full-context">{total}</div>', unsafe_allow_html=True)

                                    active_messages.append({
                                        "role": "assistant", "content": "[RAG]",
                                        "is_rag": True, "primary_knowledge": primary, "full_context": total
                                    })
                else:
                    msg = "관련 법조문을 찾지 못했습니다. 다른 방식으로 질문해 주시겠어요?"
                    st.markdown(msg)
                    active_messages.append({"role": "assistant", "content": msg})
    st.rerun()