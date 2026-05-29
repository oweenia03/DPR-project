"""
========================================
2026.5.3. 
과제용
숙명여자대학교 공과대학 인공지능공학부 DPR 강좌, 박영호교수
이중 언어 Hybrid Transformer Chatbot Project (Ko/Eng)
========================================
[직접 구현 Transformer + Pre-trained 임베딩 결합]

핵심적인 본 프로그램의 아이디어:
  - Transformer 전체 구조 (Encoder, Decoder, MultiHeadAttention 등)는
    처음부터 직접 구현 
  - 단어 임베딩만 Pre-trained (KoGPT2 / DialoGPT)에서 가져옴
    → 540쌍의 소규모 데이터로도 단어 의미를 이미 알고 있어 품질 향상

- 한글로 질문하면 한글로, 영어로 질문하면 영어로 답합니다.
- 데이터 셋은 수강생 여러분이 직접 확대하여 학습합니다.
- 학생 PC의 학습 시간이 크면 코드 하단 EPOCHS = 500 으로 줄여 테스트한다.
- 한국어 대화 카타고리 참고하셈:
  감정 표현 심화 (설레다/의욕없다/자책), 건강 (허리통증/눈피로),
  IT 심화 (블록체인/메타버스), 한국 음식 심화 (불고기/순대/냉면/된장찌개),
  여행 (동남아), 취업/이직/사업, 우주 등
- 영어 대화는 아래 카타고리 등이니 참고하셈:
  감정 표현 (lost/lacking motivation), 건강 (back pain/eye strain),
  IT 관련 대화 (blockchain/metaverse), 영미 등 글로벌 문화 (kpop/bts/hallyu),
  계절/여행 심화, 자기계발 심화, 음식/음료 다양화 등

Requirements:
    pip install torch transformers pandas scikit-learn numpy

Dataset Files (dataset/ 폴더에 위치):
    dialogs_ko.txt  <- 탭 구분 한국어 Q&A (질문 탭 답변)
    dialogs_en.txt  <- 탭 구분 영어  Q&A (Question 탭 Answer)

Usage:
    python transformer_chatbot_v3.py
"""

import os, re, math, pickle, warnings
warnings.filterwarnings("ignore")

# 환경 변수로 경고 억제
os.environ["SAFETENSORS_FAST_GPU"]             = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]   = "1"

# torch.load 패치 (torch 2.5.x 보안 오류 우회)
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    kwargs["map_location"] = "cpu"
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

# transformers 의 check_torch_load_is_safe 완전 무력화
# (torch 2.5.x 에서 2.6.0 요구 오류를 막음)
try:
    import transformers.utils.import_utils as _tu
    _tu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ======================================================
# 0. 기본 설정
# ======================================================

import faiss
from sentence_transformers import SentenceTransformer

# ✅ 상단에 pickle 임포트 확인 (이미 되어있음)

class LegalRetriever:
    # 1. 파일명을 .pkl로 변경
    def __init__(self, index_path="./law_knowledge.index", metadata_path="./metadata.pkl"):
        self.index = faiss.read_index(index_path)
        # 2. 피클 파일 로드 방식으로 변경
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    def retrieve(self, query, k=5, max_len=300):
        # 3. DB 생성할 때와 동일하게 normalize_embeddings=True 적용
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype('float32')
        # faiss.normalize_L2(query_vec) <- 이건 삭제해도 됩니다 (위에서 이미 정규화됨)
        
        distances, indices = self.index.search(query_vec, k)
        
        # ✅ 인덱스 에러 방어 코드
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return " ".join(results)[:max_len]
        
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

PAD_IDX  = 0
SOS_IDX  = 1
EOS_IDX  = 2
UNK_IDX  = 3
SPECIALS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

# Pre-trained 토크나이저 모델명
PRETRAINED_KO = "skt/kogpt2-base-v2"
PRETRAINED_EN = "microsoft/DialoGPT-small"


# ======================================================
# 1. 언어 감지
# ======================================================
def detect_language(text: str) -> str:
    for ch in text:
        if '\uAC00' <= ch <= '\uD7A3':
            return 'ko'
        if '\u3131' <= ch <= '\u314E':
            return 'ko'
        if '\u314F' <= ch <= '\u3163':
            return 'ko'
    return 'en'


# ======================================================
# 2. 데이터 전처리
# ======================================================
def clean_text(text, lang):
    if pd.isna(text):
        return ''
    text = str(text).strip()
    if lang == 'en':
        text = text.lower()
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ======================================================
# 3. Pre-trained 토크나이저 기반 Vocab
#    단어 임베딩 가중치는 GPT 모델에서 추출
# ======================================================
class PretrainedVocab:
    """
    Pre-trained 토크나이저의 어휘 사전을 그대로 사용.
    기존 scratch Vocab 과 동일한 인터페이스 제공.
    """
    def __init__(self, tokenizer, lang):
        self.tok           = tokenizer
        self.lang          = lang
        self.pad_token_id  = tokenizer.convert_tokens_to_ids('<PAD>')
        self.sos_token_id  = tokenizer.convert_tokens_to_ids('<SOS>')
        self.eos_token_id  = tokenizer.convert_tokens_to_ids('<EOS>')
        self.unk_token_id  = tokenizer.convert_tokens_to_ids('<UNK>')

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, token_str):
        ids = self.tok.encode(token_str, add_special_tokens=False)
        return ids[0] if ids else self.unk_token_id

    # scratch Vocab 호환 메서드
    def lookup_tokens(self, ids):
        return [self.tok.decode([i], skip_special_tokens=True) for i in ids]


def get_pretrained_embedding(model_name, vocab_size, emb_size, device):
    """
    Pre-trained GPT 모델에서 임베딩 가중치만 추출하여 반환.
    모든 연산을 CPU 에서 수행 후 반환 (device 불일치 방지).
    """
    from transformers import AutoModelForCausalLM
    print(f"  Pre-trained 임베딩 추출 중: {model_name}")

    # 항상 CPU 로 로드 (device 불일치 방지)
    gpt = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype       = torch.float32,
        use_safetensors   = False,
        low_cpu_mem_usage = True,
    )
    # CPU 에서 임베딩 가중치 추출
    emb_weight   = gpt.get_input_embeddings().weight.data.float().cpu().clone()
    actual_vocab = emb_weight.shape[0]
    actual_emb   = emb_weight.shape[1]
    del gpt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  Pre-trained 임베딩: ({actual_vocab}, {actual_emb})"
          f"  →  목표: ({vocab_size}, {emb_size})")

    # emb_size 가 다르면 CPU 에서 Linear 투영
    if actual_emb != emb_size:
        proj       = nn.Linear(actual_emb, emb_size, bias=False)  # CPU
        emb_weight = proj(emb_weight).detach()                     # CPU 결과

    # vocab_size 맞춤 (모두 CPU)
    if actual_vocab >= vocab_size:
        emb_weight = emb_weight[:vocab_size].contiguous()
    else:
        extra      = torch.randn(vocab_size - actual_vocab, emb_size) * 0.02
        emb_weight = torch.cat([emb_weight, extra], dim=0)

    return emb_weight.cpu()   # 반드시 CPU 로 반환


# ======================================================
# 4. 데이터셋
# ======================================================
class QADataset(Dataset):
    def __init__(self, pairs, vocab: PretrainedVocab,
                 input_seq_len: int, target_seq_len: int):
        self.pairs          = pairs
        self.vocab          = vocab
        self.input_seq_len  = input_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_ids = self.vocab.encode(q)
        a_ids = self.vocab.encode(a)

        eos = self.vocab.eos_token_id
        sos = self.vocab.sos_token_id
        pad = self.vocab.pad_token_id

        enc_src = self._pad(q_ids + [eos],       self.input_seq_len,  pad)
        dec_src = self._pad([sos] + a_ids, self.target_seq_len, pad)
        trg     = self._pad(a_ids + [eos], self.target_seq_len, pad)
        return enc_src, dec_src, trg

    def _pad(self, seq, max_len, pad_val):
        seq = seq[:max_len]
        return F.pad(torch.LongTensor(seq),
                     (0, max_len - len(seq)), value=pad_val)


# ======================================================
# 5. Pre-trained 임베딩을 사용하는 위치 임베딩 레이어
#    (핵심: GPT 가중치로 초기화된 word_emb 사용)
# ======================================================
class PretrainedPositionEmbedding(nn.Module):
    """
    [직접 구현 + Pre-trained 결합 핵심 부분]
    - word_emb: Pre-trained GPT 임베딩으로 초기화 (단어 의미 보유)
    - pe: 직접 구현한 Sinusoidal 위치 인코딩
    """
    def __init__(self, vocab_size, max_seq_len, emb_size, pretrained_weight=None):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)

        # Pre-trained 가중치로 초기화
        if pretrained_weight is not None:
            # CPU 텐서를 임베딩 레이어의 device 에 맞춰 복사
            self.word_emb.weight.data.copy_(pretrained_weight.cpu())
            print(f"  Pre-trained 임베딩 가중치 로드 완료 ({vocab_size} x {emb_size})")

        # Sinusoidal 위치 인코딩 (직접 구현)
        pos   = torch.arange(max_seq_len).unsqueeze(1).float()
        denom = torch.exp(torch.arange(0, emb_size, 2).float() *
                          (-math.log(10000.0) / emb_size))
        pe    = torch.zeros(max_seq_len, emb_size)
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.word_emb(x) + self.pe[:x.size(1)]


# ======================================================
# 6. 직접 구현 Transformer 구성 요소 (기존과 동일)
# ======================================================

class MultiHeadAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention (직접 구현)"""
    def __init__(self, emb_size, heads):
        super().__init__()
        assert emb_size % heads == 0, "emb_size 는 heads 의 배수여야 합니다."
        self.heads    = heads
        self.head_dim = emb_size // heads
        self.W_v      = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k      = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_q      = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out   = nn.Linear(emb_size, emb_size)

    def forward(self, values, keys, queries, mask=None):
        B   = queries.shape[0]
        V_L = values.shape[1]
        K_L = keys.shape[1]
        Q_L = queries.shape[1]

        V = values.reshape(B,  self.heads, V_L, self.head_dim)
        K = keys.reshape(B,    self.heads, K_L, self.head_dim)
        Q = queries.reshape(B, self.heads, Q_L, self.head_dim)

        V = self.W_v(V)
        K = self.W_k(K)
        Q = self.W_q(Q)

        energy = torch.matmul(Q, K.transpose(2, 3)) / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attn = torch.softmax(energy, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.reshape(B, Q_L, self.heads * self.head_dim)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, ff_expansion, dropout):
        super().__init__()
        self.attn  = MultiHeadAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff    = nn.Sequential(
            nn.Linear(emb_size, ff_expansion * emb_size),
            nn.GELU(),  # ◀ 성능이 더 좋은 최신 활성화 함수로 교체!
            nn.Linear(ff_expansion * emb_size, emb_size),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        x   = self.drop(self.norm1(self.attn(value, key, query, mask) + query))
        out = self.norm2(self.drop(self.ff(x) + x))
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads,
                 ff_exp, dropout, pretrained_weight=None):
        super().__init__()
        self.emb    = PretrainedPositionEmbedding(
            vocab_size, seq_len, emb_size, pretrained_weight)
        self.layers = nn.ModuleList([
            TransformerBlock(emb_size, heads, ff_exp, dropout)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.drop(self.emb(x))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, emb_size, heads, ff_exp, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_size, heads)
        self.norm      = nn.LayerNorm(emb_size)
        self.cross_blk = TransformerBlock(emb_size, heads, ff_exp, dropout)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        query = self.drop(self.norm(self.self_attn(x, x, x, trg_mask) + x))
        return self.cross_blk(enc_out, enc_out, query, src_mask)


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads,
                 ff_exp, dropout, pretrained_weight=None):
        super().__init__()
        self.emb    = PretrainedPositionEmbedding(
            vocab_size, seq_len, emb_size, pretrained_weight)
        self.layers = nn.ModuleList([
            DecoderBlock(emb_size, heads, ff_exp, dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        out = self.drop(self.emb(x))
        for layer in self.layers:
            out = layer(out, enc_out, src_mask, trg_mask)
        return self.fc_out(out)


class TransformerScratch(nn.Module):
    """인코더 + 디코더 전체 Transformer (직접 구현)"""
    def __init__(self, vocab_size, src_pad_idx,
                 emb_size=256, n_layers=2, heads=8,
                 ff_exp=4, dropout=0.1, max_seq_len=50,
                 device=torch.device('cpu'),
                 pretrained_weight=None):
        super().__init__()
        self.src_pad = src_pad_idx
        self.dev     = device

        self.encoder = Encoder(vocab_size, max_seq_len, emb_size,
                               n_layers, heads, ff_exp, dropout,
                               pretrained_weight).to(device)
        self.decoder = Decoder(vocab_size, max_seq_len, emb_size,
                               n_layers, heads, ff_exp, dropout,
                               pretrained_weight).to(device)

    def _src_mask(self, src):
        return (src != self.src_pad).unsqueeze(1).unsqueeze(2).to(self.dev)

    def _trg_mask(self, trg):
        B, T = trg.shape
        return torch.tril(torch.ones(T, T)).expand(B, 1, T, T).to(self.dev)

    def forward(self, src, trg):
        enc_out = self.encoder(src, self._src_mask(src))
        return self.decoder(trg, enc_out,
                            self._src_mask(src), self._trg_mask(trg))


# ======================================================
# 7. 학습 유틸리티
# ======================================================
def _step(model, batch, loss_fn, pad_idx, dev):
    enc, dec, trg = [t.to(dev) for t in batch]
    logits = model(enc, dec)
    loss   = loss_fn(logits.view(-1, logits.size(-1)), trg.view(-1))
    
    mask = (trg != pad_idx) # 패딩 제외 마스크
    correct = ((logits.argmax(-1) == trg) & mask).sum().item()
    acc = correct / max(mask.sum().item(), 1)
    return loss, acc


def train_epoch(model, loader, opt, loss_fn, clip, pad_idx, dev):
    model.train()
    total_loss = total_acc = 0
    for batch in loader:
        opt.zero_grad()
        loss, acc = _step(model, batch, loss_fn, pad_idx, dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item()
        total_acc  += acc
    return total_loss / len(loader), total_acc / len(loader)


def eval_epoch(model, loader, loss_fn, pad_idx, dev):
    model.eval()
    total_loss = total_acc = 0
    with torch.no_grad():
        for batch in loader:
            loss, acc = _step(model, batch, loss_fn, pad_idx, dev)
            total_loss += loss.item()
            total_acc  += acc
    return total_loss / len(loader), total_acc / len(loader)


# ======================================================
# 8. 학습 루프 (Early Stopping 포함)
# ======================================================
def run_training(model, train_loader, val_loader, pad_idx, dev,
                 epochs=3000, lr=1e-4, clip=1, lang='??',
                 patience=500, min_delta=5e-3,
                 warmup_epochs=200, smooth_window=20):
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    best_epoch       = 0
    best_train_acc   = 0.0
    val_loss_history = []

    print("\n" + "="*55)
    print(f"  [{lang.upper()}] 숙대 미니 트랜스포머챗봇 학습 시작")
    print(f"  epochs={epochs}  lr={lr}  warmup={warmup_epochs}")
    print(f"  Early Stopping: patience={patience}  min_delta={min_delta}")
    print("="*55)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt,
                                      loss_fn, clip, pad_idx, dev)
        val_str = ''
        es_flag = ''

        if val_loader:
            v_loss, v_acc = eval_epoch(model, val_loader, loss_fn, pad_idx, dev)
            val_str = f'  ||  Val Loss: {v_loss:.3f}  Val Acc: {v_acc*100:.2f}%'

            val_loss_history.append(v_loss)
            win = min(len(val_loss_history), smooth_window)
            smoothed = sum(val_loss_history[-win:]) / win

            # warmup 마지막 epoch: 초기 best_state 확보
            if epoch == warmup_epochs and best_state is None:
                best_state     = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                best_epoch     = epoch
                best_train_acc = tr_acc
                best_val_loss  = smoothed

            # warmup 이후에만 Best 갱신 및 patience 카운트
            if epoch > warmup_epochs:
                if smoothed < best_val_loss - min_delta:
                    best_val_loss    = smoothed
                    best_state       = {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}
                    patience_counter = 0
                    best_epoch       = epoch
                    best_train_acc   = tr_acc
                else:
                    patience_counter += 1
                    remaining = patience - patience_counter
                    if remaining <= 100:
                        es_flag = f'  [patience 잔여: {remaining}]'

        if epoch % 100 == 0 or epoch == 1:
            print(f'Epoch {epoch:05d}  '
                  f'Train Loss: {tr_loss:.3f}  Train Acc: {tr_acc*100:.2f}%'
                  f'{val_str}{es_flag}')

        if val_loader and epoch > warmup_epochs and patience_counter >= patience:
            print(f'\n[Early Stopping] Epoch {epoch} 에서 학습 종료')
            print(f'  최적 모델: Epoch {best_epoch:05d}  '
                  f'Best Val Loss(평활화): {best_val_loss:.3f}')
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        print(f'\n  최적 모델 (Epoch {best_epoch}) 가중치로 복원 완료')
    else:
        best_epoch     = epochs
        best_train_acc = tr_acc
        print(f'\n  학습 완료 (전체 {epochs} epoch)')

    return model, best_epoch, best_train_acc


# ======================================================
# 9. 응답 생성
# ======================================================
def generate_rag_response(model, vocab, question, retriever, max_len, temperature, dev, lang):
    model.eval()
    eos = vocab.eos_token_id
    sos = vocab.sos_token_id
    pad = vocab.pad_token_id

    context = retriever.retrieve(question, max_len=100)
    print("\n[DEBUG] 질문:", question)
    print("[DEBUG] context:", context)
    clean_q = clean_text(question, lang) 
    
    # ✅ 훈련 데이터와 동일하게 언어별 태그 분기
    ref_tag = "[참고]" if lang == 'ko' else "[Reference]"
    q_tag = "[질문]" if lang == 'ko' else "[Question]"
    
    if context:
        prompt = f"{ref_tag} {context} {q_tag} {clean_q}"
    else:
        prompt = f"{q_tag} {clean_q}"
        
    enc_ids = vocab.encode(prompt)
    enc_ids = enc_ids[:max_len - 1] + [eos] 
    
    enc_src = F.pad(torch.LongTensor(enc_ids),
                    (0, max_len - len(enc_ids)),
                    value=pad).unsqueeze(0).to(dev)
    dec_src = torch.LongTensor([[sos]]).to(dev)

    generated = []
    repeat_penalty = 1.5 
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(enc_src, dec_src)
            next_token_logits = logits[:, -1, :]
            
            next_token_logits[0, pad] = -1e9
            next_token_logits[0, sos] = -1e9

            for token_id in set(generated):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repeat_penalty
                else:
                    next_token_logits[0, token_id] *= repeat_penalty

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                nxt = next_token_logits.argmax(dim=-1)
                
            if nxt.item() == eos:
                break
                
            dec_src = torch.cat([dec_src, nxt.unsqueeze(-1)], dim=1)
            if dec_src.size(1) > max_len:
                break
            generated.append(nxt.item())
            
            if len(generated) > 10 and len(set(generated[-5:])) == 1:
                break

    if not generated:
        return '...'

    skip = {sos, eos, pad}
    ids  = [t for t in generated if t not in skip]
    return vocab.decode(ids) if ids else '...'




# ======================================================
# 9-1. 안정형 검색 기반 응답 보정
# ======================================================
def retrieve_answer(question, qa_pairs, lang, threshold=0.70):
    if not qa_pairs:
        return None

    q_clean = clean_text(question, lang)
    if not q_clean:
        return None

    best_score = 0.0
    best_answer = None

    for q, a in qa_pairs:
        # raw_pairs에는 순수 질문만 있으므로 곧바로 clean_text 적용
        q2 = clean_text(q, lang)

        if q_clean == q2:
            return a

        score = SequenceMatcher(None, q_clean, q2).ratio()
        if q_clean in q2 or q2 in q_clean:
            score = max(score, 0.90)

        if score > best_score:
            best_score = score
            best_answer = a

    if best_score >= threshold:
        return best_answer

    return None


def load_rag_pairs_from_file(data_path, lang, retriever):
    df = pd.read_csv(data_path, sep='\t', names=['Q', 'A'], encoding='utf-8', on_bad_lines='skip')
    df['Q'] = df['Q'].apply(lambda x: clean_text(x, lang))
    df['A'] = df['A'].apply(lambda x: clean_text(x, lang))
    df = df[(df['Q'].str.strip() != '') & (df['A'].str.strip() != '')].reset_index(drop=True)
    
    rag_pairs = []
    raw_pairs = []
    print(f"  [INFO] {lang.upper()} 데이터셋에 RAG 컨텍스트를 주입하여 학습 데이터를 생성합니다...")
    
    # ✅ 언어별 태그 분기
    ref_tag = "[참고]" if lang == 'ko' else "[Reference]"
    q_tag = "[질문]" if lang == 'ko' else "[Question]"
    
    for q, a in zip(df['Q'], df['A']):
        context = retriever.retrieve(q, k=1, max_len=100)
        
        if context:
            rag_q = f"{ref_tag} {context} {q_tag} {q}"
        else:
            rag_q = f"{q_tag} {q}"
            
        rag_pairs.append((rag_q, a))
        raw_pairs.append((q, a))
        
    return rag_pairs, raw_pairs


def prepare_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({
        'pad_token': '<PAD>',
        'bos_token': '<SOS>',
        'eos_token': '<EOS>',
        'unk_token': '<UNK>',
    })
    return tokenizer

# ======================================================
# 10. 저장 / 불러오기
# ======================================================
def save_bundle(bundle, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  저장 완료: {path}')


def load_bundle(path):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    print(f'  불러오기 완료: {path}')
    return bundle


# ======================================================
# 11. 채팅 루프
# ======================================================
def bilingual_chat(models, vocabs, max_lens, qa_pairs, retriever, temperature=0.8, dev=device):
    print("\n" + "-"*50)
    print("  Bilingual Hybrid Transformer Chatbot")
    print("  한글로 입력하면 한글로, 영어로 입력하면 영어로 답합니다")
    print("  종료: 'bye' 또는 '종료' 입력")
    print("-"*50 + "\n")

    while True:
        try:
            user_input = input("SM-눈송이: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nT-Bot: 안녕히 가세요! Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('bye', '종료', '끝', 'exit', 'quit'):
            print("T-Bot: 안녕히 가세요! Goodbye!")
            break

        lang     = detect_language(user_input)
        model    = models[lang]
        vocab    = vocabs[lang]
        mlen     = max_lens[lang]
        lang_tag = "[KR]" if lang == 'ko' else "[EN]"

        response = retrieve_answer(user_input, qa_pairs.get(lang, []), lang)
        if response is None:
                    if retriever:
                        # 테스트 코드를 지우고, 진짜 모델이 읽고 쓰게 만듭니다!
                        response = generate_rag_response(model, vocab, user_input, retriever, mlen, temperature, dev, lang) 
                    else:
                        response = '벡터 DB가 연결되지 않아 답변을 생성할 수 없습니다.'

        print(f"T-Bot {lang_tag}: {response}")
        print()


# ======================================================
# 12. 메인 실행
# ======================================================
MODELS_DIR = "./transformer/models_v3_stable"
DATA_KO = "./dataset/legal_qa_dataset.txt"
DATA_EN = "./dataset/legal_qa_dataset_en.txt"

# 학습 하이퍼파라미터
EPOCHS         = 1000
LR             = 5e-5     # Pre-trained 임베딩 파인튜닝에 맞는 lr
BATCH_SIZE     = 32
EMB_SIZE       = 768     # 임베딩 차원 (GPT → 투영)
N_LAYERS       = 4
HEADS          = 8
FF_EXP         = 4
DROPOUT        = 0.1
MAX_LEN        = 256
TEMPERATURE    = 0.0
PATIENCE       = 200
MIN_DELTA      = 1e-3
WARMUP_EPOCHS  = 200
SMOOTH_WINDOW  = 20

PRETRAINED = {'ko': PRETRAINED_KO, 'en': PRETRAINED_EN}


if __name__ == "__main__":
    
    # 1. 벡터 DB 검색기 인스턴스 생성
    try:
        retriever = LegalRetriever()
        print("\n[TEST] Retriever 단독 테스트")
        print(retriever.retrieve("연차 유급휴가"))
        print(retriever.retrieve("근로기준법 제60조"))
        print("="*50)
    except Exception as e:
        print(f"[ERROR] 벡터 DB 로드 실패: {e}")
        retriever = None
        
    models = {}
    vocabs = {}
    max_lens = {}
    qa_pairs_dict = {}

    for lang, data_path in [('ko', DATA_KO), ('en', DATA_EN)]:

        print(f"\n{'─'*55}")
        print(f"  [{lang.upper()}] 언어 모델 준비 중...")
        print(f"{'─'*55}")

        import glob
        existing = sorted(glob.glob(f"{MODELS_DIR}/2_bundle_{lang}_*.pkl"))
        bundle_path = existing[-1] if existing else None

        # ── Pre-trained 토크나이저 로드 ──────────────
        print(f"  Pre-trained 토크나이저 로드: {PRETRAINED[lang]}")
        tokenizer = prepare_tokenizer(PRETRAINED[lang])
        vocab = PretrainedVocab(tokenizer, lang)
        vocab_size = len(vocab)
        print(f"  어휘 사전 크기: {vocab_size}")

        # ✅ [핵심 수정] 여기서 RAG 데이터를 먼저 로드!
        if retriever:
            pairs, raw_pairs = load_rag_pairs_from_file(data_path, lang, retriever)
        else:
            print("  [WARNING] 벡터 DB가 없어 일반 텍스트 쌍으로만 진행합니다.")
            df = pd.read_csv(data_path, sep='\t', names=['Q', 'A'], encoding='utf-8', on_bad_lines='skip')
            df['Q'] = df['Q'].apply(lambda x: clean_text(x, lang))
            df['A'] = df['A'].apply(lambda x: clean_text(x, lang))
            df = df[(df['Q'].str.strip() != '') & (df['A'].str.strip() != '')].reset_index(drop=True)
            pairs = list(zip(df['Q'], df['A']))
            raw_pairs = pairs # 벡터 DB 없으면 원본 그대로 사용

        if bundle_path:
            print(f"  저장된 모델 발견: {os.path.basename(bundle_path)}")
            bundle  = load_bundle(bundle_path)
            max_seq = bundle['max_seq']
            print(f"  로드된 모델 Max Sequence Length (최대 길이): {max_seq} 토큰")
            model   = TransformerScratch(
                vocab_size   = vocab_size,
                src_pad_idx  = vocab.pad_token_id,
                emb_size     = EMB_SIZE,
                n_layers     = N_LAYERS,
                heads        = HEADS,
                ff_exp       = FF_EXP,
                dropout      = DROPOUT,
                max_seq_len  = max_seq,
                device       = device,
            ).to(device)
            model.load_state_dict(bundle['state_dict'])
            model.eval()

        else:
            if len(pairs) > 10:
                tr_pairs, val_pairs = train_test_split(
                    pairs, test_size=0.1, random_state=42)
            else:
                tr_pairs, val_pairs = pairs, []

            # max_seq 결정
            all_texts = [q for q, _ in tr_pairs] + [a for _, a in tr_pairs]
            max_seq   = min(
                max(len(vocab.encode(t)) + 2 for t in all_texts),
                MAX_LEN
            )
            print(f"  계산된 모델 Max Sequence Length (최대 길이): {max_seq} 토큰")
            print(f"  학습: {len(tr_pairs)}쌍  검증: {len(val_pairs)}쌍  "
                  f"max_seq: {max_seq}")

            # ── Pre-trained 임베딩 추출 ────────────
            pt_weight = get_pretrained_embedding(
                PRETRAINED[lang], vocab_size, EMB_SIZE, device)

            # ── 데이터로더 ─────────────────────────
            tr_ds = QADataset(tr_pairs, vocab, max_seq, max_seq)
            tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_dl = None
            if val_pairs:
                val_ds = QADataset(val_pairs, vocab, max_seq, max_seq)
                val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

            # ── 모델 생성 (Pre-trained 임베딩 주입) ──
            model = TransformerScratch(
                vocab_size   = vocab_size,
                src_pad_idx  = vocab.pad_token_id,
                emb_size     = EMB_SIZE,
                n_layers     = N_LAYERS,
                heads        = HEADS,
                ff_exp       = FF_EXP,
                dropout      = DROPOUT,
                max_seq_len  = max_seq,
                device       = device,
                pretrained_weight = pt_weight,
            ).to(device)

            # ── 학습 ──────────────────────────────
            model, best_ep, best_acc = run_training(
                model, tr_dl, val_dl,
                pad_idx       = vocab.pad_token_id,
                dev           = device,
                epochs        = EPOCHS,
                lr            = LR,
                lang          = lang,
                patience      = PATIENCE,
                min_delta     = MIN_DELTA,
                warmup_epochs = WARMUP_EPOCHS,
                smooth_window = SMOOTH_WINDOW,
            )

            # ── 저장 ──────────────────────────────
            acc_pct     = int(round(best_acc * 100))
            bundle_path = (f"{MODELS_DIR}/2_bundle_{lang}"
                           f"_ep{best_ep:05d}_acc{acc_pct:03d}.pkl")
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_bundle({
                'max_seq':    max_seq,
                'state_dict': model.state_dict(),
                'best_epoch': best_ep,
                'best_acc':   best_acc,
                'pairs':      pairs,
            }, bundle_path)

        models[lang]   = model
        vocabs[lang]   = vocab
        max_lens[lang] = max_seq
        qa_pairs_dict[lang] = raw_pairs

    # 3. 채팅 루프 인자 추가
    bilingual_chat(models, vocabs, max_lens, qa_pairs_dict, retriever, temperature=TEMPERATURE, dev=device)