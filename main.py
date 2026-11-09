import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "너는 친절하고 정확한 한국어 AI 도우미야. 사용자의 질문에 이해하기 쉽게 답해."
}

st.set_page_config(page_title="무료 로컬 AI 챗봇", page_icon="🤖", layout="centered")
st.title("🤖 무료 로컬 AI 챗봇")
st.caption("Qwen2.5-1.5B-Instruct · OpenAI API 키 불필요")

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("설정")
    max_new_tokens = st.slider("답변 길이", min_value=64, max_value=800, value=400, step=32)
    temperature = st.slider("창의성", min_value=0.1, max_value=1.2, value=0.7, step=0.1)
    if st.button("대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.write("**모델:**", MODEL_NAME)
    st.write("**실행 장치:**", "CUDA GPU" if torch.cuda.is_available() else "CPU")
    st.caption("최초 실행 시 모델 파일을 다운로드합니다. CPU에서는 답변 생성이 느릴 수 있습니다.")

# 이전 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_answer(user_text: str) -> str:
    tokenizer, model = load_model()
    context = [SYSTEM_MESSAGE] + st.session_state.messages[-10:] + [
        {"role": "user", "content": user_text}
    ]
    prompt = tokenizer.apply_chat_template(
        context,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

user_text = st.chat_input("무엇이든 질문해 보세요.")
if user_text:
    if user_text.strip() == "/초기화":
        st.session_state.messages = []
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        with st.chat_message("assistant"):
            with st.spinner("AI가 답변을 생성하고 있습니다... (첫 실행은 모델 로딩으로 더 오래 걸립니다)"):
                answer = generate_answer(user_text)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as error:
        # 생성에 실패한 사용자 메시지는 기록에서 제거
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
        st.error(f"실행 중 오류가 발생했습니다: {error}")
        st.info("인터넷 연결, 메모리 용량, 패키지 설치 상태를 확인해 주세요.")
