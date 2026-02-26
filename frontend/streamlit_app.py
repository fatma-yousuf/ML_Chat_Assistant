import os
import re
import json
import time
import streamlit as st
import requests
from pathlib import Path
from datetime import datetime
import markdown as md_lib

# ── Config ──────────────────────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://localhost:8000/chat")
HISTORY_FILE = Path(os.getenv("HISTORY_FILE", "chat_history.json"))
MAX_RETRIES  = 2
TIMEOUT      = 300

st.set_page_config(
    page_title="ML Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1rem; max-width: 900px; }

/* ── Chat window ── */
.chat-container {
    background: transparent;
    padding: 0;
    margin-bottom: 16px;
}

.chat-container.has-messages {
    background: var(--bg-chat, #1c1c1c);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.07);
}

/* ── Bubbles ── */
.chat-wrap {
    width: 100%;
    display: flex;
    margin: 12px 0;
}

.user-bubble {
    margin-left: auto;
    background: linear-gradient(135deg, #1a5c38 0%, #27ae60 100%);
    color: #fff;
    padding: 13px 18px;
    border-radius: 18px 18px 4px 18px;
    font-size: 15px;
    line-height: 1.6;
    max-width: 80%;
    box-shadow: 0 3px 10px rgba(39,174,96,0.25);
    white-space: pre-wrap;
    word-wrap: break-word;
}

.assistant-bubble {
    background: var(--bg-assist, #F1F3F6);
    color: var(--fg-assist, #111);
    padding: 13px 18px;
    border-radius: 18px 18px 18px 4px;
    font-size: 15px;
    line-height: 1.6;
    max-width: 80%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    word-wrap: break-word;
}
.assistant-bubble * { margin: 0 !important; padding: 0 !important; }
.assistant-bubble p { margin-bottom: 4px !important; }
.assistant-bubble ul, .assistant-bubble ol { margin: 2px 0 4px 16px !important; }
.assistant-bubble li { margin: 1px 0 !important; }
.assistant-bubble h1, .assistant-bubble h2, .assistant-bubble h3,
.assistant-bubble h4, .assistant-bubble h5 { margin-top: 14px !important; margin-bottom: 2px !important; font-weight: 700; }
.assistant-bubble strong { font-weight: 700; }
.assistant-bubble code { background: rgba(0,0,0,0.12); border-radius: 4px; padding: 1px 4px !important; font-size: 13px; }
.assistant-bubble pre { background: rgba(0,0,0,0.15); border-radius: 8px; padding: 10px !important; overflow-x: auto; margin: 4px 0 !important; }
.assistant-bubble pre code { background: none; padding: 0 !important; }

.chat-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 5px;
    opacity: .7;
}
.user-bubble   .chat-label { color: rgba(255,255,255,.85); }

.chat-time {
    font-size: 10px;
    opacity: .5;
    margin-top: 5px;
    text-align: right;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 50px 20px;
    color: #aaa;
    font-size: 15px;
}
.empty-state span { font-size: 48px; display: block; margin-bottom: 12px; }

/* ── Input area ── */
.input-box {
    background: var(--bg-input, #FFFFFF);
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

textarea {
    border-radius: 12px !important;
    font-size: 15px !important;
}

/* Override Streamlit's red focus ring with green */
textarea:focus,
div[data-focused="true"] textarea,
.stTextArea textarea:focus {
    border-color: #27ae60 !important;
    box-shadow: 0 0 0 1px #27ae60 !important;
    outline: none !important;
}

/* Also target Streamlit's internal focus wrapper */
div[data-baseweb="textarea"]:focus-within {
    border-color: #27ae60 !important;
    box-shadow: 0 0 0 2px rgba(39,174,96,0.35) !important;
}

button[kind="primary"],
button[kind="primary"]:active,
button[kind="primary"]:focus,
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px !important;
    background: linear-gradient(135deg, #1a5c38 0%, #27ae60 100%) !important;
    background-color: #27ae60 !important;
    color: #fff !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 700 !important;
    transition: opacity .2s !important;
}
button[kind="primary"]:hover,
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1a5c38 0%, #2ecc71 100%) !important;
    background-color: #2ecc71 !important;
    opacity: .92 !important;
}

/* ── Sidebar ── */
.sidebar-stat {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}

/* ── Dark mode overrides ── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-chat:   #1c1c1c;
        --bg-assist: #2a2a2a;
        --fg-assist: #e8e8e8;
        --bg-input:  #1c1c1c;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Persistence helpers ──────────────────────────────────────────────────────
def load_history() -> dict:
    """Load chat history from disk (graceful fallback to empty)."""
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_history(chats: dict) -> None:
    """Persist chat history to disk, silently ignore errors."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(chats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ── Text cleanup ─────────────────────────────────────────────────────────────
def clean_model_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?analysis>", "", text, flags=re.IGNORECASE)
    # Collapse 2+ blank lines → 1
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ── API call with retry ───────────────────────────────────────────────────────
def call_api(chat_id: str, query: str) -> str:
    payload = {"chat_id": chat_id, "query": query}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            res.raise_for_status()
            return res.json().get("response", "No response from server.")
        except requests.exceptions.Timeout:
            last_err = "Request timed out. The server took too long to respond."
        except requests.exceptions.ConnectionError:
            last_err = f"Cannot reach backend at `{API_URL}`. Is it running?"
        except requests.exceptions.HTTPError as e:
            last_err = f"Server error {e.response.status_code}: {e.response.text[:200]}"
            break  # Don't retry on HTTP errors
        except Exception as e:
            last_err = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(1.5 * attempt)

    return f"⚠️ Error after {attempt} attempt(s): {last_err}"


# ── Session init ─────────────────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = load_history()

if "active_chat" not in st.session_state:
    ids = list(st.session_state.chats.keys())
    st.session_state.active_chat = ids[0] if ids else None

if "pending_send" not in st.session_state:
    st.session_state.pending_send = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Chats")

    # ── New chat ──
    with st.expander("➕ New chat", expanded=not st.session_state.chats):
        new_name = st.text_input("Chat name", placeholder="e.g. Neural Networks Q&A", key="new_name_input")
        if st.button("Create", use_container_width=True, type="primary") and new_name.strip():
            chat_id = f"chat_{int(time.time())}"
            st.session_state.chats[chat_id] = {
                "name": new_name.strip(),
                "messages": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.active_chat = chat_id
            save_history(st.session_state.chats)
            st.rerun()

    st.divider()

    # ── Chat list ──
    if st.session_state.chats:
        chat_ids   = list(st.session_state.chats.keys())
        chat_names = [st.session_state.chats[c]["name"] for c in chat_ids]

        try:
            current_idx = chat_ids.index(st.session_state.active_chat)
        except (ValueError, TypeError):
            current_idx = 0

        selected_idx = st.radio(
            "Select chat",
            options=range(len(chat_ids)),
            format_func=lambda i: chat_names[i],
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.active_chat = chat_ids[selected_idx]

        active = st.session_state.chats[st.session_state.active_chat]
        msg_count = len(active["messages"]) // 2
        created   = active.get("created_at", "")
        st.markdown(
            f'<div class="sidebar-stat">💬 {msg_count} exchange(s)'
            + (f" · {created[:10]}" if created else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Rename ──
        with st.expander("✏️ Rename chat"):
            new_label = st.text_input("New name", value=active["name"], key="rename_input")
            if st.button("Save name", use_container_width=True):
                active["name"] = new_label.strip() or active["name"]
                save_history(st.session_state.chats)
                st.rerun()

        # ── Clear ──
        if st.button("🗑️ Clear messages", use_container_width=True):
            active["messages"] = []
            save_history(st.session_state.chats)
            st.rerun()

        # ── Delete ──
        if st.button("❌ Delete chat", use_container_width=True):
            del st.session_state.chats[st.session_state.active_chat]
            remaining = list(st.session_state.chats.keys())
            st.session_state.active_chat = remaining[0] if remaining else None
            save_history(st.session_state.chats)
            st.rerun()

        st.divider()

    st.caption(f"🔗 Backend: `{API_URL}`")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🤖 ML Chat Assistant")
st.caption("Powered by *Hands-On Machine Learning* knowledge base")

if st.session_state.active_chat is None:
    st.info("👈 Create a chat in the sidebar to get started.")
    st.stop()

chat = st.session_state.chats[st.session_state.active_chat]
messages = chat["messages"]

# ── Process pending send (runs after rerun so spinner shows) ─────────────────
if st.session_state.pending_send:
    user_text = st.session_state.pending_send
    st.session_state.pending_send = None

    with st.spinner("Thinking…"):
        raw = call_api(st.session_state.active_chat, user_text)
    answer = clean_model_text(raw)

    ts = datetime.now().strftime("%H:%M")
    messages.append({"role": "user",      "content": user_text, "time": ts})
    messages.append({"role": "assistant", "content": answer,    "time": ts})
    save_history(st.session_state.chats)
    st.rerun()

# ── Chat display ─────────────────────────────────────────────────────────────
container_class = "chat-container has-messages" if messages else "chat-container"
st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)

if not messages:
    st.markdown(
        '<div class="empty-state"><span>💡</span>'
        "Ask anything about Machine Learning!<br>"
        "<small>Try: <em>What is gradient descent?</em></small></div>",
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        ts      = msg.get("time", "")

        if role == "user":
            st.markdown(
                f'<div class="chat-wrap">'
                f'<div class="user-bubble">'
                f'<div class="chat-label">You</div>{content}'
                f'<div class="chat-time">{ts}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                html_content = md_lib.markdown(content, extensions=["nl2br", "fenced_code", "tables"])
            except Exception:
                html_content = content.replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-wrap"><div class="assistant-bubble">'
                f'<div class="chat-label">Assistant</div>'
                f'{html_content}'
                f'<div class="chat-time">{ts}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

st.markdown('</div>', unsafe_allow_html=True)

# ── Input area ───────────────────────────────────────────────────────────────
st.markdown('<div class="input-box">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area(
            "Message",
            placeholder="Type your question… (Shift+Enter for new line)",
            height=100,
            label_visibility="collapsed",
        )
    with col2:
        st.write("")  # vertical padding
        st.write("")
        send_btn = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

if send_btn and user_input.strip():
    st.session_state.pending_send = user_input.strip()
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
