"""En enkel lokal AI-chat bygget med Streamlit og Ollama."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st


MODEL = "phi3.5"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
SYSTEM_PROMPT = (
    "Du er en venlig og hjælpsom AI-assistent. Svar på samme sprog som brugeren, "
    "medmindre brugeren beder om noget andet. Vær tydelig og præcis."
)


st.set_page_config(
    page_title="Simpel AI-chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(91, 124, 250, .11), transparent 28rem),
                radial-gradient(circle at 90% 20%, rgba(126, 231, 190, .10), transparent 24rem),
                #f8fafc;
        }
        .block-container {
            max-width: 820px;
            padding-top: 2.6rem;
            padding-bottom: 7rem;
        }
        .hero-kicker {
            color: #536179;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: .13em;
            margin-bottom: .35rem;
            text-transform: uppercase;
        }
        .hero-title {
            color: #172033;
            font-size: clamp(2.15rem, 6vw, 3.55rem);
            font-weight: 750;
            letter-spacing: -.045em;
            line-height: 1.02;
            margin: 0;
        }
        .hero-copy {
            color: #657189;
            font-size: 1.02rem;
            line-height: 1.6;
            margin: .8rem 0 1.7rem;
            max-width: 620px;
        }
        .model-chip {
            align-items: center;
            background: #eef2ff;
            border: 1px solid #dce3ff;
            border-radius: 999px;
            color: #465579;
            display: inline-flex;
            font-size: .82rem;
            font-weight: 650;
            gap: .45rem;
            padding: .38rem .7rem;
        }
        .model-dot {
            background: #7c8ff5;
            border-radius: 50%;
            height: .48rem;
            width: .48rem;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, .78);
            border: 1px solid rgba(218, 225, 237, .95);
            border-radius: 18px;
            box-shadow: 0 7px 22px rgba(34, 47, 72, .045);
            padding: .35rem .45rem;
        }
        [data-testid="stChatInput"] {
            border-radius: 16px;
            box-shadow: 0 12px 35px rgba(34, 47, 72, .12);
        }
        div.stButton > button[kind="primary"] {
            border-radius: 12px;
            font-weight: 700;
            min-height: 2.8rem;
        }
        div.stButton > button[kind="secondary"] {
            border-radius: 12px;
            min-height: 2.8rem;
        }
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_url(path: str) -> str:
    """Byg en URL til Ollamas lokale HTTP-API."""
    return f"{OLLAMA_URL}{path}"


def ollama_models() -> list[str]:
    """Kontrollér forbindelsen, og returnér installerede modelnavne."""
    response = requests.get(api_url("/api/tags"), timeout=5)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


def model_is_installed(models: list[str]) -> bool:
    """Match både `phi3.5` og Ollamas fulde `phi3.5:latest`-navn."""
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in models)


def pull_model(status: Any, progress: Any) -> None:
    """Hent modellen og vis Ollamas fremdrift i Streamlit."""
    with requests.post(
        api_url("/api/pull"),
        json={"name": MODEL, "stream": True},
        stream=True,
        timeout=(10, 900),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if error := event.get("error"):
                raise RuntimeError(error)

            message = event.get("status", "Henter model …")
            status.update(label=message.capitalize(), state="running")

            total = event.get("total")
            completed = event.get("completed")
            if total and completed is not None:
                progress.progress(
                    min(completed / total, 1.0),
                    text=f"Henter {MODEL} · {completed / total:.0%}",
                )


def warm_up_model() -> None:
    """Indlæs modellen i hukommelsen, så første chatsvar starter hurtigere."""
    response = requests.post(
        api_url("/api/generate"),
        json={
            "model": MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": "30m",
        },
        timeout=300,
    )
    response.raise_for_status()


def stream_chat(messages: list[dict[str, str]]) -> Iterator[str]:
    """Stream ét AI-svar fra Ollama."""
    payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    with requests.post(
        api_url("/api/chat"),
        json={
            "model": MODEL,
            "messages": payload_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": 0.7},
        },
        stream=True,
        timeout=(10, 600),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if error := event.get("error"):
                raise RuntimeError(error)
            content = event.get("message", {}).get("content", "")
            if content:
                yield content


def friendly_error(error: Exception) -> str:
    """Omsæt typiske lokale fejl til en kort dansk besked."""
    if isinstance(error, requests.ConnectionError):
        return (
            "Jeg kan ikke få forbindelse til Ollama. Start Ollama-programmet "
            "(eller kør `ollama serve`) og prøv igen."
        )
    if isinstance(error, requests.Timeout):
        return "Ollama svarede ikke i tide. Prøv igen om et øjeblik."
    if isinstance(error, requests.HTTPError):
        detail = ""
        if error.response is not None:
            try:
                detail = error.response.json().get("error", "")
            except (ValueError, AttributeError):
                detail = error.response.text[:200]
        return f"Ollama returnerede en fejl{f': {detail}' if detail else '.'}"
    return f"Noget gik galt: {error}"


if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False


st.markdown('<div class="hero-kicker">Lokal · privat · enkel</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Din egen AI-chat</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Indlæs Phi-3.5 via Ollama, og begynd at chatte. '
    "Samtalen og modellen bliver på din computer.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="model-chip"><span class="model-dot"></span> Ollama · phi3.5</span>',
    unsafe_allow_html=True,
)
st.write("")

load_col, clear_col = st.columns([2, 1])
with load_col:
    load_clicked = st.button(
        "Indlæs AI" if not st.session_state.model_ready else "AI er indlæst",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.model_ready,
    )
with clear_col:
    if st.button(
        "Ryd samtale",
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()

if load_clicked:
    try:
        with st.status("Forbinder til Ollama …", expanded=True) as load_status:
            models = ollama_models()
            progress_bar = st.progress(0, text="Kontrollerer model …")

            if model_is_installed(models):
                load_status.write(f"{MODEL} findes allerede på computeren.")
                progress_bar.progress(1.0, text="Modellen er klar")
            else:
                load_status.write(
                    f"Henter {MODEL} første gang. Det kan tage et par minutter."
                )
                pull_model(load_status, progress_bar)
                progress_bar.progress(1.0, text="Modellen er hentet")

            load_status.update(label="Indlæser modellen i hukommelsen …", state="running")
            warm_up_model()
            st.session_state.model_ready = True
            load_status.update(label="AI'en er klar", state="complete", expanded=False)
    except (requests.RequestException, RuntimeError, ValueError) as error:
        st.session_state.model_ready = False
        st.error(friendly_error(error))

if st.session_state.model_ready:
    st.success("AI'en er klar. Skriv en besked nedenfor.", icon="✅")
else:
    st.info("Tryk på **Indlæs AI** for at gøre chatten klar.", icon="ℹ️")

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Skriv en besked …" if st.session_state.model_ready else "Indlæs AI for at starte …",
    disabled=not st.session_state.model_ready,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            answer = st.write_stream(stream_chat(st.session_state.messages))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer or ""}
            )
        except (requests.RequestException, RuntimeError, ValueError) as error:
            st.error(friendly_error(error))
