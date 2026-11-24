import json

import streamlit as st

from core import llm, ui

ui.header(None, "✨ AI assistant", "Ask anything about your data, your models or the decision. The assistant only sees the "
          "numbers Glassbox has computed, shown below, so it explains rather than invents.")

if not llm.available():
    st.info(
        "**Not connected yet.** Add the class OpenAI key to `.streamlit/secrets.toml`:\n\n"
        "```toml\nOPENAI_API_KEY = \"sk-...\"\nOPENAI_MODEL = \"gpt-4o-mini\"   # or the model your instructor specifies\n```\n\n"
        "then restart the app. Everything else in Glassbox works without it."
    )

with st.expander("What the assistant can see"):
    st.code(json.dumps(llm.context(), indent=2, default=str), language="json")

for m in st.session_state.chat:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("e.g. Which model should I trust, and why?", disabled=not llm.available()):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            answer = llm.ask(st.session_state.chat[-10:])
        except Exception as e:
            answer = f"Sorry, the assistant is unavailable right now ({e})."
        st.markdown(answer)
    st.session_state.chat.append({"role": "assistant", "content": answer})
