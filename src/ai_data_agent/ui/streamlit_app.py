# src/ai_data_agent/ui/streamlit_app.py

import streamlit as st
import plotly.graph_objects as go

from ai_data_agent.logging_config import configure_logging
from ai_data_agent.exceptions import DataIngestionError, DataValidationError, AnalystAgentError
from ai_data_agent.ingestion.loaders import load_dataset
from ai_data_agent.ingestion.profiler import profile_dataset
from ai_data_agent.ingestion.validators import validate_dataset_quality
from ai_data_agent.agent.graph import build_graph

configure_logging()

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")


def init_session_state():
    defaults = {
        "df": None,
        "profile": None,
        "quality_warnings": [],
        "conversation_history": [],
        "chat_messages": [],
        "agent_app": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

if st.session_state["agent_app"] is None:
    st.session_state["agent_app"] = build_graph()


st.sidebar.title("📊 AI Data Analyst")
st.sidebar.markdown("Upload a dataset to get started.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, or Parquet",
    type=["csv", "xlsx", "parquet"],
)

if uploaded_file is not None:
    if st.session_state.get("_last_uploaded_name") != uploaded_file.name:
        try:
            file_bytes = uploaded_file.read()
            df = load_dataset(file_bytes, uploaded_file.name)
            profile = profile_dataset(df)
            warnings = validate_dataset_quality(profile)

            st.session_state["df"] = df
            st.session_state["profile"] = profile
            st.session_state["quality_warnings"] = warnings
            st.session_state["conversation_history"] = []
            st.session_state["chat_messages"] = []
            st.session_state["_last_uploaded_name"] = uploaded_file.name

            st.sidebar.success(f"Loaded '{uploaded_file.name}' — {len(df)} rows, {len(df.columns)} columns.")

        except (DataIngestionError, DataValidationError) as exc:
            st.sidebar.error(f"Could not load file: {exc}")
        except Exception as exc:
            st.sidebar.error("An unexpected error occurred while loading the file.")
            st.sidebar.caption(f"Technical details: {exc}")

if st.session_state["quality_warnings"]:
    with st.sidebar.expander("⚠️ Data quality warnings"):
        for w in st.session_state["quality_warnings"]:
            st.write(f"- {w}")


st.title("AI Data Analyst Agent")

if st.session_state["df"] is None:
    st.info("👈 Upload a dataset in the sidebar to begin.")
    st.stop()

df = st.session_state["df"]
profile = st.session_state["profile"]

with st.expander("📋 Dataset preview", expanded=False):
    st.dataframe(df.head(20))
    st.caption(profile.to_prompt_context())


st.subheader("Ask a question about your data")

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_spec"):
            st.plotly_chart(go.Figure(msg["chart_spec"]), use_container_width=True)
        if msg.get("code"):
            with st.expander("View generated code"):
                st.code(msg["code"], language="python")


user_question = st.chat_input("e.g. What is the total revenue by region?")

if user_question:
    st.session_state["chat_messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                initial_state = {
                    "user_question": user_question,
                    "dataframe": df,
                    "dataset_profile": profile,
                    "conversation_history": st.session_state["conversation_history"],
                    "retry_count": 0,
                }
                result = st.session_state["agent_app"].invoke(initial_state)
                final = result["final_response"]

                st.session_state["conversation_history"] = result.get(
                    "conversation_history", st.session_state["conversation_history"]
                )

                if final.get("success"):
                    st.markdown(final["insight"])
                    if final.get("chart_spec"):
                        st.plotly_chart(go.Figure(final["chart_spec"]), use_container_width=True)
                    with st.expander("View generated code"):
                        st.code(final["code"], language="python")

                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": final["insight"],
                        "chart_spec": final.get("chart_spec"),
                        "code": final.get("code"),
                    })
                else:
                    error_msg = (
                        f"I wasn't able to answer that confidently. "
                        f"Reason: {final.get('error', 'Unknown issue.')}\n\n"
                        f"Try rephrasing your question, or asking something more specific."
                    )
                    st.warning(error_msg)
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": error_msg,
                    })

            except AnalystAgentError as exc:
                st.error(f"Something went wrong: {exc}")
            except Exception as exc:
                st.error("An unexpected error occurred while processing your question.")
                st.caption(f"Technical details: {exc}")