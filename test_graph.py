from ai_data_agent.logging_config import configure_logging
configure_logging()

from ai_data_agent.agent.graph import build_graph
from ai_data_agent.ingestion.loaders import load_dataset
from ai_data_agent.ingestion.profiler import profile_dataset

with open("data/sample_sales.csv", "rb") as f:
    file_bytes = f.read()

df = load_dataset(file_bytes, "sample_sales.csv")
profile = profile_dataset(df)

app = build_graph()

# Turn 1
state1 = {
    "user_question": "What is the total revenue by region?",
    "dataframe": df,
    "dataset_profile": profile,
    "conversation_history": [],
    "retry_count": 0,
}
result1 = app.invoke(state1)
print("TURN 1 CODE:", result1["final_response"]["code"])
print("TURN 1 INSIGHT:", result1["final_response"]["insight"])

# Turn 2 — a genuine follow-up, referencing "that" and "instead"
state2 = {
    "user_question": "Now show that as a chart sorted from highest to lowest instead.",
    "dataframe": df,
    "dataset_profile": profile,
    "conversation_history": result1["conversation_history"],
    "retry_count": 0,
}
result2 = app.invoke(state2)
print("\nTURN 2 CODE:", result2["final_response"]["code"])
print("TURN 2 CHART TYPE:", result2["final_response"]["chart_type"])
print("TURN 2 INSIGHT:", result2["final_response"]["insight"])