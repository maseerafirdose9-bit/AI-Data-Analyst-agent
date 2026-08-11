from ai_data_agent.ingestion.loaders import load_dataset
from ai_data_agent.ingestion.profiler import profile_dataset
from ai_data_agent.ingestion.validators import validate_dataset_quality

with open("data/sample_sales.csv", "rb") as f:
    file_bytes = f.read()

df = load_dataset(file_bytes, "sample_sales.csv")
profile = profile_dataset(df)
print(profile.to_prompt_context())

warnings = validate_dataset_quality(profile)
print(warnings)