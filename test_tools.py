from ai_data_agent.ingestion.loaders import load_dataset
from ai_data_agent.ingestion.profiler import profile_dataset
from ai_data_agent.tools.dataset_tools import inspect_dataset, describe_dataset, DescribeDatasetInput

with open("data/sample_sales.csv", "rb") as f:
    file_bytes = f.read()

df = load_dataset(file_bytes, "sample_sales.csv")
profile = profile_dataset(df)

inspection = inspect_dataset(profile)
print("INSPECTION:", inspection.model_dump())

description = describe_dataset(df, DescribeDatasetInput())
print("DESCRIPTION:", description.model_dump())

# Test the defensive handling of a bad column name
bad_description = describe_dataset(df, DescribeDatasetInput(columns=["revenue", "not_a_real_column"]))
print("BAD COLUMN TEST:", bad_description.model_dump())