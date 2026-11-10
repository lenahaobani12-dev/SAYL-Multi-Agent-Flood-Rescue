
from langsmith import Client
from sayl.severity import analyze, load_dataset

DATASET = "SAYL-Arabic-Severity"
client = Client()

# Load the existing labeled Arabic messages
texts, labels = load_dataset()

# Upload examples only if the dataset is empty
if not any(client.list_examples(dataset_name=DATASET)):
    examples = [
        {
            "inputs": {"text": text},
            "reference_outputs": {"severity": int(label)},
        }
        for text, label in zip(texts, labels)
    ]

    client.create_examples(
        dataset_name=DATASET,
        examples=examples,
    )
    print(f"Uploaded {len(examples)} examples.")

# Run the existing SAYL classifier
def predict(inputs: dict) -> dict:
    result = analyze(inputs["text"])
    return {"severity": int(result["severity"])}

# Compare prediction with the reference label
def severity_accuracy(outputs: dict, reference_outputs: dict):
    return {
        "key": "severity_exact_match",
        "score": int(
            outputs["severity"] == reference_outputs["severity"]
        ),
    }

# Create the LangSmith experiment
results = client.evaluate(
    predict,
    data=DATASET,
    evaluators=[severity_accuracy],
    experiment_prefix="SAYL-Severity-Evaluation",
    max_concurrency=1,
)

list(results)
print("Evaluation completed. Check LangSmith Experiments.")
