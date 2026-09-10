from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)

result = classifier(
    "Quantum computing is interesting."
)

print(result)