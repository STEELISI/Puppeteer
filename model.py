from transformers import AutoModelForSequenceClassification, AutoTokenizer # type: ignore

NLI_MODEL = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli') # type: ignore
NLI_TOKENIZER = AutoTokenizer.from_pretrained('facebook/bart-large-mnli') # type: ignore
