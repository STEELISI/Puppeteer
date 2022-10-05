from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer # type: ignore

NLI_MODEL = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli") # type: ignore
NLI_TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-large-mnli") # type: ignore

NEURAL_DIALOG_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  # type: ignore
NEURAL_DIALOG_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium") # type: ignore
