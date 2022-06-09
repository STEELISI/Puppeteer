# Project Setup

1. Create a Python 3.8 Anaconda environment (or your favorite other means of creating a virtual environment):
```
conda create -y --name puppeteer python=3.8
conda activate puppeteer  
```
2. Install requirements:
```
pip install -r requirements.txt
python -m snips_nlu download en
python -m spacy download en_core_web_lg
```
3. Confirm install:
```
python -c "import puppeteer"
```

# Contributing

Run `make precommit` before commiting.  
