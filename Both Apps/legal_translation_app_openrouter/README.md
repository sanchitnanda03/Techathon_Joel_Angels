# Legal Translation App (OpenRouter Edition)

This Streamlit app uses OpenRouter's LLaMA 3 to translate NDA contracts from Language Y into Language X using RAG (Retrieval-Augmented Generation).

## Setup

```bash
pip install -r requirements.txt
streamlit run legal_translation_app.py
```

## Notes

- You'll need an OpenRouter API key: https://openrouter.ai
- Place 5 `.txt` NDA reference documents in the `reference_docs/` folder.
