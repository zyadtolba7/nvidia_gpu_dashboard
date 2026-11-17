# GPU Manufacturing Story Dashboard

Interactive Streamlit dashboard inspired by the supplied storytelling dashboard screenshots. It reads the synthetic NVIDIA GPU manufacturing workbook, cleans intentionally messy fields, and presents production, QA, process, and cost views with shared filters.

> The dataset is fictional and for learning only. It is not actual NVIDIA production, supplier, or quality data.

The workbook keeps the source in `raw_gpu_manufacturing`, includes `clean_gpu_manufacturing` for the prepared records, and documents cleaning counts and rules in `cleaning_summary`. The dashboard loads the raw sheet and applies the same cleaning rules when it starts.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional AI Q&A

The dashboard works without an API key using locally calculated observations. To enable generated answers, add `OPENAI_API_KEY` to Streamlit Secrets or the environment. Only aggregate metrics for the currently filtered selection are sent; individual serials and record rows are not included.

For Streamlit Community Cloud, set `app.py` as the main file and include this folder's files in the connected GitHub repository.
