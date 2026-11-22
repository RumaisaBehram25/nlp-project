# Log anomaly detection (NLP project)

Upload this folder to Google Drive as `MyDrive/nlp_log_anomaly` (keep `data/raw/HDFS_v1.zip` and `data/raw/BGL.zip`).
Open `log_anomaly_detection.ipynb` in Colab, pick a GPU runtime and run the cells top to bottom.

Sections: setup, HDFS and BGL parsing, SBERT embeddings and reworded templates, TF-IDF baseline, DeepLog LSTM,
transformer (ids vs SBERT), robustness, cross-system, results and plots.

Models are saved to `models/`, results to `results/`.

## Web app (`app/`)

- `index.html`: results dashboard (works by double-click).
- `try.html`: upload logs, pick models, see predictions and scores. Needs the server.

Run the server (needs `pip install -r app/requirements.txt`):

    cd app && python3 server.py        # then open http://localhost:8000/try.html

Or in Colab: upload the `app` folder to `MyDrive/nlp_log_anomaly/app` and run `app/run_in_colab.ipynb`.


## What is in this repo
- `log_anomaly_detection.ipynb` - clean notebook (no outputs), run it in Colab.
- `results/log_anomaly_detection.ipynb` - the same notebook after a full run, with all outputs.
- `results/` - result tables (`results.csv`, `table_*.csv`) and `figures/`.
- `app/models/` - the 10 trained models used by the web app.
- `app/` - LogLens web app (dashboard `index.html`, Try it `try.html` + Python server).
- The raw HDFS/BGL datasets are not included (too big); download them and put them in `data/raw/`.
