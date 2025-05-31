# Gene Expression Explorer

A web-based tool for exploring gene expression differences between healthy and disease samples using public datasets. Designed for researchers, students, and clinicians to visualize, analyze, and interpret gene expression data without coding.

---

## Features

- **Gene Search:** Instantly look up any gene and see its expression in selected datasets.
- **Multiple Datasets:** Analyze gene expression across several public GEO datasets.
- **Visualization:** Generate bar, box, or violin plots comparing healthy vs. disease samples.
- **Statistical Analysis:** View log2 fold change and p-value for gene expression differences.
- **Gene Information:** Fetches gene details (name, function, chromosome, related diseases, etc.).
- **GO Enrichment:** Shows biological processes and pathways related to the gene.
- **Top Genes Table:** Displays the most differentially expressed genes in the dataset.
- **Modern UI:** Responsive, clean interface with animations and loading indicators.
- **Error Handling:** Suggests similar gene names if the searched gene is not found.

---


## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **(Optional) Download additional datasets if needed.**

---

## Usage

1. **Run the application:**
   ```
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Enter a gene symbol, select a dataset and plot type, then view results.**

---

## Project Structure

- `app.py` – Main Flask application.
- `data/` – Cleaned gene expression and label CSV files.
- `static/` – Static files (CSS, images, etc.).
- `templates/` – HTML templates for the web interface.
- `process.py` – Data processing scripts.
- `train.py` – Example machine learning script (optional).

---

## Datasets

- Uses public datasets from GEO (e.g., GSE7305, GSE19804, GSE42568).
- Each dataset includes expression values and sample labels (healthy/disease).

---

## Technologies Used

- **Backend:** Python, Flask, pandas, numpy, scipy, matplotlib, seaborn, requests
- **Frontend:** Jinja2, Tailwind CSS, custom CSS, JavaScript

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

- Gene expression data from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- Gene info from [MyGene.info](https://mygene.info/)
- GO enrichment via [g:Profiler](https://biit.cs.ut.ee/gprofiler/)

---

*Feel free to contribute or open issues for suggestions and improvements!*