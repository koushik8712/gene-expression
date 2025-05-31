from flask import Flask, request, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import ttest_ind
import pickle
from functools import lru_cache
import requests
import random
import difflib

app = Flask(__name__)

# Define datasets and file paths
DATASETS = {
    "GSE7305": {
        "expression": "data/GSE7305_expression_clean.csv",
        "labels": "data/GSE7305_labels.csv"
    },
    "GSE19804": {
        "expression": "data/GSE19804_expression_clean.csv",
        "labels": "data/GSE19804_labels.csv"
    },
    "GSE42568": {
        "expression": "data/GSE42568_expression_clean.csv",
        "labels": "data/GSE42568_labels.csv"
    },
     "GSE44077": {
        "expression": "data/GSE44077_expression_clean.csv",
        "labels": "data/GSE44077_labels.csv"
    },
     "GSE44076": {
        "expression": "data/GSE44076_expression_clean.csv",
        "labels": "data/GSE44076_labels.csv"
},
      "GSE14905": {
        "expression": "data/GSE14905_expression_clean.csv",
        "labels": "data/GSE14905_labels.csv"
}
}

# Global cache for datasets and top genes
DATA_CACHE = {}
TOP_GENES_CACHE = {}

def load_dataset(dataset_name):
    """Load and cache dataset"""
    if dataset_name not in DATA_CACHE:
        print(f"Loading {dataset_name}...")
        try:
            # Try to load from pickle first (much faster)
            pickle_expr = f"data/{dataset_name}_expression.pkl"
            pickle_labels = f"data/{dataset_name}_labels.pkl"
            
            if os.path.exists(pickle_expr) and os.path.exists(pickle_labels):
                expression = pd.read_pickle(pickle_expr)
                labels = pd.read_pickle(pickle_labels)
            else:
                # Load from CSV and save as pickle for next time
                expression = pd.read_csv(DATASETS[dataset_name]["expression"], index_col=0)
                labels = pd.read_csv(DATASETS[dataset_name]["labels"])
                # Save as pickle for faster loading next time
                expression.to_pickle(pickle_expr)
                labels.to_pickle(pickle_labels)
            
            DATA_CACHE[dataset_name] = {
                'expression': expression,
                'labels': labels,
                'labels_dict': labels.set_index("sample")["label"].to_dict()
            }
            print(f"Loaded {dataset_name} with {expression.shape[0]} genes and {expression.shape[1]} samples")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    return DATA_CACHE[dataset_name]

def precompute_top_genes(dataset_name, data):
    """Precompute top differential genes"""
    if dataset_name in TOP_GENES_CACHE:
        return TOP_GENES_CACHE[dataset_name]
    
    print(f"Computing top genes for {dataset_name}...")
    try:
        expression = data['expression']
        labels_dict = data['labels_dict']
        
        healthy_samples = [s for s in expression.columns if labels_dict.get(s) == "healthy"]
        disease_samples = [s for s in expression.columns if labels_dict.get(s) == "disease"]
        
        if len(healthy_samples) > 1 and len(disease_samples) > 1:
            # Vectorized computation for speed
            healthy_data = expression[healthy_samples]
            disease_data = expression[disease_samples]
            
            # Calculate means
            healthy_means = healthy_data.mean(axis=1)
            disease_means = disease_data.mean(axis=1)
            
            # Calculate absolute differences
            abs_diffs = np.abs(healthy_means - disease_means)
            
            # Get top 10 genes by absolute difference
            top_indices = abs_diffs.nlargest(10).index
            
            gene_stats = []
            for gene_name in top_indices:
                healthy_vals = healthy_data.loc[gene_name].dropna()
                disease_vals = disease_data.loc[gene_name].dropna()
                
                mean_healthy = healthy_vals.mean()
                mean_disease = disease_vals.mean()
                diff = mean_healthy - mean_disease
                log2fc = np.log2((mean_healthy + 1e-9) / (mean_disease + 1e-9))
                
                # Only compute t-test for top genes
                if len(healthy_vals) > 1 and len(disease_vals) > 1:
                    _, pval = ttest_ind(healthy_vals, disease_vals, equal_var=False)
                else:
                    pval = np.nan
                
                gene_stats.append({
                    "gene": gene_name,
                    "mean_healthy": mean_healthy,
                    "mean_disease": mean_disease,
                    "abs_diff": abs(diff),
                    "log2fc": log2fc,
                    "pval": pval
                })
            
            TOP_GENES_CACHE[dataset_name] = gene_stats
            print(f"Computed top genes for {dataset_name}")
            # --- Write to CSV for download ---
            csv_path = os.path.join(app.root_path, 'static', 'top_genes.csv')
            pd.DataFrame(gene_stats).to_csv(csv_path, index=False)
        else:
            TOP_GENES_CACHE[dataset_name] = []
    except Exception as e:
        print(f"Error computing top genes for {dataset_name}: {e}")
        TOP_GENES_CACHE[dataset_name] = []
    
    return TOP_GENES_CACHE[dataset_name]

def create_optimized_plot(gene, dataset_name, data, plot_type, plot_path):
    """Create plot with optimized settings"""
    try:
        expression = data['expression']
        labels_dict = data['labels_dict']

        if gene not in expression.index:
            return False

        expr_values = expression.loc[gene]
        samples = expr_values.index

        # Filter samples and create plot data
        plot_data = []
        for s in samples:
            label = labels_dict.get(s)
            if label in ["healthy", "disease"]:
                plot_data.append({"Sample": s, "Label": label, "Expression": expr_values[s]})

        if not plot_data:
            return False

        df_plot = pd.DataFrame(plot_data)

        plt.figure(figsize=(8, 6))
        plt.clf()  # Clear the figure

        # Calculate statistics for annotation
        stats = df_plot.groupby('Label')['Expression'].agg(['mean', 'median'])

        # Plot type
        if plot_type == "bar":
            ax = sns.barplot(data=df_plot, x="Label", y="Expression", ci="sd", 
                             palette={"healthy": "lightblue", "disease": "lightcoral"})
        elif plot_type == "box":
            ax = sns.boxplot(data=df_plot, x="Label", y="Expression",
                             palette={"healthy": "lightblue", "disease": "lightcoral"})
            # Overlay all data points
            sns.stripplot(data=df_plot, x="Label", y="Expression", 
                          color='gray', size=5, jitter=True, ax=ax, dodge=True, alpha=0.6)
        else:  # violin
            ax = sns.violinplot(data=df_plot, x="Label", y="Expression",
                                palette={"healthy": "lightblue", "disease": "lightcoral"}, inner="box")
            # Overlay all data points
            sns.stripplot(data=df_plot, x="Label", y="Expression", 
                          color='gray', size=5, jitter=True, ax=ax, dodge=True, alpha=0.6)

        # Add mean and median annotations at their actual y-values
        for i, label in enumerate(['healthy', 'disease']):
            mean_val = stats.loc[label, 'mean']
            median_val = stats.loc[label, 'median']
            # Draw a horizontal line for mean (blue)
            ax.hlines(mean_val, i-0.3, i+0.3, colors='blue', linestyles='-', lw=2, alpha=0.8)
            # Draw a horizontal line for median (red)
            ax.hlines(median_val, i-0.3, i+0.3, colors='red', linestyles='--', lw=2, alpha=0.8)
            # Annotate mean and median
            ax.text(i, mean_val + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    f'Mean: {mean_val:.2f}', 
                    ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
            ax.text(i, median_val - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    f'Median: {median_val:.2f}', 
                    ha='center', va='top', fontsize=10, color='red', fontweight='bold')

        # Add grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        # Set axis labels and title
        ax.set_xlabel("Label", fontsize=13, fontweight='bold')
        ax.set_ylabel("Expression", fontsize=13, fontweight='bold')
        ax.set_title(f"{gene} Expression in {dataset_name}", fontsize=15, fontweight='bold', pad=20)

        # Tighter layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"Error creating plot: {e}")
        return False

def fetch_gene_info(symbol):
    """Fetch gene info from MyGene.info API"""
    try:
        url = f"https://mygene.info/v3/query"
        params = {
            "q": symbol,
            "fields": "symbol,name,summary,chrom,genomic_pos,entrezgene,alias,uniprot,disgenet",
            "species": "human",
            "size": 1
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            return None
        gene = hits[0]
        # Disease links (from disgenet, if available)
        diseases = []
        if "disgenet" in gene and isinstance(gene["disgenet"], list):
            for d in gene["disgenet"][:5]:  # Limit to 5
                diseases.append({
                    "name": d.get("disease_name"),
                    "score": d.get("score"),
                    "id": d.get("diseaseid")
                })
        return {
            "symbol": gene.get("symbol"),
            "name": gene.get("name"),
            "summary": gene.get("summary"),
            "chrom": gene.get("chrom"),
            "location": gene.get("genomic_pos", {}).get("chr", None),
            "entrez": gene.get("entrezgene"),
            "uniprot": gene.get("uniprot", {}).get("Swiss-Prot") if "uniprot" in gene else None,
            "diseases": diseases
        }
    except Exception as e:
        print(f"Gene info fetch error: {e}")
        return None

def go_enrichment(genes):
    """Query g:Profiler for GO enrichment given a list of gene symbols."""
    try:
        url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
        payload = {
            "organism": "hsapiens",
            "query": genes,
            "sources": ["GO:BP", "GO:MF", "GO:CC"],  # Biological Process, Molecular Function, Cellular Component
            "user_threshold": 0.05,
            "no_evidences": True
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("result", [])
        # Return top 5 enriched terms
        go_terms = []
        for r in results[:5]:
            go_terms.append({
                "name": r["name"],
                "source": r["source"],
                "term_id": r["native"],
                "p_value": r["p_value"]
            })
        return go_terms
    except Exception as e:
        print(f"GO enrichment error: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    gene = None
    plot_exists = False
    gene_info = None
    top_genes = []
    selected_dataset = None
    plot_type = None
    go_terms = []  # Initialize GO terms

    if request.method == "POST":
        selected_dataset = request.form.get("dataset", "GSE7305")
        plot_type = request.form.get("plot_type", "bar")
        data = load_dataset(selected_dataset)
        top_genes = precompute_top_genes(selected_dataset, data) if data else []

        gene = request.form.get("gene", "").strip().upper()
        if not gene:
            # Recommend a random top-5 gene if available
            recommended_gene = None
            if top_genes:
                top5 = top_genes[:5]
                recommended_gene = random.choice(top5)["gene"]
            result = {
                "error": "Please enter a gene symbol.",
                "recommendation": f"Try one of these interesting genes: {', '.join([g['gene'] for g in top_genes[:5]])}" if top_genes else None,
                "recommended_gene": recommended_gene
            }
            return render_template("index.html", 
                                gene=None,
                                result=result,
                                selected_dataset=selected_dataset,
                                datasets=DATASETS.keys(),
                                plot_exists=False,
                                plot_type=plot_type,
                                top_genes=top_genes,
                                gene_info=None)
        
        # Load selected dataset
        if not data:
            result = {"error": f"Could not load dataset {selected_dataset}"}
            return render_template("index.html",
                                gene=gene,
                                result=result,
                                selected_dataset=selected_dataset,
                                datasets=DATASETS.keys(),
                                plot_exists=False,
                                plot_type=plot_type,
                                top_genes=top_genes,
                                gene_info=None)
        
        expression = data['expression']
        if gene not in expression.index:
            # Suggest closest gene symbol
            all_genes = list(expression.index)
            suggestion = difflib.get_close_matches(gene, all_genes, n=1)
            result = {
                "error": f"Gene '{gene}' not found in dataset {selected_dataset}.",
                "suggestion": f"Did you mean: {suggestion[0]}?" if suggestion else None
            }
            return render_template("index.html",
                gene=gene,
                result=result,
                selected_dataset=selected_dataset,
                datasets=DATASETS.keys(),
                plot_exists=False,
                plot_type=plot_type,
                top_genes=top_genes,
                gene_info=None
            )
        
        try:
            static_dir = os.path.join(app.root_path, 'static')
            os.makedirs(static_dir, exist_ok=True)
            plot_path = os.path.join(static_dir, "plot.png")
            labels_dict = data['labels_dict']
            expr_values = expression.loc[gene]
            samples = expr_values.index
            healthy_vals = [expr_values[s] for s in samples if labels_dict.get(s) == "healthy"]
            disease_vals = [expr_values[s] for s in samples if labels_dict.get(s) == "disease"]
            if not healthy_vals or not disease_vals:
                result = {"error": "Insufficient data for analysis"}
                plot_exists = False
            else:
                healthy_avg = np.mean(healthy_vals)
                disease_avg = np.mean(disease_vals)
                healthy_std = np.std(healthy_vals, ddof=1) if len(healthy_vals) > 1 else 0
                disease_std = np.std(disease_vals, ddof=1) if len(disease_vals) > 1 else 0
                log2fc = np.log2((healthy_avg + 1e-9) / (disease_avg + 1e-9))
                if len(healthy_vals) > 1 and len(disease_vals) > 1:
                    _, pval = ttest_ind(healthy_vals, disease_vals, equal_var=False)
                else:
                    pval = np.nan
                result = {
                    "dataset": selected_dataset,
                    "gene": gene,
                    "healthy_avg": round(healthy_avg, 2),
                    "disease_avg": round(disease_avg, 2),
                    "healthy_std": round(healthy_std, 2),
                    "disease_std": round(disease_std, 2),
                    "log2fc": round(log2fc, 2),
                    "pval": f"{pval:.3e}" if not np.isnan(pval) else "N/A"
                }
                plot_exists = create_optimized_plot(gene, selected_dataset, data, plot_type, plot_path)
        except Exception as e:
            result = {"error": f"Error processing request: {str(e)}"}
            plot_exists = False

        gene_info = fetch_gene_info(gene)

        # --- GO enrichment analysis ---
        if result and "log2fc" in result:
            go_terms = go_enrichment([gene])
            # If no GO terms found, try with a random top gene
            if not go_terms and top_genes:
                fallback_gene = random.choice(top_genes)["gene"]
                if fallback_gene != gene:
                    go_terms = go_enrichment([fallback_gene])
    else:
        # On GET, do not load any data or compute anything
        selected_dataset = None
        plot_type = None
        top_genes = []
        gene_info = None
        plot_exists = False
        result = None
        gene = None

    return render_template("index.html",
                         gene=gene,
                         result=result,
                         selected_dataset=selected_dataset,
                         datasets=DATASETS.keys(),
                         plot_exists=plot_exists,
                         plot_type=plot_type,
                         top_genes=top_genes,
                         gene_info=gene_info,
                         go_terms=go_terms
)

if __name__ == "__main__":
    print("Starting Flask app...")
    
    # Pre-load the most common dataset
    print("Pre-loading default dataset...")
    default_data = load_dataset("GSE7305")
    if default_data:
        precompute_top_genes("GSE7305", default_data)
    
    print("App ready!")
    app.run(debug=False, threaded=True)  # Disable debug mode and enable threading