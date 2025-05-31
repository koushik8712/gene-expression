import GEOparse
import pandas as pd
import os

def process_gse(gse_id):
    print(f"\n📦 Processing {gse_id}...")
    file_path = f"./data/{gse_id}_family.soft.gz"
    gse = GEOparse.get_GEO(filepath=file_path, annotate_gpl=True)

    # Expression matrix
    expression_data = gse.pivot_samples("VALUE")

    # Platform
    platform_id = list(gse.gpls.keys())[0]
    print(f"🧬 Platform: {platform_id}")
    gpl = gse.gpls[platform_id]

    # Probe to gene mapping
    probe_map = {}
    for _, row in gpl.table.iterrows():
        if 'Gene Symbol' in row and row['Gene Symbol'] and row['Gene Symbol'] != '---':
            probe_map[row['ID']] = row['Gene Symbol']

    expression_data['Gene Symbol'] = expression_data.index.map(probe_map)
    expression_data = expression_data.dropna(subset=['Gene Symbol'])

    # Group by gene symbol
    expression_by_gene = expression_data.groupby('Gene Symbol').mean()

    # Save expression matrix
    expression_file = f"data/{gse_id}_expression_clean.csv"
    expression_by_gene.to_csv(expression_file)
    print(f"✅ Expression data saved to {expression_file}")

    # Generate sample labels
    meta_info = []
    for sample in gse.metadata['sample_id']:
        gsm = gse.gsms[sample]
        title = gsm.metadata.get('title', [''])[0].lower()

        if "normal" in title:
            label = "healthy"
        elif "tumor" in title or "cancer" in title or "disease" in title:
            label = "disease"
        else:
            label = "unknown"

        meta_info.append({"Sample": sample, "Label": label})  # Capitalized keys

    labels_df = pd.DataFrame(meta_info)
    labels_file = f"data/{gse_id}_labels.csv"
    labels_df.to_csv(labels_file, index=False)
    print(f"✅ Labels saved to {labels_file}")

# List of datasets to process
DATASETS = [ "GSE14905" ]

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Process each
for gse_id in DATASETS:
    process_gse(gse_id)
