import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_excel_any_sheet(path: str) -> pd.DataFrame:
    """
    Loads the Excel file. If multiple sheets exist, concatenates them vertically,
    aligning common columns. Non-empty sheets only.
    """
    xls = pd.ExcelFile(path)
    sheets = []
    for name in xls.sheet_names:
        df = xls.parse(name)
        if df is not None and len(df) > 0:
            df["__sheet__"] = name
            sheets.append(df)
    if not sheets:
        raise ValueError("No non-empty sheets found in Excel.")
    # Align columns across sheets (outer join on columns, fill missing with NaN)
    all_cols = list(set().union(*[set(s.columns) for s in sheets]))
    sheets_aligned = [s.reindex(columns=all_cols) for s in sheets]
    merged = pd.concat(sheets_aligned, axis=0, ignore_index=True)
    return merged

def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # Wybierz tylko kolumny, które mają sens jako liczby
    numeric_cols = ["Balance", "Price", "Profit"]
    df_numeric = df[numeric_cols].copy()

    # Konwersja na liczby
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    # Usuń wiersze z brakami
    df_numeric = df_numeric.dropna()

    print(f"Numeric features selected: shape={df_numeric.shape}, cols={list(df_numeric.columns)}")
    return df_numeric

def choose_k(n_samples: int) -> int:
    """
    Simple heuristic for number of clusters based on dataset size.
    """
    if n_samples < 50:
        return 3
    if n_samples < 200:
        return 4
    if n_samples < 1000:
        return 5
    return 6

def main():
    # 1) Ensure artifacts exist
    os.makedirs("artifacts", exist_ok=True)

    # 2) Load Excel
    excel_path = "../data/data_challenge_imperium.xlsx"
    df_raw = load_excel_any_sheet(excel_path)
    print(f"Loaded Excel: shape={df_raw.shape}, sheets merged.")

    # 3) Select numeric features
    features = select_numeric_features(df_raw)
    print(f"Numeric features: shape={features.shape}")

    # 4) Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # 5) PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X)

    # 6) KMeans clustering
    k = choose_k(features.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # 7) Build result frame (preserve original indices for traceability)
    result = df_raw.loc[features.index].copy()
    result["PCA1"] = pca_2d[:, 0]
    result["PCA2"] = pca_2d[:, 1]
    result["cluster"] = clusters

    # 8) Save parquet
    out_parquet = "artifacts/cluster_assignments.parquet"
    result.to_parquet(out_parquet)
    print(f"Saved: {out_parquet}")

    # 9) Plot
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(result["PCA1"], result["PCA2"], c=result["cluster"], cmap="tab10", alpha=0.75, s=30)
    plt.title("Clusters from Excel data (PCA 2D)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    # Legend with cluster labels
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.75)
    plt.legend(handles, [f"Cluster {i}" for i in range(k)], title="Clusters", loc="best", frameon=True)
    out_png = "artifacts/clusters_excel.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print("Artifacts saved in:", os.path.abspath("artifacts"))
if __name__ == "__main__":
    main()