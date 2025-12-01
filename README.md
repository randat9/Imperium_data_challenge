# Imperium Data Challenge – Excel-based Clustering

## Overview
This project provides a complete solution for the Imperium Data Challenge.  
It processes the provided Excel file (`data/data_challenge_imperium.xlsx`), extracts numeric features, applies dimensionality reduction (PCA), performs clustering (KMeans), and outputs both a labeled dataset and a visualization.

## Project Structure
imperium-data-challenge/ ├── data/ │ └── data_challenge_imperium.xlsx ├── src/ │ └── solution.py ├── artifacts/ │ ├── cluster_assignments.parquet │ └── clusters_excel.png └── README.md

Kod

## Methodology
- **Data loading:** The script merges all non-empty sheets from the Excel file.  
- **Feature selection:** Numeric columns (`Balance`, `Price`, `Profit`) are converted to floats and cleaned.  
- **Standardization:** Features are scaled to ensure comparability.  
- **Dimensionality reduction:** PCA reduces the dataset to two principal components for visualization.  
- **Clustering:** KMeans groups the data into clusters based on similarity.  
- **Output:** Results are saved as both a parquet file and a scatter plot.

## Outputs
- `artifacts/cluster_assignments.parquet`  
  Contains the original rows (with usable numeric features), plus:
  - `PCA1` and `PCA2`: coordinates in reduced PCA space  
  - `cluster`: assigned cluster label  

- `artifacts/clusters_excel.png`  
  A scatter plot of the PCA results, with points color-coded by cluster.

## How to Run
From the project root:
```bash
python src/solution.py
Artifacts will be generated in the artifacts/ folder.

Notes
The pipeline is modular and can be extended to other datasets or instruments.

If additional numeric columns are available, they can be added to the feature selection step.

The clustering result demonstrates how unsupervised learning can uncover hidden structure in financial data.
