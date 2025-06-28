# miRNA Expression Analysis in Prostate Cancer Progression: A Comprehensive Multi-Modal Approach

## Overview

This repository contains a comprehensive computational pipeline for analyzing microRNA (miRNA) expression patterns in prostate cancer progression using the GSE211692 dataset. Our multi-modal research approach combines machine learning classification, gene set enrichment analysis (GSEA), hierarchical clustering, pathway analysis, and network biology to identify differentially expressed miRNAs that serve as potential biomarkers for prostate cancer diagnosis, staging, and progression monitoring.

## Research Objectives

1. **Dual-Approach Classification**: Develop machine learning models for both stage-based and disease-based classification of prostate cancer samples
2. **Comprehensive GSEA**: Perform gene set enrichment analysis across both progression stages and disease states
3. **Hierarchical Clustering**: Identify miRNA and gene expression clusters associated with cancer progression
4. **Pathway Analysis**: Map miRNA-regulated pathways that change across cancer progression
5. **Network Biology**: Construct and analyze miRNA-gene interaction networks
6. **Biomarker Discovery**: Identify predictive miRNA signatures for clinical applications
7. **Progressive Analysis**: Track miRNA and pathway changes through cancer progression stages

## Dataset Information

- **Source**: Gene Expression Omnibus (GEO) accession GSE211692
- **Platform**: miRNA expression profiling
- **Sample Groups**:
  - Control samples (no cancer): 5,643 samples
  - Benign prostate disease samples: 230 samples
  - Prostate cancer samples by stage:
    - Stage 1, Stage 2, Stage 3, Stage 4
- **Total Features**: 2,570 miRNA expression measurements
- **Analysis Scope**: >16,000 individual expression measurements

## Repository Structure

```
miRNA_PC/
├── README.md                           # This comprehensive documentation
├── data_normalzied.ipynb              # Main data normalization and quality control
├── data_preprocessing/                 # Initial data exploration and sample categorization
│   ├── prostate_cancer_exploration.ipynb
│   └── series_matrix_processing.ipynb
├──
├── STAGE-BASED ANALYSIS PIPELINE:
├── stage_classification/              # Machine learning models for stage progression
│   ├── final_ctl_s1.ipynb            # Control vs Stage 1 classification
│   ├── final_s1_s2.ipynb             # Stage 1 vs Stage 2 classification
│   ├── final_s2_s3.ipynb             # Stage 2 vs Stage 3 classification
│   ├── final_s3_s4.ipynb             # Stage 3 vs Stage 4 classification
│   └── *.pkl                         # Trained models (SVM, XGBoost, Random Forest)
├── stage_gsea/                        # Gene Set Enrichment Analysis for stages
│   ├── progressive_pathways.ipynb    # Pathway changes across progression
│   ├── pathway_visualization.ipynb   # Visual pathway analysis
│   ├── miRNA_progression.ipynb       # miRNA expression progression analysis
│   ├── gene_progression.ipynb        # Gene expression progression analysis
│   ├── ctl_s1/, s1_s2/, s2_s3/, s3_s4/  # Stage-specific GSEA results
│   └── genecards_comparisons/        # GeneCards pathway integration
├── stage_clustering_GSEA/             # Hierarchical clustering for stages
│   ├── stage_clustering.ipynb        # Stage-specific clustering analysis
│   ├── combined_clustering_gsea.ipynb # Integrated clustering and GSEA
│   ├── miRNA/                        # miRNA clustering results
│   ├── genes/                        # Gene clustering results
│   └── gene_cards/                   # GeneCards clustering integration
├──
├── DISEASE-BASED ANALYSIS PIPELINE:
├── disease_classification/            # Machine learning for disease states
│   ├── final_ctl_b.ipynb            # Control vs Benign classification
│   ├── final_ctl_c.ipynb            # Control vs Cancer classification
│   ├── final_b_c.ipynb              # Benign vs Cancer classification
│   └── *.pkl                        # Trained disease classification models
├── disease_gsea/                      # Disease-focused GSEA analysis
│   ├── progressive_pathways.ipynb    # Disease progression pathway analysis
│   ├── pathway_visualization.ipynb   # Disease pathway visualization
│   ├── miRNA_progression.ipynb       # Disease-specific miRNA analysis
│   ├── ctl_c/, ctl_b/, b_c/         # Disease comparison GSEA results
│   └── genecards_comparisons/        # Disease GeneCards integration
├── disease_clustering_GSEA/           # Disease-based clustering analysis
│   ├── stage_clustering.ipynb        # Disease clustering methodology
│   ├── combined_clustering_gsea.ipynb # Integrated disease clustering
│   ├── miRNA/                        # Disease miRNA clusters
│   ├── genes/                        # Disease gene clusters
│   └── gene_cards/                   # Disease GeneCards clusters
├──
├── INTEGRATED ANALYSIS:
├── GSEA/                              # Core GSEA analysis infrastructure
│   ├── miRNA/                        # All miRNA GSEA results (50-feature sets)
│   ├── networks/                     # Network analysis components
│   │   ├── filtered_gene_miRNA_network.graphml
│   │   └── filtered_genes_for_enrichment_analysis.csv
│   └── misc/                         # Miscellaneous analysis files
├── network_figures/                   # Network analysis and visualization
│   ├── network_figure_2.ipynb       # Primary network analysis
│   ├── miRNA_disease_network.graphml # Disease network structure
│   ├── predictive_miRNA_genes.csv   # Key miRNA-gene interactions (1,413 genes)
│   ├── overlapping_targeted_genes_matrix_sorted_clustered.xlsx
│   └── Prostate_miRNA_Network.gephi  # Gephi network file
├── results & figures/                 # Publication-ready results
│   ├── output/                       # Main figures and tables
│   │   ├── figure_1.ipynb           # Primary differential expression results
│   │   ├── figure_2.ipynb           # Machine learning performance analysis
│   │   ├── figure_3.ipynb           # ROC curves and validation metrics
│   │   ├── figure_4.ipynb           # Network and pathway visualization
│   │   ├── supplemental_table_1.xlsx # Complete differential expression data
│   │   ├── top_20_miRNA_comparisons.xlsx # Top miRNA features per comparison
│   │   ├── table2.xlsx              # Summary statistics and performance
│   │   └── miRNA Disease Association Search.xlsx # Disease associations
│   ├── final_figures/               # Publication-quality figures
│   │   ├── biomarker_expression_heatmap.ipynb
│   │   └── figure_s1_final.csv      # Supplementary analysis results
│   ├── accuracy_heatmap.ipynb       # Model performance visualization
│   ├── stage_clustering.ipynb       # Stage clustering results
│   ├── supplementary_figure_1.ipynb # Additional validation analyses
│   └── figure_s1_data/              # Supplementary data files
└── Configuration Files:
    ├── .gitignore                    # Git configuration
    └── .vscode/                      # Development environment settings
```

## Methodology

### Multi-Modal Analysis Framework

Our research employs a comprehensive multi-modal approach with two parallel analysis pipelines:

#### 1. Stage-Based Analysis Pipeline

- **Classification**: Control → Stage 1 → Stage 2 → Stage 3 → Stage 4 progression
- **GSEA**: Stage-specific pathway enrichment analysis
- **Clustering**: Hierarchical clustering of miRNAs across stages

#### 2. Disease-Based Analysis Pipeline

- **Classification**: Control vs Benign vs Cancer comparisons
- **GSEA**: Disease state-specific pathway analysis
- **Clustering**: Disease-focused miRNA and gene clustering

### Machine Learning Classification

We employed three complementary algorithms across both pipelines:

- **Support Vector Machine (SVM)**: Linear classification with feature importance
- **XGBoost**: Gradient boosting with feature ranking
- **Random Forest**: Ensemble method for robust predictions

#### Comprehensive Classification Tasks:

**Stage Progression Analysis:**

1. Control vs Stage 1 (83 features identified)
2. Stage 1 vs Stage 2 (29 features identified)
3. Stage 2 vs Stage 3 (56 features identified)
4. Stage 3 vs Stage 4 (52 features identified)

**Disease State Analysis:**

1. Control vs Benign disease (sampling ratio: 1:24.5)
2. Control vs Cancer (comprehensive cancer detection)
3. Benign vs Cancer (disease progression markers)

### Gene Set Enrichment Analysis (GSEA)

**Dual GSEA Framework:**

- **miEAA Integration**: miRNA enrichment analysis with pathway annotation
- **GeneCards Integration**: Comprehensive pathway mapping and scoring
- **Progressive Analysis**: Tracking pathway changes across cancer progression
- **Cross-Platform Validation**: Multiple enrichment databases

**Key GSEA Results:**

- **Cell Cycle, Mitotic**: 3.56 log fold change across progression
- **Innate Immune System**: 3.56 log fold change
- **RNA Polymerase I Promoter Opening**: 3.21 log fold change
- **Metabolism of Proteins**: 2.75 log fold change

### Hierarchical Clustering Analysis

**Multi-Level Clustering:**

- **miRNA Clustering**: 62 unique miRNAs clustered into 4 functional groups
- **Gene Clustering**: Disease and stage-specific gene clusters
- **Pathway Clustering**: Functional pathway groupings
- **Cross-Validation**: Silhouette analysis for optimal cluster numbers

**Cluster Characteristics:**

- **Cluster 1**: 5 miRNAs (early progression markers)
- **Cluster 2**: 21 miRNAs (intermediate progression)
- **Cluster 3**: 6 miRNAs (aggressive progression)
- **Cluster 4**: 18 miRNAs (late-stage markers)

### Network Analysis

**Comprehensive Network Construction:**

- **1,413 genes** with significant miRNA interactions
- **14 top miRNAs** identified as network hubs
- **Multi-target analysis**: miRNAs targeting multiple genes
- **Pathway integration**: Network-pathway cross-analysis

**Key Network Hubs:**

- **FOXK1**: 4-5 miRNA interactions across comparisons
- **BARHL1**: Multiple interaction networks
- **ERC1**: 3-miRNA regulatory hub

### Statistical Rigor

- **Multiple Testing Correction**: FDR correction across all analyses
- **Cross-Validation**: K-fold validation for all machine learning models
- **Feature Selection**: Top 50 features per comparison
- **Sampling Strategies**: Under-sampling and over-sampling for class imbalance
- **Performance Metrics**: Accuracy, ROC-AUC, sensitivity, specificity

## Key Findings

### miRNA Biomarker Discovery

**Stage Progression Biomarkers:**

- **hsa-miR-6769b-5p**: Present in multiple stage transitions
- **hsa-miR-548h-5p**: Critical for S1→S2 and S2→S3→S4 progression
- **hsa-miR-139-3p**: Key regulator in S1→S2 and S3→S4 transitions
- **hsa-miR-6756-5p**: Links Control→S1 and S3→S4 stages

**Disease State Biomarkers:**

- **Control vs Cancer**: hsa-miR-1469, hsa-miR-5100 (critical differentiators)
- **Progression Monitoring**: Stage-specific miRNA signatures identified
- **Early Detection**: Control vs Stage 1 specific markers

### Machine Learning Performance

**Classification Accuracy:**

- **Stage Transitions**: 85-95% cross-validation accuracy
- **Disease States**: >90% accuracy for Control vs Cancer
- **Feature Consensus**: High agreement across SVM, XGBoost, and Random Forest
- **Balanced Performance**: Effective handling of class imbalance

### Pathway Analysis Results

**Progressive Pathway Changes:**

- **Cell Cycle Regulation**: Strongest progressive alteration (3.56 log fold change)
- **Immune System Dysregulation**: Significant immune pathway changes
- **Metabolic Reprogramming**: Progressive metabolic pathway alterations
- **Signal Transduction**: Comprehensive signaling pathway disruption

**Network Integration:**

- **1,413 target genes** identified with significant miRNA regulation
- **Multi-pathway integration**: Single miRNAs affecting multiple pathways
- **Hub gene identification**: Critical regulatory nodes in cancer progression

## Clinical Significance

### Diagnostic Applications

**Early Detection Biomarkers:**

1. **Control vs Stage 1**: 42 miRNA signature for early detection
2. **Benign vs Malignant**: Specific miRNA patterns for differential diagnosis
3. **Multi-modal validation**: Both stage and disease approaches confirm findings

### Prognostic Applications

**Stage Progression Monitoring:**

1. **S1→S2 Transition**: 15 miRNA progression signature
2. **S2→S3 Transition**: 28 miRNA advanced progression markers
3. **S3→S4 Transition**: 26 miRNA late-stage progression indicators

### Therapeutic Target Identification

**Network-Based Targets:**

- **FOXK1 regulatory network**: Multi-miRNA therapeutic target
- **Cell cycle pathway**: Targetable pathway disruptions
- **Immune modulation pathways**: Immunotherapy target identification

## Reproducibility Instructions

### Prerequisites

```bash
# Core Python packages
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Specialized analysis packages
networkx>=2.6.0
xlsxwriter>=3.0.0
upsetplot>=0.4.0
matplotlib-venn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0

# Machine learning and sampling
imblearn>=0.8.0
optuna>=2.10.0
joblib>=1.0.0

# Network analysis
gephi (external software for network visualization)
```

### Complete Analysis Pipeline

#### 1. Data Preprocessing and Normalization

```bash
# Initial data processing
jupyter notebook data_preprocessing/prostate_cancer_exploration.ipynb
jupyter notebook data_normalzied.ipynb
```

#### 2. Stage-Based Analysis Pipeline

```bash
# Stage classification models
jupyter notebook stage_classification/final_ctl_s1.ipynb
jupyter notebook stage_classification/final_s1_s2.ipynb
jupyter notebook stage_classification/final_s2_s3.ipynb
jupyter notebook stage_classification/final_s3_s4.ipynb

# Stage GSEA analysis
jupyter notebook stage_gsea/progressive_pathways.ipynb
jupyter notebook stage_gsea/pathway_visualization.ipynb
jupyter notebook stage_gsea/miRNA_progression.ipynb

# Stage clustering
jupyter notebook stage_clustering_GSEA/stage_clustering.ipynb
jupyter notebook stage_clustering_GSEA/combined_clustering_gsea.ipynb
```

#### 3. Disease-Based Analysis Pipeline

```bash
# Disease classification models
jupyter notebook disease_classification/final_ctl_b.ipynb
jupyter notebook disease_classification/final_ctl_c.ipynb
jupyter notebook disease_classification/final_b_c.ipynb

# Disease GSEA analysis
jupyter notebook disease_gsea/progressive_pathways.ipynb
jupyter notebook disease_gsea/pathway_visualization.ipynb

# Disease clustering
jupyter notebook disease_clustering_GSEA/combined_clustering_gsea.ipynb
```

#### 4. Network Analysis and Visualization

```bash
# Network construction and analysis
jupyter notebook network_figures/network_figure_2.ipynb

# Integrated network analysis
jupyter notebook GSEA/order_miRNA.ipynb
```

#### 5. Results Generation and Visualization

```bash
# Primary analysis figures
jupyter notebook results\ \&\ figures/output/figure_1.ipynb
jupyter notebook results\ \&\ figures/output/figure_2.ipynb
jupyter notebook results\ \&\ figures/output/figure_3.ipynb
jupyter notebook results\ \&\ figures/output/figure_4.ipynb

# Performance analysis
jupyter notebook results\ \&\ figures/accuracy_heatmap.ipynb
jupyter notebook results\ \&\ figures/stage_clustering.ipynb

# Supplementary analyses
jupyter notebook results\ \&\ figures/supplementary_figure_1.ipynb
jupyter notebook results\ \&\ figures/final_figures/biomarker_expression_heatmap.ipynb
```

### Data Requirements

**Primary Data:**

- `GSE211692_processed.txt`: Main expression data (place in `data/source_data/`)

**Generated Data Structure:**

```
data/
├── source_data/
│   └── GSE211692_processed.txt
├── control/
│   └── control_samples.csv
├── benign/
│   └── benign_prostate_samples.csv
└── cancer/
    ├── prostate_cancer_samples.csv
    ├── stage_1_prostate_cancer_samples.csv
    ├── stage_2_prostate_cancer_samples.csv
    ├── stage_3_prostate_cancer_samples.csv
    └── stage_4_prostate_cancer_samples.csv
```

## Output Files and Results

### Machine Learning Models

- **18 trained classification models** (`.pkl` files)
- **Cross-validation results** for all models
- **Feature importance rankings** for each comparison
- **Performance metrics** (accuracy, ROC-AUC, confusion matrices)

### GSEA Results

- **miEAA pathway enrichment** results for all comparisons
- **GeneCards pathway analysis** with scoring
- **Progressive pathway visualization** data
- **Clustered pathway matrices** (Excel format)

### Network Analysis Output

- **miRNA-gene interaction networks** (GraphML format)
- **1,413 target genes** with interaction data
- **Network hub analysis** results
- **Gephi network files** for visualization

### Publication-Ready Figures

**Main Figures:**

- **Figure 1**: Comprehensive differential expression analysis and volcano plots
- **Figure 2**: Machine learning classification performance and ROC analysis
- **Figure 3**: Pathway progression analysis and heatmaps
- **Figure 4**: Network visualization and hub gene analysis

**Supplementary Materials:**

- **Supplementary Figure 1**: Additional validation and clustering analyses
- **Supplementary Tables**: Complete feature lists and pathway data
- **Performance Heatmaps**: Model comparison across all tasks

### Data Tables

- `supplemental_table_1.xlsx`: Complete differential expression results
- `top_20_miRNA_comparisons.xlsx`: Top features for each comparison
- `predictive_miRNA_genes.csv`: miRNA-gene interaction predictions (1,413 genes)
- `table2.xlsx`: Summary statistics and model performance metrics
- `miRNA Disease Association Search.xlsx`: Clinical disease associations

## Innovation and Contributions

### Methodological Innovations

1. **Dual-Pipeline Approach**: Parallel stage-based and disease-based analysis
2. **Multi-Modal Integration**: GSEA, clustering, classification, and network analysis
3. **Progressive Analysis**: Tracking changes across cancer progression
4. **Cross-Platform Validation**: Multiple enrichment and pathway databases

### Scientific Contributions

1. **Comprehensive miRNA Landscape**: Most extensive miRNA analysis of prostate cancer progression
2. **Novel Biomarker Discovery**: Stage-specific and disease-specific miRNA signatures
3. **Pathway Integration**: First integrated miRNA-pathway-network analysis in prostate cancer
4. **Clinical Translation**: Actionable biomarkers for diagnosis and monitoring

## Data Availability

- **Raw data**: Available from GEO database (GSE211692)
- **Processed data**: All preprocessing steps documented and reproducible
- **Analysis results**: Complete results provided in repository
- **Model files**: All trained models available for validation and application

## Contact Information

For questions regarding the analysis pipeline, data interpretation, or collaboration opportunities:

- eahintz@colgate.edu
- Colgate University Department of Computer Science

## Funding and Acknowledgments

- Gene Expression Omnibus (GEO) for providing the GSE211692 dataset
- Open-source community for machine learning and bioinformatics tools
- Funded by Colgate University

---

**Note**: This repository represents a comprehensive computational framework for multi-modal miRNA expression analysis in cancer progression. The dual-pipeline approach (stage-based and disease-based) with integrated GSEA, clustering, and network analysis provides a robust foundation for biomarker discovery and clinical translation. All analyses are designed for reproducibility and can be adapted for other cancer types and progression studies.
