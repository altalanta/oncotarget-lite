# Model Card: Oncotarget-lite Immunotherapy Target Classifier

## Model Details
- **Model Type**: Random Forest / MLP Classifier
- **Version**: 0.2.0
- **Date**: September 2024
- **Authors**: Oncotarget-lite Team
- **License**: MIT

## Intended Use

### Primary Intended Uses
- **Educational demonstration** of ML pipelines for biomarker discovery
- **Research prototype** for understanding tumor target prioritization
- **Teaching tool** for interpretable ML in computational biology

### Primary Intended Users
- ML researchers and practitioners in computational biology
- Students learning biomedical data science
- Software engineers building bioinformatics pipelines

### Out-of-Scope Use Cases
- **Clinical decision making** - This model is NOT validated for patient care
- **Drug development** - Predictions should not guide therapeutic decisions
- **Regulatory submissions** - Not suitable for FDA or other regulatory use

## Data

### Training Data
- **Source**: Synthetic datasets generated from public summary statistics
- **Inspiration**: GTEx (normal tissue), TCGA (tumor), DepMap (essentiality), UniProt (annotations)
- **Size**: ~50 genes with engineered features
- **Generation Process**:
  1. Sampled realistic expression ranges from published statistics
  2. Added biological noise and correlations
  3. Created tumor vs normal contrast features
  4. Incorporated safety flags from dependency scores

### Data Limitations
- **Synthetic nature**: Does not reflect real biological complexity
- **Small scale**: Limited to ~50 genes vs genome-wide analysis
- **Missing modalities**: No genomics, proteomics, or clinical data
- **Population bias**: Not representative of diverse patient populations

## Preprocessing & Features

### Feature Engineering
- **Tumor vs Normal Contrasts**: (TCGA tumor - GTEx normal) expression
- **Safety Scores**: DepMap dependency scores for essentiality assessment
- **Annotation Features**: Molecular weight, transmembrane domains, signal peptides
- **Normalization**: StandardScaler for continuous features

### Feature Count
- Input features: 8-12 (varies by available annotations)
- No feature selection applied

## Model Architecture

### Random Forest (Default)
- **Estimators**: 100 trees
- **Max Depth**: Unlimited
- **Random State**: 42 (reproducible)

### MLP Alternative
- **Architecture**: Input → Dense(32) → ReLU → Dropout(0.15) → Dense(16) → ReLU → Dropout(0.15) → Dense(1)
- **Training**: Adam optimizer, learning rate 1e-3, early stopping (patience=10)
- **Regularization**: Dropout for generalization

## Metrics & Performance

### Primary Metrics
- **AUROC**: Area under ROC curve (discrimination)
- **Average Precision**: Area under precision-recall curve
- **Brier Score**: Calibration quality (lower is better)
- **ECE**: Expected Calibration Error (reliability)

### Typical Performance (Bootstrap 95% CI)
- **Test AUROC**: 0.75-0.85
- **Test AP**: 0.70-0.80
- **Brier Score**: 0.15-0.25
- **ECE**: 0.05-0.15

### Evaluation Methodology
- **Cross-validation**: 80/20 train/test split, stratified
- **Bootstrap CI**: 1000 resamples for confidence intervals
- **Calibration**: 10-bin reliability diagram
- **Overfitting Check**: Train vs test metric gaps

## Fairness & Limitations

### Known Limitations
1. **Dataset Limitations**:
   - Synthetic data may not capture real biological relationships
   - Limited gene coverage compared to full transcriptome
   - No consideration of tumor heterogeneity or subtypes

2. **Model Limitations**:
   - Simple feature engineering without domain expertise
   - No handling of batch effects or technical confounders
   - Binary classification ignores graded target attractiveness

3. **Evaluation Limitations**:
   - Single synthetic dataset limits generalizability assessment
   - No external validation cohorts
   - No assessment across cancer types or patient subgroups

### Potential Failure Modes
- **Distribution Shift**: Real data may have different feature distributions
- **Label Noise**: Synthetic labels may not reflect true therapeutic potential
- **Feature Correlation**: May learn spurious correlations from synthetic generation
- **Overfitting**: Small dataset size increases overfitting risk

### Fairness Considerations
- **Population Representation**: Synthetic data does not model demographic differences
- **Cancer Type Bias**: Limited to three cancer types (BRCA, LUAD, COAD)
- **Socioeconomic Factors**: No consideration of access to care or treatment disparities

## Interpretability

### SHAP Explanations
- **Global Importance**: Feature ranking across all predictions
- **Local Explanations**: Per-sample feature contributions
- **Visualization**: Summary plots, waterfall diagrams, feature importance

### Key Features Typically Identified
- Tumor vs normal expression contrasts
- Dependency scores (safety assessment)
- Molecular properties (druggability proxies)

## Model Governance

### Update Cadence
- **Research Setting**: Retrain when new synthetic data generated
- **Production Setting**: Would require monthly evaluation on new data

### Performance Monitoring
- Monitor AUROC, calibration metrics on holdout sets
- Track feature drift and prediction distributions
- Alert on significant performance degradation

### Adding New Data
1. Validate schema compatibility with existing features
2. Check for distribution shifts in key features
3. Retrain model with expanded dataset
4. Evaluate on fresh holdout set
5. Update model card with new performance metrics

### Responsible Use Guidelines
1. **Never use for clinical decisions** without extensive validation
2. **Validate findings** with orthogonal experimental assays
3. **Involve domain experts** in interpreting predictions
4. **Consider ethical implications** of target selection
5. **Maintain audit trail** of model versions and decisions

## Contact & Support

### Technical Issues
- **Repository**: [GitHub Issues](https://github.com/example/oncotarget-lite/issues)
- **Documentation**: README.md and code comments

### Model Questions
- **Authors**: See repository contributors
- **Method Questions**: Refer to methodology section in README

### Responsible AI Concerns
- **Ethics Review**: Consult institutional review board for human subjects research
- **Bias Reports**: Report potential bias through GitHub issues
- **Safety Concerns**: Contact authors immediately for safety-critical applications

---

*This model card follows the recommendations from Mitchell et al. (2019) and the Partnership on AI's framework for responsible AI development.*