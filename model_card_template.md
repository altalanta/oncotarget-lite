# Model Card: OncoTarget Lite

This document provides a summary of the OncoTarget Lite prediction model. It is automatically generated and updated with every new model release to ensure that all information is current and accurate.

## Model Details

| Detail              | Description                                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| **Model Version**   | `{{ model_version }}`                                                                                    |
| **Model Type**      | `{{ model_type }}`                                                                                       |
| **Release Date**    | `{{ release_date }}`                                                                                     |
| **Code Version**    | `{{ git_commit }}`                                                                                       |
| **Data Version**    | `{{ dvc_data_hash }}`                                                                                    |

## Intended Use

This model is intended to be used by researchers to triage and prioritize potential oncology targets. It provides a predictive score based on a variety of biological and network-based features. It is designed for research purposes only and is **not intended for clinical or diagnostic use**.

## Performance Metrics

The following metrics were calculated on the hold-out test set.

| Metric     | Value         |
| ---------- | ------------- |
| **AUROC**  | `{{ auroc }}` |
| **AP**     | `{{ ap }}`    |
| **Accuracy** | `{{ accuracy }}`|
| **F1 Score** | `{{ f1 }}`    |
| **Brier**  | `{{ brier }}` |
| **ECE**    | `{{ ece }}`   |

## Ethical Considerations

- **Bias**: The model is trained on publicly available biological data, which may contain inherent biases. Performance may vary for under-represented gene families or biological pathways.
- **Accountability**: The model's predictions are probabilistic and should be interpreted as a guide for further research, not as a definitive statement of a target's efficacy. The development team is responsible for monitoring model performance and addressing any identified issues.

## Training Data

The model was trained on a dataset of known oncology targets and non-targets, processed from a variety of public sources. The exact dataset used for this model version is tracked by DVC with the hash listed in the Model Details section.

## Training Parameters

| Parameter      | Value                 |
| -------------- | --------------------- |
{{ training_parameters }}

















