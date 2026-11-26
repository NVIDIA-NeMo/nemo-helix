// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Copy from the library's HTML report (multi_modal_report.j2 and multi_modal_tooltips.py).

export const REPORT_OVERVIEW_INTRO =
  'The evaluation report provides scores from 0.0 to 10.0 to measure the quality and privacy of the synthetic data compared to the training data. The scores are meant to provide a baseline understanding of "Did it work?" You are encouraged to perform your own evaluation on your downstream use case.';

export const METRIC_GROUPS = {
  tabular: {
    title: 'Numerical and Categorical Metrics',
    description:
      'Measures preservation of statistical relationships across all numerical and categorical columns.',
  },
  text: {
    title: 'Text Metrics',
    description: 'Measures pattern and meaning preservation across text columns.',
  },
  timeSeries: {
    title: 'Time-Series Metrics',
    description: 'Measures preservation of relationships between observations across time.',
  },
  protection: {
    title: 'Protection Metrics',
    description: 'Measures resistance to inference attacks on the synthetic data.',
  },
} as const;

export interface MetricCopy {
  readonly name: string;
  readonly description: string;
  readonly tooltip: string;
}

export const METRIC_COPY = {
  column_correlation_stability: {
    name: 'Column Correlation Stability',
    description: 'Measures how well correlations between columns are preserved',
    tooltip:
      'To measure Column Correlation Stability, the correlation between every pair of numeric and categorical columns is computed first in the training data, and then in the synthetic data. The absolute difference between these values is then computed and averaged across all columns. The lower this average value is, the higher the Column Correlation Stability quality score will be. To aid in the comparison of correlations, a heatmap is shown for both the training data and the synthetic data, as well as a heatmap for the computed difference of correlation values. If the intended purpose of the synthetic data is to perform statistical analysis or machine learning, maintaining the integrity of correlations can be critical.',
  },
  deep_structure_stability: {
    name: 'Deep Structure Stability',
    description: 'Measures how well structural relationships are maintained',
    tooltip:
      'Deep Structure Stability is a measure of how well the synthetic data matches the training data in terms of the structure of the data. A higher similarity indicates a better match. The deep structure stability is calculated using the cosine similarity between the embeddings of the numeric and categorical columns in the training and synthetic data.',
  },
  column_distribution_stability: {
    name: 'Column Distribution Stability',
    description: 'Measures how well value distributions are preserved',
    tooltip:
      'Column Distribution Stability is a measure of how closely the column distributions in the synthetic data mirror those in the training data. For each numeric and categorical column, we use a common approach for comparing two distributions referred to as the Jensen-Shannon (JS) Distance. The lower the JS Distance score is on average across all numeric and categorical columns, the higher the column Distribution Stability quality score will be. Note, highly unique strings (neither numeric nor categorical) will not have a distributional distance score. To aid in the comparison of training versus synthetic column, a bar chart or histogram is shown for each numeric and categorical column. Numeric bin edges are rounded, and categories with fewer than 5 training records are grouped into "Other".',
  },
  text_semantic_similarity: {
    name: 'Text Semantic Similarity',
    description: 'Measures meaning preservation in text fields',
    tooltip:
      'Text Semantic Similarity measures how well free text columns in the synthetic data capture the meaning of the original text, while penalizing memorization. A higher score is better. The text semantic similarity is based on cosine similarities of sentence embeddings, and is higher with higher similarity between the synthetic and test data, penalized for memorization by the similarity between the synthetic and training data being too high.',
  },
  text_structure_similarity: {
    name: 'Text Structure Similarity',
    description: 'Measures structural pattern preservation in text fields',
    tooltip:
      'Text Structure Similarity is a measure of how well the synthetic data matches the training data in terms of the structure of the text for any free text columns that are present, including character/word/sentence counts. A higher similarity indicates a better match.',
  },
  autocorrelation_similarity: {
    name: 'Autocorrelation Similarity',
    description: 'Measures whether numeric columns change over time like the training data',
    tooltip:
      'Autocorrelation Similarity checks whether synthetic sequences move over time like the training sequences: whether values carry over from one step to the next, and whether patterns repeat on the same schedule. Each group and numeric column is scored separately and the scores are averaged. Differences small enough to be explained by chance do not lower the score, so 10 means the timing patterns match and scores near 0 mean the synthetic data has lost them. A low score does not mean the synthetic data is bad overall, only that it does not reproduce these timing patterns.',
  },
  membership_inference_protection: {
    name: 'Membership Inference Protection',
    description: 'Protection against attacks that determine whether a record was in training data',
    tooltip:
      "Membership Inference Protection is a measure of how well-protected your data is from membership inference attacks. A membership inference attack is a type of privacy attack on machine learning models where an adversary aims to determine whether a particular data sample was part of the model's synthetic dataset. By exploiting the differences in the model's responses to data points from its training set versus those it has never seen before, an attacker can attempt to infer membership. This type of attack can have critical privacy implications, as it can reveal whether specific individuals' data was used to train the model. Based on directly analyzing the synthetic data, a high score indicates that your synthetic data is well-protected from this type of attack.",
  },
  window_membership_inference_protection: {
    name: 'Window Membership Inference Protection',
    description:
      'Protection against attacks that determine whether a sequence was in training data, from short windows of consecutive values',
    tooltip:
      'Window Membership Inference Protection checks whether the synthetic data sits closer to short stretches of the training sequences than to stretches of holdout sequences it never saw, the trace a generator leaves when it memorizes training data. Each window of consecutive values, across the numeric columns, is compared with its nearest synthetic window. Differences small enough to be explained by chance do not lower the score, so 10 means training windows are no easier to find in the synthetic data than unseen ones, and scores near 0 mean the synthetic data copies training windows. It needs a group column and at least two holdout groups. It is reported on its own and is not part of the Data Privacy Score. Any exact copy deserves a look, whatever the score.',
  },
  attribute_inference_protection: {
    name: 'Attribute Inference Protection',
    description: 'Protection against attacks that infer sensitive attributes',
    tooltip:
      "Attribute Inference Protection is a measure of how well-protected your data is from attribute inference attacks. An attribute inference attack is a type of privacy attack on machine learning models where an adversary seeks to infer missing attributes or sensitive information about individuals from their data that was used to train the model. By leveraging the model's output, the attacker can attempt to predict unknown attributes of a data sample. For a specific attribute, a high score indicates that even when other attributes are known, that specific attribute is difficult to predict. This type of attack poses significant privacy risks, as it can uncover sensitive details about individuals that were not intended to be revealed by the data owners. Based on directly analyzing the synthetic data, a high score indicates that your synthetic data is well-protected from this type of attack.",
  },
} as const satisfies Record<string, MetricCopy>;

export const CHART_TOOLTIPS = {
  autocorrelationTypical:
    'How strongly each value relates to the values before it, at each lag (the number of steps apart). Lines show the median across groups and bands cover the middle half of groups. In good synthetic data the orange and blue lines overlap. Use the dropdown to switch columns.',
  autocorrelationDifference:
    "The average gap between training and synthetic autocorrelation at each lag, across all group-column pairs. Short, pale bars are good. Tall, red bars mark lags where the synthetic data's timing pattern differs from the training data.",
  autocorrelationPairScores:
    "One dot per group and numeric column, scored from 0 to 10. White dots near 10 match the training data's timing pattern, and redder dots further left do not. The dashed line is the overall score. Hover over a dot to see its column and group.",
  windowDistance:
    'How close each training and holdout window comes to its nearest synthetic window, as the typical difference per value in training standard deviations. Windows left of the dashed line are flagged. In well-protected synthetic data the blue and gray bars overlap. Blue bars piled up near 0 mean training windows were copied.',
  windowGroups:
    "The share of each training group's windows that are flagged, for the most-flagged groups. Chance alone flags about as many as the dashed holdout line, so white bars near it are expected and redder bars mean a group's sequences are closer to the synthetic data than unseen sequences are. Hover over a bar to see its group, window count, and exact copies.",
  windowClosest:
    "The synthetic window closest to any training window. When it is an exact copy, its training window is drawn too and the lines lie on top of each other. Otherwise the training window is hidden, so the report shows no training values that the synthetic data doesn't contain. Use the dropdown to switch columns.",
} as const;

export const PII_COPY = {
  title: 'PII Replacement and Replay',
  descriptionReplaced:
    'PII found in the training data, how it was replaced, and how many original values reappear in the synthetic data',
  descriptionOff:
    'PII replacement was off. PII found in the training data and how many original values reappear in the synthetic data',
  contextOff:
    'PII replacement was off, so the model was trained on the original values. PII was detected in the training data only for this check.',
  noPii: 'No PII was found in the training data.',
  footnote:
    'Matches are against the original training values, before replacement. Synthetic Values counts synthetic rows containing at least one of them. PII Replay counts distinct training values that reappear in the synthetic data; lower is better. In free text, a value counts when it appears anywhere in a synthetic text.',
} as const;

export const TRAINING_COLUMNS_DESCRIPTION =
  'Column statistics and data types from the training dataset.';
