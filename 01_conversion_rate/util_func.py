import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal

def horizontal_barplot(feature: pd.Series):
    """Plot the horizontal barplot for categorical variable."""

    feature.value_counts().sort_values().plot.barh()
    plt.title('Distribution of Feature "%s"' % feature.name)
    plt.xlabel('Count')
    plt.ylabel(feature.name)
    plt.show()

def histogram(feature: pd.Series):
    """Plot the horizontal barplot for continuous variable."""

    plt.hist(feature)
    plt.title('Distribution of Feature "%s"' % feature.name)
    plt.xlabel(feature.name)
    plt.ylabel('Count')
    plt.show()

def integer_barplot(feature: pd.Series):
    """Plot the barplot for variable with discrete integer label."""

    feature.value_counts().sort_values().sort_index().plot.bar()
    plt.title('Distribution of Feature "%s"' % feature.name)
    plt.xlabel(feature.name)
    plt.ylabel('Count')
    plt.show()

def feature_engineering(dataset: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering for conversion data"""

    # Create dummy variables for categorical variables with base group shown in prefix
    country_dummies = pd.get_dummies(dataset['country'], drop_first=True, 
                                    prefix=sorted(dataset['country'].unique())[0])
    source_dummies = pd.get_dummies(dataset['source'], drop_first=True,
                                    prefix=sorted(dataset['source'].unique())[0])
    dataset = dataset.drop(['country', 'source'], axis=1)
    dataset = pd.concat([dataset, country_dummies, source_dummies], axis=1)
    return dataset

def feature_comparison_plot(features: list, values: np.ndarray, type: Literal['coefficient', 'importance']):
    """Plot the horizontal barplot for comparing coefficients or importance between variables."""

    plt.barh(features, values)
    plt.xlabel('Value')
    plt.ylabel('Feature')
    plt.title('Comparison of %s' % type.capitalize())
    plt.show()