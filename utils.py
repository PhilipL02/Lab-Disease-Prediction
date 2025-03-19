import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

bmi_categories = {
    1: {'label': 'Normal range', 'range': (18.5, 25.0)},
    2: {'label': 'Overweight', 'range': (25.0, 30.0)},
    3: {'label': 'Obese (class I)', 'range': (30.0, 35.0)},
    4: {'label': 'Obese (class II)', 'range': (35.0, 40.0)},
    5: {'label': 'Obese (class III)', 'range': (40.0, np.inf)}
}

ap_labels = {
    1: 'Healthy',
    2: 'Elevated',
    3: 'Stage 1 hypertension',
    4: 'Stage 2 hypertension',
    5: 'Hypertension crisis',
}


def get_blood_pressure_category(ap_hi, ap_lo):
    if ap_hi > 180 or ap_lo > 120:
        return 5  # Hypertension crisis
    elif ap_hi >= 140 or ap_lo >= 90:
        return 4  # Stage 2 hypertension
    elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
        return 3  # Stage 1 hypertension
    elif (120 <= ap_hi < 130) and ap_lo < 80:
        return 2  # Elevated
    else:
        return 1  # Healthy


def get_bmi_category(bmi):
    for category, info in bmi_categories.items():
        lower, upper = info['range']
        if lower <= bmi < upper:
            return category
    return None


def plot_cardio_risk_by_category(df):
    # Group dataframes by the categories and calculate the proportion for each category
    bmi_group = df.groupby('bmi_category')['cardio'].mean() * 100
    ap_group = df.groupby('ap_category')['cardio'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bmi_group.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Andel hjärt-kärlsjukdom per BMI-kategori')
    axes[0].set_ylabel('Procent (%)')
    axes[0].set_xticklabels([bmi_categories[i]['label'] for i in bmi_group.index], rotation=30)
    axes[0].set_xlabel(None)

    ap_group.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Andel hjärt-kärlsjukdom per blodtryckskategori')
    axes[1].set_ylabel('Procent (%)')
    axes[1].set_xticklabels([ap_labels[i] for i in ap_group.index], rotation=30)
    axes[1].set_xlabel(None)

    plt.tight_layout()
    plt.show()


def plot_cardio_distribution(df):
    counts = df['cardio'].value_counts()
    total = counts.sum()

    fig, ax = plt.figure(dpi=100), plt.axes()

    ax.bar(['Negativ', 'Positiv'], counts)

    for i, count in enumerate(counts):
        percentage = f'{(count / total) * 100:.2f}%'
        ax.text(i, count / 2, f'{count} st', ha='center', fontsize=12, color='white', fontweight='bold', verticalalignment='bottom')
        ax.text(i, count / 2, f'({percentage})', ha='center', fontsize=10, color='white', fontweight='bold', verticalalignment='top')

    ax.set_xlabel('Diagnos')
    ax.set_ylabel('Antal patienter')
    ax.set_title('Fördelning av hjärt-kärlsjukdomsfall')

    plt.show()


def plot_cholesterol_distribution(df):
    cholesterol_counts = df['cholesterol'].value_counts()

    cholesterol_value_labels = {
        1: 'Normala',
        2: 'Över normala',
        3: 'Långt över normala',
    }

    labels = [cholesterol_value_labels[value] for value in cholesterol_counts.keys()]

    fig, ax = plt.figure(dpi=100), plt.axes()

    ax.pie(cholesterol_counts.to_list(), labels=labels, autopct='%1.1f%%')

    ax.set_title('Fördelning av kolesterolvärden')

    plt.show()


def plot_smoking_pie(df):
    counts = df['smoke'].value_counts()

    fig, ax = plt.figure(dpi=100), plt.axes()

    ax.pie(counts.to_list(), labels=['Icke-rökare', 'Rökare'], autopct='%1.1f%%')

    ax.set_title('Andel rökare och icke-rökare')

    plt.show()


def plot_gender_cardio_distribution(df):
    # A copy of the dataframe is used to change the values in gender to correctly show labels in plot
    df_copy = df.copy()
    df_copy['gender'] = df_copy['gender'].replace({ 1: 'Kvinna', 2: 'Man' })

    ax = sns.countplot(df_copy, x='gender', hue='cardio')
    ax.set_xlabel('Kön')
    ax.set_ylabel('Antal patienter')
    ax.set_title('Fördelning av hjärt-kärlsjukdomsfall per kön')
    
    plt.show()


def show_histplot_for_features(df, columns, bins_list=None, default_bins=20):
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 5))
    fig.suptitle(f'Histplots för {', '.join(columns)}')

    for i, column in enumerate(columns):
        # With default_bins all the histplots will have the same amount of bins
        bins = default_bins

        # If bins_list is set as a list and length is correct, it will be used to collect bins for each separate histplot
        if type(bins_list) is list and len(bins_list) == len(columns):
            bins = bins_list[i] 

        sns.histplot(data=df, x=column, bins=bins, ax=axes[i])

    plt.show()


def show_boxplot_for_features(df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 5))
    fig.suptitle(f'Boxplots för {', '.join(columns)}')

    for i, column in enumerate(columns):
        sns.boxplot(y=df[column], ax=axes[i])

    plt.show()


def calculate_bmi(weight_in_kg, height_in_meters):
    return weight_in_kg / np.square(height_in_meters)


def print_suggestions_for_lower_upper_limit(df, column):
    # Z-score method:
    lower_limit_zscore = df[column].mean() - 3*df[column].std()
    upper_limit_zscore = df[column].mean() + 3*df[column].std()

    # Interquartile range method:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3-Q1

    lower_limit_iqr = Q1 - 1.5*IQR
    upper_limit_iqr = Q3 + 1.5*IQR
    
    # Percentile method:
    lower_limit_percentile = df[column].quantile(0.01)
    upper_limit_percentile = df[column].quantile(0.99)

    df_limits = pd.DataFrame({
        'Lägsta gräns': [lower_limit_zscore, lower_limit_iqr, lower_limit_percentile],
        'Högsta gräns': [upper_limit_zscore, upper_limit_iqr, upper_limit_percentile]
    }, index=['Z-score', 'Interquartile range', 'Percentile'])

    print(f'Förslag på gränser för kolumn: {column}', end='\n\n')
    print(df_limits)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot()
