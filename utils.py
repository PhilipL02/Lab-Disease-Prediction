import matplotlib.pyplot as plt


def plot_cardio_distribution(df):
    counts = df['cardio'].value_counts()
    total = counts.sum()

    plt.bar(['Negativ', 'Positiv'], counts, color=['darkblue', 'darkred'])

    for i, count in enumerate(counts):
        percentage = f'{(count / total) * 100:.2f}%'
        plt.text(i, count / 2, f"{count} st", ha='center', fontsize=12, color='white', fontweight='bold', verticalalignment='bottom')
        plt.text(i, count / 2, f"({percentage})", ha='center', fontsize=10, color='white', fontweight='bold', verticalalignment='top')

    plt.xlabel('Diagnos')
    plt.ylabel('Antal patienter')
    plt.title('Fördelning av hjärt-kärlsjukdomsfall')

    plt.show()


def plot_cholesterol_distribution(df):
    cholesterol_counts = df['cholesterol'].value_counts()

    cholesterol_value_labels = {
        1: 'Normala',
        2: 'Över normala',
        3: 'Långt över normala',
    }

    labels = [cholesterol_value_labels[value] for value in cholesterol_counts.keys()]

    fig, ax = plt.subplots()
    ax.pie(cholesterol_counts.to_list(), labels=labels, autopct='%1.1f%%')

    plt.title('Fördelning av kolesterolvärden')
    plt.tight_layout()

    plt.show()
