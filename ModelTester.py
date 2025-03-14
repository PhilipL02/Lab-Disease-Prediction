from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

class ModelTester():
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.best_models = {}
        self.classifiers = [
            {
                "name": "RandomForestClassifier",
                "model": RandomForestClassifier(),
                "param_grid": {
                    "n_estimators": [150, 250], 
                    "criterion": ["entropy"], 
                    "max_features": ["log2"]
                }
            },
            {
                "name": "LogisticRegression",
                "model": LogisticRegression(max_iter=1000),
                "param_grid": {
                    'C': [0.01, 0.1, 1, 10, 100]
                }
            },
            {
                "name": "KNeighborsClassifier",
                "model": KNeighborsClassifier(),
                "param_grid": {
                    "n_neighbors": [11, 13], 
                    "weights": ["uniform", "distance"], 
                    "leaf_size": [20, 30]
                }
            }
        ]


    def prepare_data(self, df):
        # Split data, handle feature scaling, and store the results

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(df)

        X_train, X_val, X_test = self.feature_scaling(X_train, X_val, X_test)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        
    def split_dataset(self, df):
        X, y = df.drop('cardio', axis=1), df['cardio']

        # TODO: Vilken test_size skulle vara bra?
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def feature_scaling(self, *datasets):
        # First standardization and then normalization
        datasets_standardized = self.feature_standardization(*datasets)

        return self.feature_normalization(*datasets_standardized)
    

    def feature_standardization(self, *datasets):
        scaler = StandardScaler()
        standardized_datasets = [scaler.fit_transform(datasets[0])]
        standardized_datasets += [scaler.transform(d) for d in datasets[1:]]

        return tuple(standardized_datasets)


    def feature_normalization(self, *datasets):
        scaler = MinMaxScaler()
        normalized_datasets = [scaler.fit_transform(datasets[0])]
        normalized_datasets += [scaler.transform(d) for d in datasets[1:]]

        return tuple(normalized_datasets)


    def train_and_optimize(self):
        for clf in self.classifiers:
            print(f"Tränar {clf['name']}...")

            grid_search = GridSearchCV(
                clf["model"], clf["param_grid"], cv=3, scoring="recall", verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)

            validation_score = grid_search.score(self.X_val, self.y_val)

            self.best_models[clf["name"]] = {
                "best_estimator": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "score": validation_score
            }

        return self.best_models


    def plot_models(self):
        if not self.best_models:
            print("Inga tränade modeller finns ännu. Kör train_and_optimize() först.")
            return

        model_names = list(self.best_models.keys())
        recall_values = [self.best_models[name]["score"] for name in model_names]

        plt.figure(figsize=(10, 5))
        plt.bar(model_names, recall_values, color="skyblue", edgecolor="black")

        plt.xlabel("Modeller")
        plt.ylabel("Score")
        plt.title("Jämförelse av modeller baserat på Recall")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)

        for i, v in enumerate(recall_values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")

        plt.show()


    def useVotingClassifier(self):
        models = []
        for name, model_info in self.best_models.items():
            model = model_info["best_estimator"]
            print("model", model)
            models.append((name, model))
        
        vote_clf = VotingClassifier(estimators=models, voting="hard")
        
        vote_clf.fit(self.X_train, self.y_train)
        
        y_pred = vote_clf.predict(self.X_test)
        
        print(classification_report(self.y_test, y_pred))
        
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot()
