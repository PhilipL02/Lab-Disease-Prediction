from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import numpy as np

class ModelTester():
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.trained_models = {}
        self.models = [
            {
                'name': 'RandomForestClassifier',
                'model': RandomForestClassifier(),
                'param_grid': {
                    'n_estimators': [50, 100, 150], 
                    'criterion': ['gini', 'entropy'], 
                    'max_features': ['sqrt', 'log2']
                }
            },
            {
                'name': 'LogisticRegression',
                'model': LogisticRegression(max_iter=10000),
                'param_grid': {
                    'C': [0.01, 0.1, 0.5, 1, 3, 10, 100],
                    'solver': ['liblinear', 'saga'],
                }
            },
            {
                'name': 'KNeighborsClassifier',
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'leaf_size': [20, 30]
                }
            }
        ]


    def prepare_data(self, X, y):
        # Split data, handle feature scaling, and store the results

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_arrays(X, y)

        X_train, X_val, X_test = self.feature_scaling(X_train, X_val, X_test)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        
    def split_data_arrays(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def combine_train_val_data(self):
        if not self.X_val is None:
            self.X_train = np.concatenate([self.X_train, self.X_val], axis=0)
            self.X_val = None

        if not self.y_val is None:
            self.y_train = np.concatenate([self.y_train, self.y_val], axis=0)
            self.y_val = None
    

    def feature_scaling(self, *datasets):
        # First standardization and then normalization
        datasets_standardized = self.feature_standardization(*datasets)

        return self.feature_normalization(*datasets_standardized)
    

    def feature_standardization(self, *datasets):
        scaler = StandardScaler()

        # The scaler will fit on the first dataset, and transform all of them
        standardized_datasets = [scaler.fit_transform(datasets[0])]
        standardized_datasets += [scaler.transform(d) for d in datasets[1:]]

        return tuple(standardized_datasets)


    def feature_normalization(self, *datasets):
        scaler = MinMaxScaler()
        
        # The scaler will fit on the first dataset, and transform all of them
        normalized_datasets = [scaler.fit_transform(datasets[0])]
        normalized_datasets += [scaler.transform(d) for d in datasets[1:]]

        return tuple(normalized_datasets)


    def train_and_optimize(self):
        for model in self.models:
            print(f'Tränar {model['name']}...')

            grid_search = GridSearchCV(
                model['model'], model['param_grid'], cv=3, scoring='recall'
            )
            grid_search.fit(self.X_train, self.y_train)

            validation_score = grid_search.score(self.X_val, self.y_val)

            print(f'Bästa hyperparametrar för {model['name']}:', grid_search.best_params_)

            self.trained_models[model['name']] = {
                'best_estimator': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'score': validation_score
            }

        return self.trained_models


    def use_voting_classifier(self):
        estimators = []
        for name, model_info in self.trained_models.items():
            model = model_info['best_estimator']
            estimators.append((name, model))
        
        vote_clf = VotingClassifier(estimators, voting='hard', n_jobs=-1)
        
        vote_clf.fit(self.X_train, self.y_train)

        validation_score = vote_clf.score(self.X_val, self.y_val)

        self.trained_models['VotingClassifier'] = {
            'best_estimator': vote_clf,
            'best_params': None,
            'score': validation_score
        }


    def plot_model_comparison(self, *instances, dataset_labels=None):
        all_instances = (self,) + instances
        num_instances = len(all_instances)

        if num_instances > 4:
            raise ValueError('Denna metod kan hantera max 4 instanser')

        model_names = list(self.trained_models.keys())
        width = max(0.8 / num_instances, 0.1)
        colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple']
        
        positions = np.arange(len(model_names))
        
        fig, ax = plt.figure(figsize=(14, 6), dpi=100), plt.axes()
        
        for i, instance in enumerate(all_instances):
            if not isinstance(instance, ModelTester):
                raise ValueError('Alla argument måste vara instanser av ModelTester')
            
            score_values = [instance.trained_models[name]['score'] for name in model_names]
            label = f'Set {i+1}'
            if dataset_labels is not None:
                label = dataset_labels[i]

            ax.bar(positions + i * width, score_values, width=width, label=label, color=colors[i % len(colors)], edgecolor='black', align='center')
            
            for j, v in enumerate(score_values):
                ax.text(positions[j] + i * width, v + 0.02, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')

        ax.set_xlabel('Modeller')
        ax.set_ylabel('Score')
        ax.set_title('Jämförelse av modeller')
        ax.set_ylim(0, 1)
        ax.set_xticks(positions + (num_instances - 1) * width / 2, model_names, rotation=30)
        ax.legend()
        
        plt.show()
