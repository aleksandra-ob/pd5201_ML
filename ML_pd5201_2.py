# Importowanie bibliotek
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tworzymy przykładowy zbiór danych
data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
    'KRAS':   [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]  
}

df = pd.DataFrame(data)

# Cechy (X) i etykiety (y)
X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42,  stratify=y)

#Stworzenie modelu
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Dokładność
print("Accuracy:", accuracy_score(y_test, y_pred))

# Raport klasyfikacji
print(classification_report(y_test, y_pred))

# Macierz pomyłek
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# do stworzenia modelu klasyfikacyjnego zastosowałam regresję logistyczną.
# Model osiągnął accuracy = 1.0, co oznacza, że wszystkie próbki testowe
# zostały sklasyfikowane poprawnie.
#
# Precision = 1.0 oznacza brak wyników fałszywie pozytywnych (false positives).
#
# Recall = 1.0 oznacza brak wyników fałszywie negatywnych (false negatives),
# czyli model wykrył wszystkie rzeczywiste przypadki pozytywne.
#
# F1-score = 1.0 wskazuje na idealną równowagę pomiędzy precision i recall.
#
# Macierz pomyłek pokazała, że spośród 4 próbek testowych
# 2 zostały sklasyfikowane jako prawdziwie pozytywne,
# a 2 jako prawdziwie negatywne.
# Model nie popełnił żadnego błędu klasyfikacji.
