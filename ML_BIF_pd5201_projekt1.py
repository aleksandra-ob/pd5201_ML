import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#1.	Pobranie danych
df = pd.read_csv("dane_projekt1.csv")

#2.	Eksploracja danych
print(df.head())
print(df.describe())
print(df['Gene_Function'].value_counts())

#3.	Przetwarzanie danych
X = df.drop(columns=['Gene_ID', 'Gene_Function'])
y = df['Gene_Function']

#4.	Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#5.	Budowa modelu ML
# Trenujemy drzewo
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predykcja
y_pred_tree = tree_model.predict(X_test)

# Ocena
print("Drzewo decyzyjne - Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))


plt.figure(figsize=(20,12))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=tree_model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Drzewo decyzyjne")
plt.show()



#6.	Ewaluacja modelu i interpretacja wyników

cm = confusion_matrix(y_test, y_pred_tree)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=tree_model.classes_
)

disp.plot(cmap="Blues")
plt.title("Macierz pomyłek")
plt.show()


# Interpretacja wyników:
# Model drzewa decyzyjnego został wykorzystany do klasyfikacji genów
# do czterech kategorii funkcjonalnych na podstawie poziomu ekspresji
# w 10 tkankach. Osiągnięta dokładność wyniosła 15%, co oznacza że model
# poprawnie sklasyfikował 2 z 13 genów oraz wskazuje na niską skuteczność klasyfikacji.
#
# Analiza raportu klasyfikacji oraz macierzy pomyłek wykazała, że model
# miał trudności z poprawnym rozróżnianiem wszystkich klas. Szczególnie
# słabo rozpoznawana była najmniej liczna klasa "enzyme", dla której recall wyniósł 0.
#
# Możliwą przyczyną niskiej skuteczności jest niewielka liczba dostępnych
# obserwacji (50 genów) oraz fakt, że dane są syntetyczne i mogą nie
# zawierać wyraźnych zależności pomiędzy profilem ekspresji a funkcją genu.
#
# Drzewo decyzyjne utworzyło wiele szczegółowych reguł opartych na niewielkiej 
# liczbie obserwacji. Może to wskazywać na przeuczenie modelu, co mogło przyczynić się
# do niskiej skuteczności klasyfikacji na zbiorze testowym.
#
#  Zaletą drzewa decyzyjnego jest łatwa interpretacja wyników oraz możliwość
# wizualizacji procesu podejmowania decyzji. W celu poprawy skuteczności
# można byłoby zwiększyć liczbę danych treningowych lub dobrać inne
# parametry modelu, np. ograniczenie głębokości drzewa.
#
# Drzewo decyzyjne jest jednak odpowiednim wyborem do tego projektu,
# ponieważ pozwala w prosty sposób analizować proces klasyfikacji i
# interpretować uzyskane wyniki.
