import pandas as pd

df = pd.read_csv("dane_projekt1.csv")
print(df.describe())

from sklearn.model_selection import train_test_split
# Przypisujemy cechy X i etykiety y
X = df.drop(columns=["Gene_Function"])
y = df["Gene_Function"]

# Dzielimy dane na 80% treningowe + walidacyjne, 20% testowe
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Dzielimy dane treningowe na 75% trening + 25% walidacja (czyli 60%/20% globalnie)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
