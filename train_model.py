import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def parse_fare(val):
    s = str(val).strip()
    if s.count('.') > 1:
        parts = s.split('.')
        return float(''.join(parts[:-1]) + '.' + parts[-1])
    return float(s)

df = pd.read_csv('data/titanic.csv', index_col='PassengerId')
df.drop(['Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Fare'] = df['Fare'].apply(parse_fare)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline_categoric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
pipeline_numeric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MinMaxScaler())
])
pipeline_fare = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('pipeline_categoric', pipeline_categoric, ['Pclass', 'Sex', 'Embarked']),
        ('pipeline_numeric', pipeline_numeric, ['SibSp', 'Parch']),
        ('pipeline_fare', pipeline_fare, ['Fare'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

parameter = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter,
    n_iter=10,
    cv=kf,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    return_train_score=True
)

model.fit(X_train, y_train)
joblib.dump(model, 'model.joblib')
print('model.joblib saved!')