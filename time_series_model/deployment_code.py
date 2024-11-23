import tensorflow as tf

def get_data(data):
    data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    
data = get_data("data.csv")

model_kalori = import_model("/model/model1.keras")
model_garam = import_model("/model/model2.keras")

def predict(data, steps=1):
    return model.predict(data)

def predict_gula_lemak(data, steps=1):
    return model_garam.predict(data)