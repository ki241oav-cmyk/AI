import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def visualize_classifier(classifier, X, y, title=''):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Крок сітки
    mesh_step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                                     np.arange(y_min, y_max, mesh_step_size))
    
    output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    output = output.reshape(x_values.shape)
    
    
    plt.figure()
    plt.pcolormesh(x_values, y_values, output, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    plt.title(title)
    plt.xlabel('Ознака 1')
    plt.ylabel('Ознака 2')
    plt.show()

classifier_type = 'rf' 

input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', marker='s', label='Class-0')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', marker='o', label='Class-1')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', marker='^', label='Class-2')
plt.title('Вхідні дані')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if classifier_type == 'rf':
    classifier = RandomForestClassifier(**params)
    title_suffix = '(Random Forest)'
else:
    classifier = ExtraTreesClassifier(**params)
    title_suffix = '(Extra Trees)'

classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Навчальний набір ' + title_suffix)

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Тестовий набір ' + title_suffix)

class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
print(f"Classifier performance ({classifier_type}) on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

print("Confidence measure:")
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print(f'Datapoint: {datapoint} | Predicted: {predicted_class} | Confidences: {probabilities}') 

visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 'Тестові точки даних ' + title_suffix)