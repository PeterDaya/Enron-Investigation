#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
#Importing difference classifiers to test best method
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest

#calc_fraction helps in defining a new programmer-defined feature
def calc_fraction(all_msgs, poi_msgs):
    if all_msgs == "NaN" or poi_msgs == "NaN":
        return 0
    fraction_msgs = poi_msgs/all_msgs
    return fraction_msgs

#plot_data is a quick and easy method for visualizing our data
def plot_data(data_dict, plot_x, plot_y):
    data = featureFormat(data_dict, [plot_x, plot_y, 'poi'])

    for p in data:
        x = p[0]
        y = p[1]
        poi = p[2]
        if poi:
            color = 'blue'
        else:
            color = 'red'

        plt.scatter(x, y, color=color)

    plt.xlabel(plot_x)
    plt.ylabel(plot_y)
    plt.show()

#test_classifier helps us in testing a variety of classifiers
def test_classifier(clf, features, labels):
    accuracy = []
    precision = []
    recall = []
    note = True

    for i in range(1000):
        features_train, features_test, labels_train, labels_test =\
        train_test_split(features, labels, test_size=0.3)

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        precision.append(precision_score(labels_test, pred))
        accuracy.append(accuracy_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))

        if i % 25 == 0:
            if note:
                sys.stdout.write("\nTesting Classifier")
            sys.stdout.write(".\n")
            sys.stdout.flush
            note = False

    print "Finished"
    print "\n"
    print "Precision: ", mean(precision)
    print "Recall: ", mean(recall)

    return mean(precision), mean(recall)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "salary", "deferral_payments", "total_payments", "loan_advances", "bonus", 
"restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses", "exercised_stock_options",
"long_term_incentive", "restricted_stock", "director_fees",
"to_messages", "from_poi_to_this_person", "from_messages", 
"from_this_person_to_poi", "shared_receipt_with_poi"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print "Number of data points before removing outliers: ", len(data_dict)
outliers = ["TOTAL", "LOCKHART EUGENE E", "THE TRAVEL AGENCY IN THE PARK"]
for i in range(len(outliers)):
    data_dict.pop(outliers[i])

cp = dict(data_dict)
for person in cp:
    if (cp[person]["salary"] == "NaN") or (cp[person]["bonus"] == "NaN"):
        data_dict.pop(person)

print "Number of data points after removing outliers: ", len(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

plot_data(data_dict, "salary", "bonus")
plot_data(data_dict, "salary",  "from_poi_to_this_person")

for i in my_dataset:
    record = my_dataset[i]
    to_messages = record["to_messages"]
    from_poi_to_this_person = record["from_poi_to_this_person"]
    from_this_person_to_poi = record["from_this_person_to_poi"]
    from_poi_fraction = calc_fraction(to_messages, from_poi_to_this_person)
    to_poi_fraction = calc_fraction(to_messages, from_this_person_to_poi)
    record["from_poi_fraction"] = from_poi_fraction
    record["to_poi_fraction"] = to_poi_fraction

new_features_list = features_list + ["from_poi_fraction", "to_poi_fraction"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
kbest = SelectKBest(k=5)
kbest.fit(features, labels)
scores = kbest.scores_
unsorted = zip(features_list[1:], scores)
sorted_info = list(reversed(sorted(unsorted, key=lambda x: x[1])))
selected_features = dict(sorted_info[:10])

features_list = ["poi"] + selected_features.keys()
data = featureFormat(my_dataset, features_list)

labels, features = targetFeatureSplit(data)

scale = preprocessing.MinMaxScaler()
features = scale.fit_transform(features)

### Task 4: Try a varity of classifiers
clf = GaussianNB()
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = \
 #   train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Accuracy Score: ", accuracy_score(pred, labels_test)
dump_classifier_and_data(clf, my_dataset, features_list)
