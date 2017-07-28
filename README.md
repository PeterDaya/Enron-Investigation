Fraud Detection
@@@@@@@@@@@@@@@@@

Using financial and email data that was made public from the Enron
Scandal, I wrote a program in Python that utilizes machine learning methods
to identify employees who may have committed fraud.

In Progress: I read the Enron data in and chose relevant features to investigate.  
More recently, I eliminated certain data points as noise.  Based on the assumption that 
persons of interest would have salary and stocked options listed, I eliminated data without any
of that information.  Furthermore, I used a simple Naive Bayes classifier just to see how cleaning up
data points will improve detection.

However, my main goals for the future include improving feature selection (since I selected all features about each person individually in my
analysis) and optimizing the algorithm I used for machine learning, which includes possibly selecting a new algorithm and choosing the best
parameters available. 
