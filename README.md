# Titanic-Machine-Learning-kaggle-Competition

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The full details of the competition can be found at  https://www.kaggle.com/c/titanic.

### The Challenge

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, I am asked to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

### What Data Will I Use in This Competition?

In this competition, I've been given access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s my job to predict these outcomes.

Using the patterns I find in the train.csv data, I must predict whether the other 418 passengers on board (found in test.csv) survived.

