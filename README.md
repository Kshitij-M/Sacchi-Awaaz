# Sacchi-Awaaz
## A web app that helps you to manage Mental Health Problems

This web app helps users in managing Mental Health problems and get rid of the habit of speaking Abusive or Offensive language. It not only prevents users from using these words, but also tries to cure the main reason that forces people to use them. People generally tend to use offesive or harsh words when they are not mentally strong or because it is their bad habit. The app works in 3 steps:

![Alt text](screenshots/1.png?raw=true "Prevent")
1. **Automatically detects use of offensive language and automatically deletes it:** In this step, I have used pretrained GloVe embeddings available on Stanford University's website. Now to label the vectors, I decided to divide the words into 2 classes - hateful and non-hateful words. I created a corpus of both of these categories. I iterated over the words typed by the user and checked its similaroty with the 2 categories. I then created a thresold value which finally classifies the words into their respective categories by comparing the word vector with the words in the categories. The offensive words are labelled seperately it's offensiveness is labelled. In the resulting message all the oensive words are automatically deleted so that user is not able to use them.

![Alt text](screenshots/3.png?raw=true "Detect")
2. **Checking if they are mentally ill:** In this step, I used another publically available dataset which contained life descriptions of poeple and it had been classified into 2 parts - mentally ill and mentally fit. I trained basic Logistic Regression model on this dataset which achieved 99.54% accuracy. Here the user have to describe their recent lives and the model will classiy if they are facing mental illness or not.

![Alt text](screenshots/5.png?raw=true "Help")
3. **Providing help in case of mental illness:** Generally people are quite careless about mentall illness and avoid speaking about them or asking for help. They are unaware of the symptoms, its cure or its effect on our body. In this step I used a publically available dataset containing frequently asked questions and answers provided by healthcare experts. Now using this dataset, I created a chatbot that answers all the question the user has related to mental illness. It also provides contact information of counseller in case it is required. I did this by finding the cosine similarity between the quesion asked by the user and those present in the dataset and then provide the suitable answer to the query. Users can also hear the answers in audio format.

Notes:
- For easier frontend development, I used Streamlit library.
- I used glove.6b.300d.txt embeddings. It's size is roughly 1gb so I cannot upload it on GitHub. for smaller embeddings thresold value needs to be tuned.
- Some of the models are compressed and need to be extracted before using.
