# CSE258-SentimentAnalysis

In the analysis section, we figure out that the positive and negative tweets have different length distribution: negative tweets distribute more in the shorter and longer ranges. At the same time, we have shown the top 10 unigrams, bigrams and trigrams for both positive and negative tweets, with wordcloud pictures. In understanding the relationship between sentiment and posting time, we find out that people tend to post positive tweets during weekends and midnights.

On the sentiment prediction models, we have implement several baselines including Naive Bayes, Random Forest, Support Vector Machine, Decision Tree, CNN and RNN. These models reveal acceptable performance. To furthur increase the accuracy, we have
implemented our own model BERT-SENTI, which is the fine tuned model on the base BERT. BERT-SENTI achieves highest accuracy of 80.84%.

There are several limitations on our BERT-SENTI model: first, although we have proved that the tweet posting time contains some degree of information by analyzing the dataset as well as measuring the baseline models, we didn’t implement the BERT-SENTI with time-related features because the the base model which we fine tune on is with only content-related features.
Secondly, due to machine capacity, we only train our model on a subset of size 160,000. Although the size is satisfying for a common classification model, the performance may increase using the whole dataset.