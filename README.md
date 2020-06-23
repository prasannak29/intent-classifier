Intent Classifier

Inherently linked to Natural Language Processing (NLP), intent classification automatically finds
purpose and goals in text. For example, imagine that you are interested in subscribing to cloud service
pCloud and drop them a line asking about their paid subscription plans:
“Hi, I’m a photographer and work with a significant amount of raw files. What kind of storage do
you offer? Is it a lifetime membership? For the right price, I’d love to purchase cloud storage.”
With an intent classifier, you could easily locate this query among the numerous user interactions
you receive on a daily basis, and automatically categorize it as a clear Purchase intent.
Ultimately, the goal of intent classification is to help you pinpoint the exact motivation behind
pieces of text. Every customer interaction has a purpose, an aim, or intention. Whether it’s purchase
intent, a request for more information, or someone who wants to unsubscribe, you should be able to
respond to sales leads quickly to increase your chances of closing the sale.

For intent classifier we are using Bidirectional GRU.We trained this model with adam
optimizer, batch size 16 and epochs 100.Different activation functions used are ReLu and Softmax.
In the dataset, there are total 6 intents which are general info, admission, placement, bot, facilities,
dept info(department information). It consits of 407 questions along with their intent. When 0.2 test
size was considered using cross-validation test-train split 325 records were trained and 82 records
were tested. Similarly, when 0.15 test size was considered 70 records were tested based on training
on 337 records.

Dependencies required:
1] Tensorflow v=1.13.1
2] Keras v=2.2.4
3] Nltk (any version)
4] Matplotlib
5] Numpy
6] sklearn
