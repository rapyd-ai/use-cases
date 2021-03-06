# -*- coding: utf-8 -*-
"""Customer-Review-Classification-Tutorial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-wu7utPLItUXf0rjmDmmesQEXkNpaFF7

# Tutorial: Customer Review Classification

# Welcome!

User reviews have become the driving force behind e-commerce transactions. However, these reviews are often not as clear as they appear to be. Text mining can help us to detect sentiment in user reviews and classify the reviews based on their sentiment scores.

For this tutorial, we are using [RAPYD.AI](https://www.rapyd.ai) to access the text mining service.

To use RAPYD.AI, [sign up here for free](https://www.rapyd.ai/free-trial/).

## Step 1: Cloning Repository

Let's clone a repository with some sample data and utility functions to keep this tutorial simple and focused.
"""

!git clone https://github.com/rapyd-ai/use-cases.git
import sys
sys.path.append('use-cases/classify-user-reviews/utils')
from rapyd_ai_utils import *

"""## Step 2: Load your dataset

For this prototype, we will need a dataset with some user reviews. The data has to come as a csv file without header, where each line is one text review that we want to classify.

If you don't have a text file at hand, or if you want to look at a sample, you can simply call the utility function `read_reviews()`.
"""

reviews_df = read_reviews("use-cases/classify-user-reviews/data/reviews.csv")

"""To copy a custom file from a remote URL to this training machine, execute `wget -O "data/reviews.csv" "https://insert-your-url.here/"`

In any case, check if the import succeeded by looking at the data:
"""

reviews_df.head()

"""## Step 3: Inspect the data

To get some intuition about your dataset, we will plot some basic properties of the data. 

One critical factor is the text length. Basically, a good rule of thumb is anything between 50 to 100 characters on average. 

The longer each text phrase gets, the less practical is it to squeeze the sentiment into one score.

You can view a distribution of your text lenght per line using the utility function `plot_text_length()`
"""

plot_text_length(reviews_df['Review_Text'])

"""## Step 4: Test Sentiment Scores for Reviews

Now that we checked our data, let's get some sentiment scores. 

We will try one sample sentence at first in order to get an idea of what we are getting from the AI as a Service.

To analyse a sentence using RAPYD.AI, you just have to define your Account_ID and Token. 

To see your Account ID and generate an access token, login to [https://my.rapyd.ai](https://my.rapyd.ai)

Define your credentials by assigning them to the variables `account_id` and `token`:
"""

#account_id = 'insert-your-account-id-here'
#token = 'insert-your-temporary-token-here'
account_id = 'F4hhJvA13RtjDNsnXJ6g'
token = 'eyJhbGciOiJSUzI1NiIsImtpZCI6ImRjMGMzNWZlYjBjODIzYjQyNzdkZDBhYjIwNDQzMDY5ZGYzMGZkZWEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vYXBpc2NvcGVhaSIsImF1ZCI6ImFwaXNjb3BlYWkiLCJhdXRoX3RpbWUiOjE1ODY4MDg3NDgsInVzZXJfaWQiOiJ5YmhaNUtFbndBUlJMTURZUXc0MUhET3JlMGsyIiwic3ViIjoieWJoWjVLRW53QVJSTE1EWVF3NDFIRE9yZTBrMiIsImlhdCI6MTU4NjgwODc0OCwiZXhwIjoxNTg2ODEyMzQ4LCJlbWFpbCI6Inp3aW5nbWFubkBpY2xvdWQuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImVtYWlsIjpbInp3aW5nbWFubkBpY2xvdWQuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoicGFzc3dvcmQifX0.nRyBctVm72kwlskw1kJQ6SthZ0Z7eWp4-9Tyv8mXkgdLI3fupwP5gFk9WgiZEYJAHy7X5GfhQOXpdxekYsxmXFA5QAj9OECsx-VNBONB56UqOt3_GYGjOCx40scAXtt7MsCThx4QjxMrwtbAGC4vh6ouTWtyMLBVho6oDcxKjEUpPRx_-aTmYe3H7OggB7uMd07CHpBWBmsC-0Xu74boDlzgO0GnY6gaHIILr1uzQIj63haFx5sQTCDUgnqkbulAUW_7xQRdxnyBe-kJEW59Sm1IHmlFzv_yQj94MHrNgKWWSSzCGPFhYx7LgsgoXAOqVBeff2tt9oNGmwcQlqa3yA'

"""In order to test the predictions, execute the utility function `get_sentiment_score()` on some text"""

text = "I am getting coverage drops."
get_sentiment_score(text, "AWS", "AUTO", account_id, token)

"""As you see, the output will be an array with - in this case - 4 different levels (positive, neutral, mixed, negative). 

The output format will vary depending on the AI provider you are choosing. 

You can easily switch AI providers by replacing the parameter with either GCP, AZURE or AWS!
"""

get_sentiment_score(text, "AZURE", "AUTO", account_id, token)

"""## Step 5: Get Sentiment Scores for dataset

Let's our newly gained knowledge to generate the sentiment scores for our dataset. 

But wait! Before we process our entire dataset, it is good practise to start with a sample. 

Why? Because every sentence that we send to RAPYD.AI will count against our quota limit. Thus, we want to make sure that everything is running correctly, before we spend our quota.

Let's sample our data to 50 rows for testing. If you are confident and have enough quota left you can simply skip this step or increase the sampling size.
"""

sample_size = 50
reviews_df = reviews_df.sample(sample_size).reset_index(drop=True)

"""Now, we will actually enrich our dataset with the sentiment scores by calling the function `add_sentiment_score_columns()`

Depending on the amount of sentences that you chose, this might take a while. For our sample of 50 sentences, however, this should take only a couple seconds.
"""

sentiment_df = add_sentiment_score_columns(reviews_df, "AWS", "AUTO", account_id, token)

"""If we look at our newly created object sentiment_df, you will notice the new sentiment scores:"""

sentiment_df.head()

"""## Step 6: Convert Sentiment Score to Star Rating

How do we turn our sentiment scores into a star rating?

The easiest approach is to define some static rules, that match a star rating to a certain condition. 

For our tutorial we defined the rules for converting AWS sentiment scores to a 5-star scale as follows:


*   5 Stars if positive sentiment == 1.00
*   4 Stars if positive sentiment >= 0.65
*   1 Star if negative sentiment >= 0.95
*   2 Stars if negative sentiment >= 0.85
*   3 Stars for everythin else

To apply this rule set, just call the function `aws_sentiment_to_stars()` to your dataset and save the output as a new column.
"""

sentiment_df['Stars'] = aws_sentiment_to_stars(sentiment_df)
sentiment_df.head(10)

"""## Step 7: Inspect Star Ratings

Lets find out how well our conversion perfomed by looking at the distribution of our newly created star rating. 

The ground truth of the sample dataset contained a balance of good and bad reviews, so we should see a similar amount of 1 star reviews and 5 star reviews with something going on in between.
"""

sentiment_df.Stars.hist()

"""Looks like a pretty good distribution! 

Another good quality indicator is given by looking at the star rating individually. 

Let's look at some 5-star reviews.
"""

sentiment_df.query("Stars == 5")

"""Or take a look at some rather mediocre reviews:"""

sentiment_df.query("Stars == 2 | Stars == 3")

"""Looking at your classified data will give you a good idea if your decision rules for the star ratings actually match the data, or if you need a bit more tweaking.

# Conclusion

Congratulations! You have completed the User Review Classification use case.

We hope that you found this tutorial useful. 

Feel free to play around with the technology and try some more things:

* Analyze your own data
* Analyze more data
* Compare AI Service provider

To build your AI prototype, you can start with any programming language you like. 

RAPYD.AI works as a restful HTTP-API which can be accessed form any interface.

For more information visit the [RAPYD.AI Documentation](https://www.rapyd.ai/docs/).

If you are working with Python, we recommend cloning the [git repository of this tutorial](https://github.com/rapyd-ai/use-cases/tree/master/classify-user-reviews) and start from there

**Enjoy prototyping AI-powered services!**
"""