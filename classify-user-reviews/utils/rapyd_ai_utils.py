# Packages
import pandas as pd
import requests

# Read data where one line contains one text item
def read_reviews(path = 'data/reviews.csv'):
  d = pd.read_csv(path, header=None, names=['Review_Text'])
  return d 

# Check the average text lenght and plot it as a histogram
def plot_text_length(s):
  hist = s.apply(len).hist()
  return(hist)

# Function to call sentiment score via RAPYD.AI
def get_sentiment_score(text, provider = "AUTO", language = "AUTO", account_id = '', token = ''):

  # Text Pre-Processing
  text = text.strip()
  text = text.replace("'","")
  text = text.replace('"','')

  # RAPYD.AI API Call as per https://www.rapyd.ai/docs
  url = "https://api.rapyd.ai/sentiment"
  payload = "{\n\"text\":\"" + text + "\",\n\"provider\":\"" + provider + "\",\n    \"language\":\""+ language + "\"\n}"
  headers = {
  'ACCOUNTID': account_id,
  'TOKEN': token,
  'Content-Type': 'application/json'
  }
  response = requests.request("POST", url, headers=headers, data = payload)
  
  # Response handling 
  if provider == "AWS" or provider == "AUTO":
    try:
      sentiment_score = {'positive':  round(response.json()['result']['sentimentScore']['positive'],2),
                         'mixed':  round(response.json()['result']['sentimentScore']['mixed'],2),
                         'neutral':  round(response.json()['result']['sentimentScore']['neutral'],2),
                         'negative':  round(response.json()['result']['sentimentScore']['negative'],2) }
    except:
      sentiment_score = {'positive':  None,
                         'mixed': None,
                         'neutral': None,
                         'negative': None }
      print(text, response.text)

  if provider == "GCP":
    try:
      #normalize data, GCP returns -1 to 1 for sentiment
      x = response.json()['result']['Score']
      min_x = -1
      max_x = 1
      normalized_sentiment = (x - min_x)/(max_x - min_x)
      sentiment_score = {'positive':  round(normalized_sentiment,2),
                         'negative':  round(1-normalized_sentiment,2)}
    except:
      sentiment_score = {'positive':  None,
                         'negative': None}
      print(text, response.text)


  if provider == "AZURE":
    try:
      sentiment_score = {'positive':  round(response.json()['result']['documentSentiment']['positiveScore'],2),
                         'neutral':  round(response.json()['result']['documentSentiment']['neutralScore'],2),
                         'negative':  round(response.json()['result']['documentSentiment']['negativeScore'],2) }
    except:
      sentiment_score = {'positive':  None,
                         'neutral': None,
                         'negative': None }
      print(text, response.text)


  return sentiment_score

# Add sentiment scores as additional column
def add_sentiment_score_columns(df, provider = "AUTO", language = "AUTO", account_id = '', token = ''):
  sentiment_scores = df[df.columns[0]].apply(lambda x : get_sentiment_score(text = x, provider = provider, language = language, account_id = account_id, token = token))
  sentiment_scores_df = pd.DataFrame.from_dict(list(sentiment_scores))
  df = pd.concat([df, sentiment_scores_df], axis = 1)
  return(df)


# Convert AWS sentiment scores to 5-Star-Rating scale
def aws_sentiment_to_stars(df):
  def f(row):
      if row['positive'] == 1.00:
        val = 5
      elif row['positive'] >= 0.65:
        val = 4
      elif row['negative'] >= 0.95:
        val = 1
      elif row['negative'] >= 0.85:
        val = 2
      else: 
        val = 3
      return val
  
  stars = df.apply(f, axis=1)
  return(stars)
