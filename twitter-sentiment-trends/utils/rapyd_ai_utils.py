# Function to call sentiment score via RAPYD.AI
def get_sentiment_score(text, provider = "AUTO", language = "AUTO", account_id = '', token = ''):

  # Text Pre-Processing
  text = text.strip()
  text = text.replace("'","")
  text = text.replace('"','')
  text = text.replace('’','')

  # RAPYD.AI API Call as per https://www.rapyd.ai/docs
  url = "https://api.rapyd.ai/sentiment"
  payload = "{\n\"text\":\"" + text + "\",\n\"provider\":\"" + provider + "\",\n    \"language\":\""+ language + "\"\n}"
  headers = {
  'ACCOUNTID': account_id,
  'TOKEN': token,
  'Content-Type': 'application/json'
  }
  response = requests.request("POST", url, headers=headers, data = payload.encode('UTF-8'))
  
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
  
def hashtag_frequency(s):
  hashtags = s
  # Clear stopwords
  ignored_hashtags = ['#covid-19','#coronavirus','#covid19','#covid_19', '#corona', '#covid', '#coronakrise', '#covid2019', '#covid2019de', '#covid19de'. '#coronavirusde']
  hashtags = hashtags.split()
  hashtags  = [word for word in hashtags if word.lower() not in ignored_hashtags]
  hashtags = ' '.join(hashtags)
  hashtags_list = hashtags.split()
  hashtag_freq = []
  for w in hashtags_list:
      hashtag_freq.append(hashtags_list.count(w))
  pairs = list(zip(hashtags_list, hashtag_freq))
  pairs = [i for n, i in enumerate(pairs) if i not in pairs[n + 1:]]
  return(pairs)
