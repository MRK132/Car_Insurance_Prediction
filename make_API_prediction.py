import json
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess # import preprocessing code

df = pd.read_csv("./carInsurance_train.csv",index_col=0)
df = preprocess.preprocess(df)

y = df['CarInsurance']
X = df.drop(columns=['CarInsurance'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sampling random customer from test set:

def retrieve_sample_customer_from_test_set():
    sample_customer = X_test.sample()
    customer_id_number=sample_customer.index
    customer_id_number = customer_id_number[0]
    return sample_customer, customer_id_number
    
print('######## Sample Customer: ########')

SAMPLE = True
while SAMPLE == True:
    try:
        sample_customer, customer_id_number = retrieve_sample_customer_from_test_set()
        
        print(f"Ground truth value for this customer: {y_test.iloc[customer_id_number]}")
        print(f"Customer ID: {customer_id_number}")
        break
    except:
        continue
        
sample_customer = sample_customer.to_dict(orient='index') # convert to dictionary format - json readable
print(sample_customer)

sample_features = sample_customer[customer_id_number]
query=json.dumps(sample_features)

# update json query with customer id and ground truth label for reference:
# parsing JSON string:
query = json.loads(query)
  
# update the query:
id_num={"Customer_ID":int(customer_id_number)}
act_val = {"Actual":int(y_test.iloc[customer_id_number])}
query.update(id_num)
query.update(act_val)
 
# new query with CustomerID and Actual value - Note these 2 variables will not be used
# in model prediction, only for displaying at final result - they are 'popped' from the json 
# in app.py for the API service and printed on-screen

url = 'https://car-insurance-prediction-2hapnkm3wq-nw.a.run.app/'  # live url

response = requests.get(url,query) 
print('##################################')
print(f"Prediction for customer {customer_id_number}:")
print(response.json())