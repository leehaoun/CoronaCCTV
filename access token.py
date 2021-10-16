import requests

url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = 'b34e330bcd9d65a5cfd4c5abb2052f16'
redirect_uri = 'https://www.example.com/oauth' # APP에서 등록한 redirect_url
authorize_code = '9b29UCAnDBHdibaER-A_NKmPbcC68ei6TXyQoS_hvnNiblkv_qYKCH9XjaBXaJrhIcLQSwo9dNoAAAF8h__qcg' 
 	#kauth.kakao.com/oauth/authorize?client_id=b34e330bcd9d65a5cfd4c5abb2052f16&redirect_uri=https://www.example.com/oauth&response_type=code
data = {
        'grant_type':'authorization_code',
        'client_id':rest_api_key,
        'redirect_uri':redirect_uri,
        'code': authorize_code,
    }

response = requests.post(url, data=data)
tokens = response.json()
print(tokens)