import requests

def refreshToken(refresh_token) -> str:
    REST_API_KEY = "b34e330bcd9d65a5cfd4c5abb2052f16"
    REDIRECT_URI = "https://kauth.kakao.com/oauth/token"

    data = {
        "grant_type": "refresh_token", # 얘는 단순 String임. "refresh_token"
        "client_id":f"{REST_API_KEY}",
        "refresh_token": refresh_token # 여기가 위에서 얻은 refresh_token 값
    }    
 
    resp = requests.post(REDIRECT_URI , data=data)
    new_token = resp.json()

    return new_token['access_token']
refreshToken("Z8lJ5AFIQOshY5KwRfwFLhijoW_HdakPy15uGgopcJ4AAAF8aBWALw")