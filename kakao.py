import json
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

def send_message(now, stage, object):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    # 사용자 토큰
    headers = {
        "Authorization": "Bearer " + refreshToken("zymvATLqKcFezyX9245qKeb2kyU7znt_Ud2Hbwo9c5oAAAF8aCLTxA")
    }
    data = {
        "template_object" : json.dumps({ "object_type" : "text",
                                        "text" : "방역 미이행발생!!! 시간:%s, 장소:%s, 대상:%s"%(now,stage,object),
                                        "link" : {
                                                    "web_url" : "www.naver.com"
                                                }
        })      
    }
    response = requests.post(url, headers=headers, data=data)
    if response.json().get('result_code') == 0:
      print('메시지를 성공적으로 보냈습니다.')
    else:
      print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))