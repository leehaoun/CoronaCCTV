import json
import requests

def send_message(now, stage, object):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    # 사용자 토큰
    headers = {
        "Authorization": "Bearer " + "9zpNJy0Mdg79rQeUUfXx4aum-cPViy0Zo6U4QAo9dBEAAAF8Y3Qf9A"
    }
    data = {
        "template_object" : json.dumps({ "object_type" : "text",
                                        "text" : "방역 미이행발생!!! 시간:%s, 장소:%s, 대상:%s"%(now,stage,object),
                                        "link" : {
                                                    "web_url" : "www.naver.com"
                                                }
        })
    }

    response = requests.post(url, headers=headers, data=data, verify=False)
