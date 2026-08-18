import requests
import time

HEALTH_URL = "http://127.0.0.1:7890/health"

def check():
    try:
        r = requests.get(HEALTH_URL,timeout=3)
        if r.status_code ==200:
            return True
    except Exception as e:
        print("服务异常:",e)
    return False

if __name__ == "__main__":
    while True:
        ok = check()
        if not ok:
            print("RAG服务挂掉，需要重启")
        time.sleep(10)