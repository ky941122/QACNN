#coding=utf-8
import sys 
import requests
import json
import threading
#import uniout
reload(sys)
sys.setdefaultencoding("utf-8")

#url = "localhost:9200/mi_update_v2/_search?pretty"
#url = "http://10.136.12.21:9250/new_net_v2/_search?pretty" ##dev02
#url = "http://10.136.32.33:9250/mi_v2/_search?pretty" ##preview
url = "http://10.136.32.33:9250/new_net_v2/_search?pretty" ##preview
headers = {"Content-Type": "application/json"}

post_json =  \
{
    "size": 200,
    "query":{
        "match" : { 
            "question" : { 
                "query" : "九号平衡车plus相比九号平衡车有什么优点",
                "type" : "boolean"
            }
        }
    }   
}

lock = threading.Lock()
#out_file = open("./data/nsa.train.esinfo", "a+")
#out_file = open("./data/nsa.train.esinfo", "w")

def get_esinfo(query_stdQ_dict):
    #print 'I am %s. start' % (threading.currentThread().getName(), )
    for query, stdQ in query_stdQ_dict.items():
        post_json["query"]["match"]["question"]["query"] = query.strip()
        params = json.dumps(post_json)
        r = requests.post(url, params,headers=headers)
        r = json.loads(r.text)    
        r["ask_query"] = query.strip()
        r["stdQ"] = stdQ.strip()
        #out_file.write(json.dumps(r)+"\n")
        if lock.acquire():
            print(json.dumps(r))
            lock.release()
    #print 'I am %s. done' % (threading.currentThread().getName(), )
        
threads = []
count = 0
query_list = []
query_stdQ_dict = {}

for line in sys.stdin:
    datas = line.strip().split("\t")
    query = datas[0].encode("utf-8")
    stdQ = datas[1].encode("utf-8")
    if count == 100:
        threads.append(threading.Thread(target=get_esinfo, args=(query_stdQ_dict, )))
        count = 0
        query_stdQ_dict = {}
    query_stdQ_dict[query] = stdQ
    count += 1

threads.append(threading.Thread(target=get_esinfo, args=(query_stdQ_dict, )))

for t in threads:
    #t.setDaemon(True)
    t.start()

for t in threads:
    t.join()
        
