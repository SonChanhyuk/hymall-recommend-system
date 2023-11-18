import json
import random
import pymysql
import uvicorn
from fastapi import FastAPI, Query, Request
from elasticsearch import Elasticsearch
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
origins = [
    "*"
]

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



conn = pymysql.connect(host='localhost', user='root', password='', db='commerce',
                       charset='utf8mb4')

es = Elasticsearch(hosts=[{'host': '', 'port': 9243, 'scheme': "https"}],
                   request_timeout=300, max_retries=10, retry_on_timeout=True,
                   basic_auth=('', '')
                   )

@app.get("/")
async def get_items():
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT * FROM items_table ORDER BY RAND() LIMIT 50;"
    cursor.execute(sql)
    data = cursor.fetchall()
    hits = [{"_source": d} for d in data]
    return {"hits": {"hits": hits}}


@app.post("/reco_api")
async def reco_api(request: Request):
    data = await request.json()  # json() 메서드를 이용하여 요청 body에서 데이터를 가져옴
    item_no_list = data.get('item_no',[])

    item_no_str = ",".join(item_no_list)
    item_no = item_no_str.split(',')

    item_vectors = []
    for item in item_no:
        query_dsl = {"_id": item}
        item_vectors.append(query_dsl)
    print(item_vectors)

    multi_query = {
        "docs": item_vectors,
        "_source": ["reco_vector"]
    }
    vector_res = es.mget(index="hyreco512", body=multi_query)

    vectorList = []
    for item in vector_res['docs']:
        if 'reco_vector' in item['_source']:
            vectorList.append(item['_source']['reco_vector'])

    itemList = []
    for j in vectorList:
        query_dsl2 = {
            "script_score": {
                "query": {"bool": {"must": [{"exists": {"field": "reco_vector"}}]}},
                "script": {
                    "source": "def score = cosineSimilarity(params.query_vector, 'reco_vector'); if (Double.isFinite(score)) { return score + 1.0; } else { return 0; }",
                    "params": {"query_vector": j}

                }
            }
        }

        res2 = es.search(index="hyreco512", query=query_dsl2, size=20,
                         filter_path=['hits.hits._id', 'hits.hits._source.image_name'])
        json_res2 = json.dumps(res2, default=dict)
        data = json.loads(json_res2)
        hits = data.get('hits')
        if hits:
            res3 = hits.get('hits', [])
            if res3:
                itemList.extend(res3)

    reco_count = 20
    reco_list = []
    if itemList:
        reco_items = random.sample(itemList, min(len(itemList), reco_count))
        reco_list = [{"item_no_idx": item["_id"], "image_name": item["_source"]["image_name"]} for item in reco_items]
    else:
        # 벡터 값이 존재하지 않을 경우 랜덤하게 20개의 상품을 불러옴
        res4 = es.search(index="hyreco512", query={"match_all": {}}, size=20,
                         filter_path=['hits.hits._id', 'hits.hits._source.image_name'])
        json_res4 = json.dumps(res4, default=dict)
        data = json.loads(json_res4)
        hits = data.get('hits')
        if hits:
            reco_items = hits.get('hits', [])
            reco_list = [{"item_no_idx": item["_id"], "image_name": item["_source"]["image_name"]} for item in reco_items]

    return JSONResponse(content=reco_list)

@app.post("/retrieval_api")
async def retrieval_api(request: Request):
    data = await request.json()
    item_no = data.get("item_no")

    if item_no is None:
        return JSONResponse(content={"error": "Item number is required."}, status_code=400)

    query_dsl = {
        "query": {
            "term": {"_id": item_no}
        },
        "_source": ["retrieval_vector"]
    }

    res = es.search(index="hyreco512", body=query_dsl)

    hits = res.get("hits", {}).get("hits", [])
    if hits:
        retrieval_vector = hits[0].get("_source", {}).get("retrieval_vector")
        if retrieval_vector is not None:
            query_dsl2 = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, doc['retrieval_vector']) + 1.0",
                            "params": {"query_vector": retrieval_vector}
                        }
                    }
                },
                "_source": ["image_name"]
            }

            res2 = es.search(index="hyreco512", body=query_dsl2, size=10)
            return res2

    return {"hits": {"hits": []}}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)