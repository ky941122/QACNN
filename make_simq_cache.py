import jieba
import json

def make():
    f1 = open("data/database", "r")
    database = json.load(f1)
    database = database["data"]

    tokenizer = jieba.Tokenizer()
    tokenizer.load_userdict("data/tokenize_dic4j")

    f2 = open("data/entities_xiaomiwang4j", "r")
    entities = f2.readlines()
    entities = [entity.strip().lower() for entity in entities]

    cache = dict()

    for data in database:
        if "similarQueries" not in data:
            continue
        simqs = data["similarQueries"]
        if len(simqs) > 0:
            for simq in simqs:
                if "query" not in simq:
                    continue
                q = simq["query"]
                q = q.strip().lower()

                if q not in cache:
                    cache[q] = []
                else:
                    print(q)
                    continue

                ws = tokenizer.cut(q)
                for w in ws:
                    w = w.strip()
                    if w in entities:
                        w = "[ENT]"
                    cache[q].append(w)

    writejson = json.dumps(cache)
    f3 = open("simq_cache_2", "w")
    f3.write(writejson)



if __name__ == "__main__":
    make()


