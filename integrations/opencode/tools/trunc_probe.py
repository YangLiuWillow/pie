import json,sys,threading,urllib.request
url,model,key = sys.argv[1],sys.argv[2],(sys.argv[3] if len(sys.argv)>3 else "bench")
H={"Content-Type":"application/json","Authorization":f"Bearer {key}"}
def go(i,out):
    b={"model":model,"messages":[{"role":"user","content":f"Case {i}: count from one to forty in words."}],
       "max_tokens":64,"temperature":0,"stream":False}
    r=urllib.request.Request(f"{url}/v1/chat/completions",data=json.dumps(b).encode(),headers=H)
    try:
        d=json.loads(urllib.request.urlopen(r,timeout=600).read())
        c=d["choices"][0]
        out[i]=(c.get("finish_reason"),(d.get("usage") or {}).get("completion_tokens"))
    except Exception as e: out[i]=(f"ERR {type(e).__name__}",0)
for label,n in [("sequential",1),("concurrent x2",2),("concurrent x4",4)]:
    out={}
    ts=[threading.Thread(target=go,args=(i,out)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    toks=[out[k][1] for k in sorted(out)]; fin=[out[k][0] for k in sorted(out)]
    print(f"  {label:14s} tokens={toks}  finish={fin}")
