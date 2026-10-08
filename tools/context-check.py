#!/usr/bin/env python3
"""Verify an advertised 128K context with a counted long request and overflow rejection."""
import argparse, json, time, urllib.request, urllib.error
from pathlib import Path
from bench import call, long_code, reply_text, broken, AGENT_SYSTEM, AGENT_TOOLS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://127.0.0.1:8798')
    ap.add_argument('--api-key')
    ap.add_argument('--json',required=True)
    ap.add_argument('--decode-tokens',type=int,default=0,help='optional sustained decode probe at the same long context')
    ap.add_argument('--combined-agent',action='store_true',help='verify long retrieval and tool calling in one counted request')
    a=ap.parse_args()
    if a.decode_tokens < 0: ap.error('--decode-tokens must be nonnegative')
    headers={'Content-Type':'application/json'}
    if a.api_key: headers['Authorization']='Bearer '+a.api_key
    def count(text):
        req={'model':'strata','messages':[{'role':'user','content':text}],'max_tokens':128}
        q=urllib.request.Request(a.url+'/v1/messages/count_tokens',json.dumps(req).encode(),headers)
        with urllib.request.urlopen(q,timeout=60) as res:return json.load(res)['input_tokens']
    source=long_code(36000)
    filler=(source+'\n')*16
    def prompt(chars):
        third=chars//3
        return ('Remember these three exact audit labels, despite unrelated code between them.\n'
                'AUDIT_ALPHA=cedar\n'+filler[:third]+'\nAUDIT_BETA=quartz\n'+
                filler[:third]+'\nAUDIT_GAMMA=otter\n'+filler[:third]+
                '\nReturn only the three audit values in alpha,beta,gamma order. Do not execute code.')
    lo,hi=0,len(filler)
    for _ in range(10):
        mid=(lo+hi)//2
        if count(prompt(mid))<124000:lo=mid
        else:hi=mid
    text=prompt(lo); n=count(text)
    print('Long request input tokens:',n,flush=True)
    if not 120000<=n<=126000:raise RuntimeError('Prompt does not exercise 128K capacity')
    def valid_tool(response, expected):
        for tc in response.get('choices',[{}])[0].get('message',{}).get('tool_calls') or []:
            fn=tc.get('function') or {}
            try: args=json.loads(fn.get('arguments') or '{}')
            except json.JSONDecodeError: continue
            if fn.get('name')=='read_file' and args.get('path')==expected:return True
        return False
    def agent_request(task):
        return call(a.url,a.api_key,text+'\n'+task,256,
            {'stream':True,'tools':AGENT_TOOLS,'tool_choice':{'type':'function','function':{'name':'read_file'}},
             'messages':[{'role':'system','content':AGENT_SYSTEM},{'role':'user','content':text+'\n'+task}]})
    if a.combined_agent:
        # The final instruction gives no audit values: they must be retrieved
        # from the beginning and middle of the actual 124K input.
        res=agent_request("Final task: instead of a text answer, join the three AUDIT values in alpha,beta,gamma order with '/' and call read_file with that exact path.")
        answer=reply_text(res);reason=broken(answer)
        retrieved=valid_tool(res,'cedar/quartz/otter')
    else:
        res=call(a.url,a.api_key,text,128)
        answer=reply_text(res);reason=broken(answer)
        retrieved=all(x in answer.lower() for x in ['cedar','quartz','otter'])
    print('Long reply:',repr(answer), 'wall:',round(res['_wall'],2),flush=True)
    out={'counted_input_tokens':n,'response':res,'retrieval_ok':retrieved,'broken':reason,'combined_agent':a.combined_agent}
    Path(a.json).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    if a.combined_agent:
        agent=res;ok=retrieved
    else:
        agent=agent_request('Now use read_file to read README.md.')
        ok=valid_tool(agent,'README.md')
    out['long_agent_response']=agent
    out['long_agent_tool_call']=ok
    Path(a.json).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('Long streaming agent tool call:',ok,flush=True)
    if a.decode_tokens:
        task='Final task: write a complete Python implementation of a thread-safe LRU cache with get, put, delete, clear and statistics. Produce code only, at least 100 lines. Do not call tools.'
        decode=call(a.url,a.api_key,text+'\n'+task,a.decode_tokens,
            {'stream':True,'tools':AGENT_TOOLS,'tool_choice':'none',
             'messages':[{'role':'system','content':AGENT_SYSTEM},{'role':'user','content':text+'\n'+task}]})
        out['long_decode_response']=decode
        out['long_decode_broken']=broken(reply_text(decode))
        out['long_decode_sufficient']=decode.get('usage',{}).get('completion_tokens',0)>=min(192,a.decode_tokens)
        Path(a.json).write_text(json.dumps(out,ensure_ascii=False,indent=2))
        print('Long-context sustained decode:',decode.get('usage'),decode.get('timings'),flush=True)
    oversized=prompt(hi+70000)
    over_n=count(oversized)
    if over_n<131072:raise RuntimeError('Overflow control is too short')
    try:
        call(a.url,a.api_key,oversized,128)
        out['overflow_rejected']=False
    except urllib.error.HTTPError as e:
        # Store only the status; response bodies may contain request details.
        out['overflow_rejected']=e.code==400
        out['overflow_status']=e.code
    out['overflow_input_tokens']=over_n
    Path(a.json).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('retrieval_ok:',out['retrieval_ok'],'overflow_rejected:',out['overflow_rejected'],flush=True)
    return 3 if out.get('long_decode_broken') or (a.decode_tokens and not out.get('long_decode_sufficient')) or reason or not out['retrieval_ok'] or not out['overflow_rejected'] or not out['long_agent_tool_call'] else 0
if __name__=='__main__':raise SystemExit(main())
