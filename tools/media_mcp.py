#!/usr/bin/env python3
"""Optional CPU audio/video bridge for OpenAI-compatible vision endpoints.
Use --stdio for MCP, or FILE --prompt TEXT for a direct request.
Requires ffmpeg; speech additionally needs faster-whisper. No GPU is used for STT.
"""
import argparse, base64, contextlib, json, mimetypes, os, shutil, subprocess, sys, tempfile
import urllib.request
from pathlib import Path

TRANSCRIBER = None

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if result.returncode:
        raise ValueError('Media conversion failed: ' + result.stderr[-1500:])
    return result.stdout

def local_source(source, work):
    if source.startswith(('https://', 'http://')):
        dest = work / 'download'
        with urllib.request.urlopen(source, timeout=60) as response, dest.open('wb') as out:
            total = 0
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > 512*1024*1024:
                    raise ValueError('Media exceeds the 512 MiB download limit')
                out.write(chunk)
        return dest
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Media file does not exist: ' + str(path))
    return path

def transcribe(path, work, language=None):
    global TRANSCRIBER
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ValueError('Speech support needs: python -m pip install -r tools/media-requirements.txt') from None
    wav = work / 'speech.wav'
    run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn','-ac','1','-ar','16000','-t','600',str(wav)])
    if TRANSCRIBER is None:
        TRANSCRIBER = WhisperModel(os.environ.get('ARC_STT_MODEL','small'), device='cpu', compute_type='int8',
                                  cpu_threads=min(4, os.cpu_count() or 1))
    segments, info = TRANSCRIBER.transcribe(str(wav),language=language,beam_size=3,vad_filter=True)
    return {'language':info.language,'segments':[{'start':round(s.start,2),'end':round(s.end,2),'text':s.text.strip()} for s in segments],
            'limit_seconds':600, 'method':'CPU speech transcription; non-speech sounds are not classified'}

def query(source, prompt, frames=6, speech=True, language=None):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('ffmpeg and ffprobe must be installed and available in PATH')
    frames = max(1,min(int(frames),16))
    with tempfile.TemporaryDirectory(prefix='arc-media-') as tmp:
        work=Path(tmp); path=local_source(source,work)
        meta=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        streams=meta.get('streams',[])
        video=next((s for s in streams if s.get('codec_type')=='video'),None)
        audio=any(s.get('codec_type')=='audio' for s in streams)
        content=[]; details={'source':source,'sampled_frames':[]}
        duration=float(meta.get('format',{}).get('duration') or (video or {}).get('duration') or 0)
        if video:
            times=[duration*(i+.5)/frames for i in range(frames)] if duration>.1 else [0]
            for i,t in enumerate(times):
                image=work/f'frame-{i}.jpg'
                run(['ffmpeg','-nostdin','-v','error','-ss',str(t),'-i',str(path),'-frames:v','1',
                     '-vf','scale=768:768:force_original_aspect_ratio=decrease','-q:v','3',str(image)])
                content += [{'type':'text','text':f'Frame at {t:.2f} seconds.'},
                            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode()}}]
                details['sampled_frames'].append(round(t,2))
        if audio and speech:
            details['speech']=transcribe(path,work,language)
            content.append({'type':'text','text':'Speech transcript (may contain recognition errors):\n'+json.dumps(details['speech'],ensure_ascii=False)})
        if not content:
            raise ValueError('No supported image/video stream or speech track found')
        content.append({'type':'text','text':prompt+'\nFor video, infer only from these sampled frames and the transcript. State uncertainty if relevant.'})
        body={'model':os.environ.get('ARC_MODEL','strata'),'messages':[{'role':'user','content':content}],
              'max_tokens':768,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
        base=os.environ.get('ARC_BASE_URL','http://127.0.0.1:8798/v1').rstrip('/')
        key=os.environ.get('ARC_API_KEY') or os.environ.get('STRATA_API_KEY') or os.environ.get('API_TOKEN')
        if not key:raise ValueError('Set ARC_API_KEY for the authenticated model API')
        req=urllib.request.Request(base+'/chat/completions',data=json.dumps(body).encode(),
                                  headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=900) as r: response=json.load(r)
        details['answer']=response['choices'][0]['message']; details['usage']=response.get('usage')
        return details

TOOLS=[{'name':'analyze_media','description':'Analyze a local image or video, or transcribe speech and answer questions about it. Video uses sampled frames; non-speech sounds are not classified. File paths belong to the computer running this MCP server.',
        'inputSchema':{'type':'object','properties':{'source':{'type':'string','description':'Local file path or HTTP(S) media URL'},'prompt':{'type':'string'},'frames':{'type':'integer','minimum':1,'maximum':16,'default':6},'speech':{'type':'boolean','default':True},'language':{'type':'string','description':'Optional ISO language code for speech, e.g. ru'}},'required':['source','prompt'],'additionalProperties':False}}]

def stdio():
    for line in sys.stdin:
        ident = None
        try:
            m=json.loads(line); ident=m.get('id'); method=m.get('method'); result={}
            if ident is None:continue
            if method=='initialize': result={'protocolVersion':m.get('params',{}).get('protocolVersion','2024-11-05'),'capabilities':{'tools':{}},'serverInfo':{'name':'arc-media','version':'1.0.0'}}
            elif method=='tools/list':result={'tools':TOOLS}
            elif method=='tools/call':
                p=m.get('params',{})
                if p.get('name')!='analyze_media':raise ValueError('Unknown tool')
                # Libraries must never write progress messages into MCP stdout.
                with contextlib.redirect_stdout(sys.stderr): result={'content':[{'type':'text','text':json.dumps(query(**p.get('arguments',{})),ensure_ascii=False)}]}
            elif method!='ping':
                print(json.dumps({'jsonrpc':'2.0','id':ident,'error':{'code':-32601,'message':'Unknown method'}}),flush=True);continue
            reply={'jsonrpc':'2.0','id':ident,'result':result}
        except Exception as e:
            if 'ident' not in locals() or ident is None:continue
            message=str(e)
            for name in ['ARC_API_KEY','STRATA_API_KEY','API_TOKEN']:
                if os.environ.get(name):message=message.replace(os.environ[name],'[REDACTED]')
            reply={'jsonrpc':'2.0','id':ident,'result':{'content':[{'type':'text','text':message}],'isError':True}}
        print(json.dumps(reply,ensure_ascii=False),flush=True)

if __name__=='__main__':
    for stream in [sys.stdin, sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',nargs='?'); parser.add_argument('--stdio',action='store_true')
    parser.add_argument('--prompt',default='Опиши содержимое по-русски.')
    parser.add_argument('--frames',type=int,default=6);parser.add_argument('--no-speech',action='store_true');parser.add_argument('--language')
    args=parser.parse_args()
    if args.stdio:stdio()
    elif args.source:print(json.dumps(query(args.source,args.prompt,args.frames,not args.no_speech,args.language),ensure_ascii=False,indent=2))
    else:parser.error('Provide a media file or --stdio')
