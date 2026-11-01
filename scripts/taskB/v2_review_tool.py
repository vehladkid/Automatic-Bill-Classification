"""V2 step 4: smallest useful local annotation utility - one static HTML file (no server, no dashboard) + a flat CSV of the SUGGESTION_ONLY candidates.
Run: python scripts/taskB/v2_review_tool.py  ->  open datasets/interim/v2_pool/review/index.html in a browser (images are loaded from ../images/).
The reviewer sees the corrected page with OCR boxes, accepts a suggestion / edits the value / sets null / draws or re-draws a box / marks the page unusable, and exports annotations.json."""
import json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "datasets" / "interim" / "v2_pool"
FIELDS = ["merchant_name", "gstin", "invoice_number", "invoice_date", "subtotal", "cgst", "sgst", "igst", "hsn_sac", "total_amount", "tax_total", "payment_method"]

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>V2 invoice review</title><style>
body{margin:0;font:13px system-ui;display:flex;height:100vh;background:#222;color:#eee}
#list{width:230px;overflow:auto;background:#1b1b1b}#list div{padding:4px 8px;cursor:pointer;border-bottom:1px solid #333}#list .cur{background:#355}
.st-done{color:#7d7}.st-unusable{color:#f77;text-decoration:line-through}
#mid{flex:1;overflow:auto;position:relative;background:#444}#side{width:560px;overflow:auto;background:#1b1b1b;padding:6px}
table{width:100%;border-collapse:collapse}td{padding:2px;vertical-align:top;border-bottom:1px solid #333}input[type=text]{width:100%;box-sizing:border-box;background:#111;color:#eee;border:1px solid #555}
select{max-width:150px;background:#111;color:#eee}button{margin:1px}.armed{background:#a60}.tag{font-size:11px;color:#9cf}.warn{color:#fc6}
</style></head><body><div id="list"></div><div id="mid"><div style="position:relative;display:inline-block" id="wrap"><img id="img"><svg id="svg" style="position:absolute;left:0;top:0"></svg></div></div><div id="side"></div>
<script>
const DATA=__DATA__, FIELDS=__FIELDS__, KEY='v2_review_v1';
let st=JSON.parse(localStorage.getItem(KEY)||'{}'), cur=0, armed=null, scale=0.5, drag=null;
const COL={merchant_name:'#f0f',gstin:'#0ff',invoice_number:'#ff0',invoice_date:'#fa0',subtotal:'#0f0',cgst:'#6f6',sgst:'#9f9',igst:'#6cf',hsn_sac:'#f66',total_amount:'#f33',tax_total:'#c8f',payment_method:'#ccc'};
function D(){return DATA[cur]} function S(){const id=D().document_id;return st[id]=st[id]||{annotator:'',verified:false,unusable:false,notes:'',fields:{}}}
function save(){localStorage.setItem(KEY,JSON.stringify(st))}
function F(f){const s=S();return s.fields[f]=s.fields[f]||{status:'unset',value:null,bbox:null,orig:null}}
function renderList(){const L=document.getElementById('list');L.innerHTML='';DATA.forEach((d,i)=>{const s=st[d.document_id]||{};const e=document.createElement('div');e.className=(i==cur?'cur ':'')+(s.unusable?'st-unusable':s.verified?'st-done':'');e.textContent=d.tier+' '+d.document_id.slice(-15)+' '+d.supplier_candidate.slice(0,14);e.onclick=()=>{cur=i;load()};L.appendChild(e)})}
function load(){const d=D();const im=document.getElementById('img');im.onload=()=>{fit()};im.src='../images/'+d.document_id+'.jpg';renderList();renderSide()}
function fit(){const im=document.getElementById('img');const w=document.getElementById('mid').clientWidth;scale=Math.max(0.1,Math.min(((w>200?w:900)-10)/im.naturalWidth,1.5));zoom()}
function zoom(){const im=document.getElementById('img');im.width=im.naturalWidth*scale;const sv=document.getElementById('svg');sv.setAttribute('width',im.width);sv.setAttribute('height',im.height);drawBoxes()}
function drawBoxes(){const d=D(),sv=document.getElementById('svg');let h='';d.ocr_boxes.forEach((b,i)=>{const[x0,y0,x1,y1]=b.bbox;h+=`<rect data-i="${i}" x="${x0*scale}" y="${y0*scale}" width="${(x1-x0)*scale}" height="${(y1-y0)*scale}" fill="rgba(80,160,255,0.08)" stroke="rgba(80,160,255,0.5)" stroke-width="0.7"/>`});
 Object.entries(S().fields).forEach(([f,v])=>{if(v.bbox){const[x0,y0,x1,y1]=v.bbox;h+=`<rect x="${x0*scale}" y="${y0*scale}" width="${(x1-x0)*scale}" height="${(y1-y0)*scale}" fill="none" stroke="${COL[f]}" stroke-width="2.5"/><text x="${x0*scale}" y="${y0*scale-2}" fill="${COL[f]}" font-size="11">${f}</text>`}});
 if(drag&&drag.cur){const[x0,y0,x1,y1]=drag.rect();h+=`<rect x="${x0*scale}" y="${y0*scale}" width="${(x1-x0)*scale}" height="${(y1-y0)*scale}" fill="rgba(255,255,0,.15)" stroke="yellow"/>`}sv.innerHTML=h}
function setField(f,o){const x=F(f);Object.assign(x,o);save();drawBoxes();renderSide()}
function wordsIn(r){return D().ocr_boxes.filter(b=>{const cx=(b.bbox[0]+b.bbox[2])/2,cy=(b.bbox[1]+b.bbox[3])/2;return cx>=r[0]&&cx<=r[2]&&cy>=r[1]&&cy<=r[3]})}
const svg=document.getElementById('svg');
function pt(e){const r=svg.getBoundingClientRect();return[(e.clientX-r.left)/scale,(e.clientY-r.top)/scale]}
svg.onmousedown=e=>{if(!armed)return;const p=pt(e);drag={a:p,cur:p,rect(){return[Math.min(this.a[0],this.cur[0]),Math.min(this.a[1],this.cur[1]),Math.max(this.a[0],this.cur[0]),Math.max(this.a[1],this.cur[1])]}}};
svg.onmousemove=e=>{if(drag){drag.cur=pt(e);drawBoxes()}};
svg.onmouseup=e=>{if(!drag)return;drag.cur=pt(e);let r=drag.rect();drag=null;const f=armed;
 if(r[2]-r[0]<4&&r[3]-r[1]<4){const hit=D().ocr_boxes.filter(b=>r[0]>=b.bbox[0]&&r[0]<=b.bbox[2]&&r[1]>=b.bbox[1]&&r[1]<=b.bbox[3]);if(!hit.length){drawBoxes();return}r=hit[0].bbox.slice()}
 const ws=wordsIn(r).sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]);const x=F(f);
 setField(f,{bbox:ws.length?[Math.min(...ws.map(w=>w.bbox[0])),Math.min(...ws.map(w=>w.bbox[1])),Math.max(...ws.map(w=>w.bbox[2])),Math.max(...ws.map(w=>w.bbox[3]))]:r.map(Math.round),value:x.status=='edited'&&x.value?x.value:(ws.map(w=>w.text).join(' ')||x.value),status:'edited'})};
function renderSide(){const d=D(),s=S();let h=`<div><b>${d.document_id}</b> <span class=tag>tier ${d.tier} · ${d.supplier_candidate} · rot ${d.corrected_orientation.rotation_ccw_deg}°</span></div><div class=warn>Everything below marked SUGGESTION_ONLY is regex/OCR output, not ground truth. Check against the image.</div>
<div>annotator <input type=text id=ann style="width:140px" value="${s.annotator}"> <label><input type=checkbox id=ver ${s.verified?'checked':''}> verified against image</label> <label><input type=checkbox id=unu ${s.unusable?'checked':''}> unusable</label> zoom <input type=range min=15 max=200 value=${Math.round(scale*100)} id=zm></div>
<div><input type=text id=notes placeholder="notes" value="${s.notes.replace(/"/g,'&quot;')}"></div><table>`;
 FIELDS.forEach(f=>{const x=F(f),c=d.candidates[f]||[];h+=`<tr><td style="width:105px;color:${COL[f]}">${f}<br><span class=tag>${x.status}</span></td><td><input type=text data-f="${f}" value="${(x.value??'').toString().replace(/"/g,'&quot;')}" placeholder="${x.status=='null'?'NULL (field absent)':''}">
 <select data-s="${f}"><option value="">suggestions(${c.length})</option>${c.map((k,i)=>`<option value="${i}">${k.value} [SUGGESTION_ONLY]</option>`).join('')}</select>
 <button data-a="${f}" class="${armed==f?'armed':''}">${armed==f?'drawing…':'box'}</button><button data-n="${f}">null</button></td></tr>`});
 h+=`</table><button id=exp>Export annotations.json</button> <button id=imp>Clear this doc</button> <span class=tag>keys: n/p next/prev</span>`;document.getElementById('side').innerHTML=h;
 const sd=document.getElementById('side');sd.querySelectorAll('input[data-f]').forEach(i=>i.onchange=()=>{const f=i.dataset.f;setField(f,{value:i.value||null,status:i.value?'edited':'unset'})});
 sd.querySelectorAll('select[data-s]').forEach(i=>i.onchange=()=>{if(i.value==='')return;const f=i.dataset.s,k=d.candidates[f][+i.value];setField(f,{value:k.value,bbox:k.bbox,status:'accepted',orig:k.value})});
 sd.querySelectorAll('button[data-a]').forEach(b=>b.onclick=()=>{armed=armed==b.dataset.a?null:b.dataset.a;renderSide()});
 sd.querySelectorAll('button[data-n]').forEach(b=>b.onclick=()=>setField(b.dataset.n,{value:null,bbox:null,status:'null'}));
 document.getElementById('ann').onchange=e=>{s.annotator=e.target.value;save()};document.getElementById('ver').onchange=e=>{s.verified=e.target.checked;save();renderList()};
 document.getElementById('unu').onchange=e=>{s.unusable=e.target.checked;save();renderList()};document.getElementById('notes').onchange=e=>{s.notes=e.target.value;save()};
 document.getElementById('zm').oninput=e=>{scale=e.target.value/100;zoom()};document.getElementById('imp').onclick=()=>{delete st[d.document_id];save();renderSide();drawBoxes()};
 document.getElementById('exp').onclick=exportJSON}
function exportJSON(){const out=DATA.map(d=>{const s=st[d.document_id];if(!s)return null;const f={};FIELDS.forEach(k=>{const x=s.fields[k]||{status:'unset',value:null,bbox:null,orig:null};const sug=(d.candidates[k]||[]).map(c=>c.value);
 f[k]={value:x.status=='unset'?null:x.value,bbox:x.bbox,status:x.status,source:x.status=='accepted'?'suggestion_accepted':x.status=='edited'?'human_edit':x.status=='null'?'human_null':'unreviewed',document_id:d.document_id,
 original_suggestion:x.orig??(sug[0]??null),final_value:x.status=='unset'?null:x.value,verified_against_image:!!s.verified,agrees_with_top_suggestion:sug.length?(x.value===sug[0]):null}});
 return{document_id:d.document_id,tier:d.tier,supplier:d.supplier_candidate,working_image:d.working_image,original_image:d.original_image,annotation_status:s.unusable?'unusable':s.verified?'verified':'in_progress',annotator:s.annotator,verified_against_image:!!s.verified,notes:s.notes,fields:f}}).filter(Boolean);
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='annotations.json';a.click()}
document.onkeydown=e=>{if(e.target.tagName=='INPUT'||e.target.tagName=='SELECT')return;if(e.key=='n'){cur=Math.min(cur+1,DATA.length-1);load()}if(e.key=='p'){cur=Math.max(cur-1,0);load()}};
load();
</script></body></html>"""

def main():
    rev = json.loads((POOL / "review_candidates.json").read_text())
    (POOL / "review").mkdir(exist_ok=True)
    (POOL / "review" / "index.html").write_text(HTML.replace("__DATA__", json.dumps(rev, ensure_ascii=False)).replace("__FIELDS__", json.dumps(FIELDS)))
    with open(POOL / "review" / "review_candidates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["document_id", "tier", "original_image", "working_image", "rotation_ccw_deg", "supplier_candidate"] + [k + "_candidates(SUGGESTION_ONLY)" for k in FIELDS] + ["ocr_text"])
        for r in rev:
            w.writerow([r["document_id"], r["tier"], r["original_image"], r["working_image"], r["corrected_orientation"]["rotation_ccw_deg"], r["supplier_candidate"]] +
                       [" || ".join(c["value"] for c in r["candidates"].get(k, [])) for k in FIELDS] + [r["ocr_text"].replace("\n", " / ")])
    print("wrote", len(rev), "docs ->", POOL / "review" / "index.html")

if __name__ == "__main__":
    main()
