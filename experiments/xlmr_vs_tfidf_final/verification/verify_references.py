"""Checks every DOI in the report reference list against Crossref and every arXiv id against the arXiv API (title, authors, venue, volume/issue/pages, year are printed for comparison with the report). Needs network."""
import json, urllib.request, re, sys
dois = {1:"10.1145/3503161.3548112",2:"10.18653/v1/2021.acl-long.201",4:"10.1109/ACCESS.2022.3164674",5:"10.1016/j.ipm.2024.104046",6:"10.18653/v1/2023.eacl-main.195",8:"10.1145/3749369",10:"10.18653/v1/2022.acl-long.534",16:"10.3390/make7040167",17:"10.1016/j.neucom.2021.04.114",18:"10.1007/s10462-023-10562-9",
 19:"10.1016/j.image.2021.116601",20:"10.1109/access.2021.3096739",21:"10.1109/access.2024.3425910",22:"10.1007/s10489-022-04306-5",23:"10.1016/j.neucom.2023.127223",24:"10.1007/s10032-024-00493-8",25:"10.1007/s00521-024-09505-4",26:"10.1109/tnnls.2020.3047335",27:"10.1016/j.comnet.2022.109467"}
for k, d in dois.items():
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.crossref.org/works/{d}?mailto=student@example.org"), timeout=40))["message"]
        au = "; ".join(f"{a.get('given','')} {a.get('family','')}" for a in m.get("author", []))
        iss = m.get("issued", {}).get("date-parts", [[None]])[0][0]; pr = m.get("published-print", {}).get("date-parts", [[None]])[0][0]
        print(f"[{k}] {m['title'][0]} | {au[:200]} | {m.get('container-title',[''])[0]} | issued {iss} print {pr} | vol {m.get('volume')} no {m.get('issue')} pp {m.get('page')} art {m.get('article-number')} | {m.get('type')} | {m.get('publisher')}")
    except Exception as e: print(f"[{k}] {d} ERROR {e}")
ax = {3:"2203.02378",7:"2111.08609",9:"2510.13366",11:"2106.11539",12:"2111.15664",13:"2201.03545",14:"1706.04599",15:"2103.10213"}
x = urllib.request.urlopen("http://export.arxiv.org/api/query?id_list=" + ",".join(ax.values()) + f"&max_results={len(ax)}", timeout=40).read().decode()
for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
    i = re.search(r"arxiv.org/abs/([0-9.]+)", e).group(1); t = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split()); au = ", ".join(re.findall(r"<name>(.*?)</name>", e)); y = re.search(r"<published>(\d{4})", e).group(1)
    print("arXiv", i, "|", t, "|", au[:170], "|", y)
