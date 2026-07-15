"""Busca web V4 com múltiplos provedores, cache e resultados citáveis.

Ordem: SearXNG, Brave Search, DDG e Wikipedia PT como fallback factual.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote, urlparse

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = BASE_DIR / "dados" / "web_cache_v4.sqlite3"
USER_AGENT = "Keilinks/4.0 local research assistant"
REQUEST_TIMEOUT = float(os.getenv("KEILINKS_WEB_TIMEOUT", "12"))
MAX_RESULTS = int(os.getenv("KEILINKS_WEB_MAX_RESULTS", "6"))
CACHE_TTL_SECONDS = int(os.getenv("KEILINKS_WEB_CACHE_TTL", "21600"))
TRUSTED_SUFFIXES = (".gov.br", ".gov", ".edu", ".edu.br", ".org", "who.int",
                    "wikipedia.org", "docs.python.org", "pytorch.org", "nvidia.com",
                    "github.com", "huggingface.co")
CURRENT_TERMS = {"hoje","agora","atual","atualmente","último","ultima","última",
                 "preço","preco","notícia","noticia","presidente","ceo","versão",
                 "versao","lançamento","lancamento","placar","cotação","cotacao",
                 "lei","regra","2025","2026","2027"}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    published: str = ""
    score: float = 0.0
    content: str = ""


def _connect_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS web_cache(cache_key TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL)")
    return con


def _cache_key(query): return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def cache_get(query):
    with _connect_cache() as con:
        row=con.execute("SELECT created_at,payload FROM web_cache WHERE cache_key=?",(_cache_key(query),)).fetchone()
    if not row or time.time()-float(row[0])>CACHE_TTL_SECONDS: return None
    try: return [SearchResult(**item) for item in json.loads(row[1])]
    except Exception: return None


def cache_set(query, results):
    payload=json.dumps([asdict(r) for r in results],ensure_ascii=False)
    with _connect_cache() as con:
        con.execute("INSERT OR REPLACE INTO web_cache VALUES(?,?,?)",(_cache_key(query),time.time(),payload)); con.commit()


def precisa_buscar(pergunta: str) -> bool:
    text=pergunta.lower()
    if any(term in text for term in CURRENT_TERMS): return True
    return any(re.search(p,text) for p in (r"\bquem (é|e) (o|a) atual\b",r"\bquanto custa\b",
        r"\bqual (é|e) a versão\b",r"\bpesquis[ae]\b",r"\bconfir[am]\b",r"\bna (internet|web)\b"))


def _is_safe_public_url(url):
    try:
        parsed=urlparse(url)
        if parsed.scheme not in {"http","https"} or not parsed.hostname: return False
        hostname=parsed.hostname.lower()
        if hostname in {"localhost","localhost.localdomain"} or hostname.endswith(".local"): return False
        for addr in socket.getaddrinfo(hostname,parsed.port or 443,type=socket.SOCK_STREAM):
            ip=ipaddress.ip_address(addr[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: return False
        return True
    except Exception: return False


def _query_terms(query):
    stop={"para","como","qual","quais","uma","que","isso","essa","este","com","por"}
    return {t for t in re.findall(r"[a-záàâãéèêíìîóòôõúùûç0-9]+",query.lower()) if len(t)>=3 and t not in stop}


def _score_result(query,result):
    terms=_query_terms(query); hay=f"{result.title} {result.snippet} {result.content[:2500]}".lower()
    coverage=sum(t in hay for t in terms)/max(len(terms),1)
    host=(urlparse(result.url).hostname or "").lower()
    trust=.2 if any(host==s or host.endswith(s) for s in TRUSTED_SUFFIXES) else 0
    title=sum(t in result.title.lower() for t in terms)/max(len(terms),1)*.25
    result.score=coverage+trust+title+min(len(result.content)/5000,1)*.1
    return result.score


def _search_searxng(query,limit):
    base=os.getenv("SEARXNG_URL","").rstrip("/")
    if not base: return []
    response=requests.get(f"{base}/search",params={"q":query,"format":"json","language":"pt-BR","safesearch":1},
                          headers={"User-Agent":USER_AGENT},timeout=REQUEST_TIMEOUT); response.raise_for_status()
    return [SearchResult(str(i.get("title","")),str(i.get("url","")),str(i.get("content","")),
                         "searxng",str(i.get("publishedDate","") or "")) for i in response.json().get("results",[])[:limit]]


def _search_brave(query,limit):
    key=os.getenv("BRAVE_SEARCH_API_KEY","")
    if not key: return []
    response=requests.get("https://api.search.brave.com/res/v1/web/search",
        params={"q":query,"count":limit,"search_lang":"pt-br","safesearch":"moderate"},
        headers={"Accept":"application/json","X-Subscription-Token":key,"User-Agent":USER_AGENT},timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return [SearchResult(str(i.get("title","")),str(i.get("url","")),str(i.get("description","")),
                         "brave",str(i.get("age","") or "")) for i in response.json().get("web",{}).get("results",[])[:limit]]


def _search_ddg(query,limit):
    try:
        from ddgs import DDGS
    except ImportError:
        try: from duckduckgo_search import DDGS
        except ImportError: return []
    results=[]
    with DDGS() as client:
        for i in client.text(query,region="br-pt",safesearch="moderate",
                             timelimit="y" if precisa_buscar(query) else None,max_results=limit):
            results.append(SearchResult(str(i.get("title","")),str(i.get("href",i.get("url",""))),
                str(i.get("body",i.get("snippet",""))),"duckduckgo",str(i.get("date","") or "")))
    return results


def _search_wikipedia(query,limit):
    response=requests.get("https://pt.wikipedia.org/w/api.php",params={"action":"query","list":"search",
        "srsearch":query,"srlimit":min(limit,5),"format":"json","utf8":1},headers={"User-Agent":USER_AGENT},timeout=REQUEST_TIMEOUT)
    response.raise_for_status(); results=[]
    for i in response.json().get("query",{}).get("search",[]):
        title=str(i.get("title","")); snippet=re.sub(r"<[^>]+>","",str(i.get("snippet","")))
        results.append(SearchResult(title,f"https://pt.wikipedia.org/wiki/{quote(title.replace(' ','_'))}",snippet,"wikipedia"))
    return results


def _extract_page(url):
    if not _is_safe_public_url(url): return ""
    try:
        response=requests.get(url,headers={"User-Agent":USER_AGENT,"Accept-Language":"pt-BR,pt;q=.9,en;q=.5"},
                              timeout=REQUEST_TIMEOUT,allow_redirects=True); response.raise_for_status()
        if not _is_safe_public_url(response.url): return ""
        if not any(t in response.headers.get("content-type","").lower() for t in ("text/html","text/plain")): return ""
        html=response.text[:2000000]
        try:
            import trafilatura
            text=trafilatura.extract(html,include_comments=False,include_tables=False,favor_precision=True)
            if text: return re.sub(r"\s+"," ",text).strip()[:12000]
        except ImportError: pass
        try:
            from bs4 import BeautifulSoup
            soup=BeautifulSoup(html,"html.parser")
            for tag in soup(["script","style","nav","footer","aside","form"]): tag.decompose()
            return re.sub(r"\s+"," ",soup.get_text(" ")).strip()[:12000]
        except ImportError: return re.sub(r"<[^>]+>"," ",html)[:12000]
    except Exception: return ""


def _deduplicate(results: Iterable[SearchResult]):
    unique=[]; seen=set()
    for result in results:
        if not result.url or not result.title: continue
        parsed=urlparse(result.url); key=f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
        if key in seen: continue
        seen.add(key); unique.append(result)
    return unique


def search_web(query: str,max_results: int=MAX_RESULTS,use_cache: bool=True) -> List[SearchResult]:
    query=re.sub(r"\s+"," ",query).strip()[:500]
    if not query: return []
    if use_cache:
        cached=cache_get(query)
        if cached is not None: return cached
    results=[]
    for provider in (_search_searxng,_search_brave,_search_ddg):
        try:
            results.extend(provider(query,max_results*2))
            if len(results)>=max_results: break
        except Exception as exc: print(f"[WebV4] {provider.__name__} falhou: {exc}")
    if len(results)<2 and not precisa_buscar(query):
        try: results.extend(_search_wikipedia(query,max_results))
        except Exception as exc: print(f"[WebV4] Wikipedia falhou: {exc}")
    results=_deduplicate(results)[:max_results*2]
    with ThreadPoolExecutor(max_workers=min(4,len(results) or 1)) as executor:
        futures={executor.submit(_extract_page,r.url):r for r in results[:max_results]}
        for future in as_completed(futures): futures[future].content=future.result()
    results.sort(key=lambda item:_score_result(query,item),reverse=True); results=results[:max_results]
    if use_cache and results: cache_set(query,results)
    return results


def format_context(query,results,max_chars=8000):
    if not results: return ""
    blocks=[f"[Pesquisa web atual para: {query}]"]
    for index,result in enumerate(results,1):
        excerpt=re.sub(r"\s+"," ",(result.content or result.snippet)).strip()[:1200]
        blocks.append(f"Fonte {index}: {result.title}\nURL: {result.url}\nTrecho: {excerpt}")
        if sum(map(len,blocks))>=max_chars: break
    blocks.append("Responda apenas com o que as fontes sustentam; mencione conflitos e datas.")
    return "\n\n".join(blocks)[:max_chars]


def pesquisar(pergunta: str) -> Optional[str]:
    results=search_web(pergunta)
    return "[Fonte: Web V4]\n"+format_context(pergunta,results) if results else None
