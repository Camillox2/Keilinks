"""Entrada recomendada da API: Web V4 e retreino automático desligado."""
from __future__ import annotations
import hmac, json, os, sys, time
from pathlib import Path
from flask import jsonify, request

BASE_DIR=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BASE_DIR))
from api import servidor as legacy
from busca.web_v4 import pesquisar as pesquisar_v4, precisa_buscar as precisa_buscar_v4

legacy.pesquisar=pesquisar_v4; legacy.precisa_buscar=precisa_buscar_v4
if os.getenv("KEILINKS_AUTO_TRAIN","0").strip()!="1": legacy.RETREINAR_A_CADA=2**63-1
ADMIN_TOKEN=os.getenv("KEILINKS_ADMIN_TOKEN","").strip()
WRITE_ENDPOINTS={"/api/ensinar","/api/crawl"}
CANDIDATES_PATH=BASE_DIR/"dados"/"v4"/"candidates"/"runtime_feedback.jsonl"


def _token():
    value=request.headers.get("X-Keilinks-Admin","").strip()
    if value: return value
    auth=request.headers.get("Authorization","")
    return auth[7:].strip() if auth.startswith("Bearer ") else ""


@legacy.app.before_request
def protect_write_endpoints():
    if request.path not in WRITE_ENDPOINTS: return None
    if not ADMIN_TOKEN and request.remote_addr in {"127.0.0.1","::1",None}: return None
    if not ADMIN_TOKEN: return jsonify({"erro":"Defina KEILINKS_ADMIN_TOKEN."}),503
    if not hmac.compare_digest(_token(),ADMIN_TOKEN): return jsonify({"erro":"Token administrativo inválido."}),403
    return None


def save_candidate(question,answer,source="runtime"):
    CANDIDATES_PATH.parent.mkdir(parents=True,exist_ok=True)
    record={"timestamp":time.time(),"source":source,"status":"pending_review",
            "messages":[{"role":"user","content":question},{"role":"assistant","content":answer}]}
    with CANDIDATES_PATH.open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(record,ensure_ascii=False)+"\n")


def safe_save_conversation(question,answer): save_candidate(question,answer,"legacy-approved-source")
legacy.salvar_conversa_txt=safe_save_conversation
app=legacy.app

if __name__=="__main__":
    legacy.inicializar()
    app.run(host=os.getenv("KEILINKS_HOST","0.0.0.0"),port=int(os.getenv("KEILINKS_PORT","5000")),debug=False,threaded=True)
