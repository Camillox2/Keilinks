"""
Camada de banco de dados MySQL da Keilinks
Conexao, tabelas, CRUD para knowledge, conversas, crawler_log e memoria.
"""

import os
import pymysql
from datetime import datetime

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3309,
    'user': 'root',
    'password': 'C@mill04',
    'database': 'keilinks',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def inicializar_banco():
    """Cria o banco e as tabelas se nao existirem"""
    cfg = DB_CONFIG.copy()
    db_name = cfg.pop('database')
    cfg.pop('cursorclass')

    conn = pymysql.connect(**cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
    finally:
        conn.close()

    # Agora conecta no banco e cria tabelas
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pergunta VARCHAR(500) NOT NULL,
                    resposta TEXT NOT NULL,
                    fonte VARCHAR(50) DEFAULT 'web',
                    categoria VARCHAR(50) DEFAULT 'geral',
                    url VARCHAR(500) DEFAULT NULL,
                    relevancia INT DEFAULT 0,
                    acessos INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FULLTEXT idx_busca (pergunta, resposta)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversas (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pergunta TEXT NOT NULL,
                    resposta TEXT NOT NULL,
                    fonte VARCHAR(50) DEFAULT 'desconhecido',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crawler_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    fonte VARCHAR(50) NOT NULL,
                    topico VARCHAR(200) DEFAULT NULL,
                    sucesso TINYINT(1) DEFAULT 1,
                    fatos_novos INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memoria (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chave VARCHAR(100) UNIQUE NOT NULL,
                    valor TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        print("[MySQL] Banco e tabelas prontos")
    finally:
        conn.close()


# ─── Knowledge CRUD ──────────────────────────────────────────────────────

def knowledge_adicionar(pergunta: str, resposta: str, fonte: str = 'web',
                        categoria: str = 'geral', url: str = None, relevancia: int = 0):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge (pergunta, resposta, fonte, categoria, url, relevancia) VALUES (%s, %s, %s, %s, %s, %s)",
                (pergunta[:500], resposta, fonte, categoria, url, relevancia)
            )
        conn.commit()
    finally:
        conn.close()


def knowledge_buscar(pergunta: str, limite: int = 1):
    """Busca no knowledge usando FULLTEXT do MySQL"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pergunta, resposta, fonte, id FROM knowledge WHERE MATCH(pergunta, resposta) AGAINST(%s IN NATURAL LANGUAGE MODE) LIMIT %s",
                (pergunta, limite)
            )
            resultados = cur.fetchall()
            if resultados:
                # Incrementa acessos
                cur.execute("UPDATE knowledge SET acessos = acessos + 1 WHERE id = %s", (resultados[0]['id'],))
                conn.commit()
            return resultados
    finally:
        conn.close()


def knowledge_existe(pergunta: str) -> bool:
    """Verifica se ja existe um fato similar (para deduplicacao)"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM knowledge WHERE MATCH(pergunta, resposta) AGAINST(%s IN NATURAL LANGUAGE MODE) LIMIT 1",
                (pergunta,)
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def knowledge_total() -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM knowledge")
            return cur.fetchone()['total']
    finally:
        conn.close()


def knowledge_por_fonte() -> dict:
    """Retorna contagem de fatos por fonte"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fonte, COUNT(*) AS total FROM knowledge GROUP BY fonte")
            return {r['fonte']: r['total'] for r in cur.fetchall()}
    finally:
        conn.close()


# ─── Conversas CRUD ──────────────────────────────────────────────────────

def conversa_salvar(pergunta: str, resposta: str, fonte: str = 'desconhecido'):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversas (pergunta, resposta, fonte) VALUES (%s, %s, %s)",
                (pergunta, resposta, fonte)
            )
        conn.commit()
    finally:
        conn.close()


def conversa_historico(limite: int = 50) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pergunta, resposta, fonte, created_at FROM conversas ORDER BY id DESC LIMIT %s",
                (limite,)
            )
            rows = cur.fetchall()
            for r in rows:
                if isinstance(r['created_at'], datetime):
                    r['created_at'] = r['created_at'].isoformat()
                r['data'] = r.pop('created_at')
            return list(reversed(rows))
    finally:
        conn.close()


# ─── Crawler Log ─────────────────────────────────────────────────────────

def crawler_log_salvar(fonte: str, topico: str = None, sucesso: bool = True, fatos_novos: int = 0):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO crawler_log (fonte, topico, sucesso, fatos_novos) VALUES (%s, %s, %s, %s)",
                (fonte, topico, sucesso, fatos_novos)
            )
        conn.commit()
    finally:
        conn.close()


def crawler_log_recentes(limite: int = 20) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fonte, topico, sucesso, fatos_novos, created_at FROM crawler_log ORDER BY id DESC LIMIT %s",
                (limite,)
            )
            rows = cur.fetchall()
            for r in rows:
                if isinstance(r['created_at'], datetime):
                    r['created_at'] = r['created_at'].isoformat()
            return list(reversed(rows))
    finally:
        conn.close()


# ─── Memoria CRUD ────────────────────────────────────────────────────────

def memoria_get(chave: str, default=None):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT valor FROM memoria WHERE chave = %s", (chave,))
            row = cur.fetchone()
            return row['valor'] if row else default
    finally:
        conn.close()


def memoria_set(chave: str, valor: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memoria (chave, valor) VALUES (%s, %s) ON DUPLICATE KEY UPDATE valor = %s",
                (chave, valor, valor)
            )
        conn.commit()
    finally:
        conn.close()


def memoria_todos() -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chave, valor FROM memoria")
            return {r['chave']: r['valor'] for r in cur.fetchall()}
    finally:
        conn.close()


# ─── Migracao JSON -> MySQL ──────────────────────────────────────────────

def migrar_json_para_mysql(base_dir: str):
    """Importa dados dos JSONs existentes para o MySQL"""
    import json

    # Migra knowledge.json
    knowledge_path = os.path.join(base_dir, 'dados', 'knowledge.json')
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            try:
                fatos = json.load(f)
            except json.JSONDecodeError:
                fatos = []

        if fatos:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    for fato in fatos:
                        cur.execute(
                            "INSERT INTO knowledge (pergunta, resposta, fonte, url) VALUES (%s, %s, %s, %s)",
                            (
                                fato.get('pergunta', '')[:500],
                                fato.get('resposta', ''),
                                fato.get('fonte', 'web')[:50],
                                fato.get('url', None),
                            )
                        )
                conn.commit()
                print(f"[MySQL] Migrados {len(fatos)} fatos do knowledge.json")
            finally:
                conn.close()

    # Migra historico.json
    historico_path = os.path.join(base_dir, 'dados', 'historico.json')
    if os.path.exists(historico_path):
        with open(historico_path, 'r', encoding='utf-8') as f:
            try:
                historico = json.load(f)
            except json.JSONDecodeError:
                historico = []

        if historico:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    for h in historico:
                        cur.execute(
                            "INSERT INTO conversas (pergunta, resposta, fonte) VALUES (%s, %s, %s)",
                            (h.get('pergunta', ''), h.get('resposta', ''), h.get('fonte', 'desconhecido'))
                        )
                conn.commit()
                print(f"[MySQL] Migradas {len(historico)} conversas do historico.json")
            finally:
                conn.close()

    # Migra memoria.json
    memoria_path = os.path.join(base_dir, 'dados', 'memoria.json')
    if os.path.exists(memoria_path):
        with open(memoria_path, 'r', encoding='utf-8') as f:
            try:
                dados = json.load(f)
            except json.JSONDecodeError:
                dados = {}

        if dados:
            for chave, valor in dados.items():
                if isinstance(valor, (list, dict)):
                    import json as j
                    memoria_set(chave, j.dumps(valor, ensure_ascii=False))
                else:
                    memoria_set(chave, str(valor))
            print(f"[MySQL] Migradas {len(dados)} chaves de memoria.json")


if __name__ == '__main__':
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Inicializando banco MySQL...")
    inicializar_banco()
    if '--migrar' in sys.argv:
        print("Migrando JSONs para MySQL...")
        migrar_json_para_mysql(base)
    print("Pronto!")
