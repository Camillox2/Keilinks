"""
Camada de banco de dados MySQL da Keilinks
Conexao, tabelas, CRUD para knowledge, conversas, crawler_log, memoria, usuarios e chats.
"""

import os
import hashlib
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    senha_hash VARCHAR(255) NOT NULL,
                    nome VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    usuario_id INT NOT NULL,
                    titulo VARCHAR(200) DEFAULT 'Nova conversa',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            try:
                cur.execute("ALTER TABLE conversas ADD COLUMN chat_id INT DEFAULT NULL")
            except pymysql.err.OperationalError:
                pass 
            try:
                cur.execute("ALTER TABLE conversas ADD COLUMN usuario_id INT DEFAULT NULL")
            except pymysql.err.OperationalError:
                pass
        conn.commit()
        print("[MySQL] Banco e tabelas prontos")
    finally:
        conn.close()


def knowledge_adicionar(pergunta: str, resposta: str, fonte: str = 'web',
                        categoria: str = 'geral', url: str = None, relevancia: int = 0):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM knowledge WHERE pergunta = %s LIMIT 1",
                (pergunta[:500],)
            )
            if cur.fetchone():
                return

            resp_inicio = resposta[:200] if resposta else ''
            if resp_inicio:
                cur.execute(
                    "SELECT id FROM knowledge WHERE LEFT(resposta, 200) = %s LIMIT 1",
                    (resp_inicio,)
                )
                if cur.fetchone():
                    return

            cur.execute(
                "INSERT INTO knowledge (pergunta, resposta, fonte, categoria, url, relevancia) VALUES (%s, %s, %s, %s, %s, %s)",
                (pergunta[:500], resposta, fonte, categoria, url, relevancia)
            )
        conn.commit()
    finally:
        conn.close()


def knowledge_buscar(pergunta: str, limite: int = 1):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pergunta, resposta, fonte, id FROM knowledge WHERE MATCH(pergunta, resposta) AGAINST(%s IN NATURAL LANGUAGE MODE) LIMIT %s",
                (pergunta, limite)
            )
            resultados = cur.fetchall()
            if resultados:
                cur.execute("UPDATE knowledge SET acessos = acessos + 1 WHERE id = %s", (resultados[0]['id'],))
                conn.commit()
            return resultados
    finally:
        conn.close()


def knowledge_existe(pergunta: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM knowledge WHERE pergunta = %s LIMIT 1",
                (pergunta[:500],)
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
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fonte, COUNT(*) AS total FROM knowledge GROUP BY fonte")
            return {r['fonte']: r['total'] for r in cur.fetchall()}
    finally:
        conn.close()


def conversa_salvar(pergunta: str, resposta: str, fonte: str = 'desconhecido',
                    chat_id: int = None, usuario_id: int = None):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversas (pergunta, resposta, fonte, chat_id, usuario_id) VALUES (%s, %s, %s, %s, %s)",
                (pergunta, resposta, fonte, chat_id, usuario_id)
            )
        conn.commit()
        if chat_id:
            with conn.cursor() as cur:
                cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s", (chat_id,))
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


AUTH_SECRET = 'keilinks_secret_2024'

def _hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode('utf-8')).hexdigest()

def _gerar_token(username: str) -> str:
    h = hashlib.sha256((username + AUTH_SECRET).encode('utf-8')).hexdigest()
    return f"{username}::{h}"


def usuario_criar(username: str, senha: str, nome: str = None) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM usuarios WHERE username = %s", (username,))
            if cur.fetchone():
                return None
            senha_hash = _hash_senha(senha)
            cur.execute(
                "INSERT INTO usuarios (username, senha_hash, nome) VALUES (%s, %s, %s)",
                (username, senha_hash, nome or username)
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, nome FROM usuarios WHERE username = %s", (username,))
            user = cur.fetchone()
        user['token'] = _gerar_token(username)
        return user
    finally:
        conn.close()


def usuario_login(username: str, senha: str) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, nome, senha_hash FROM usuarios WHERE username = %s",
                (username,)
            )
            user = cur.fetchone()
            if not user:
                return None
            if user['senha_hash'] != _hash_senha(senha):
                return None
            del user['senha_hash']
            user['token'] = _gerar_token(username)
            return user
    finally:
        conn.close()


def usuario_por_token(token: str) -> dict | None:
    if not token or '::' not in token:
        return None
    username, assinatura = token.split('::', 1)
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, nome FROM usuarios WHERE username = %s", (username,))
            user = cur.fetchone()
            if user and _gerar_token(user['username']) == token:
                return user
        return None
    finally:
        conn.close()


def chat_criar(usuario_id: int, titulo: str = 'Nova conversa') -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (usuario_id, titulo) VALUES (%s, %s)",
                (usuario_id, titulo)
            )
        conn.commit()
        chat_id = cur.lastrowid
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM chats WHERE id = %s", (chat_id,))
            row = cur.fetchone()
            for k in ('created_at', 'updated_at'):
                if isinstance(row.get(k), datetime):
                    row[k] = row[k].isoformat()
            return row
    finally:
        conn.close()


def chat_listar(usuario_id: int) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, titulo, created_at, updated_at FROM chats WHERE usuario_id = %s ORDER BY updated_at DESC",
                (usuario_id,)
            )
            rows = cur.fetchall()
            for r in rows:
                for k in ('created_at', 'updated_at'):
                    if isinstance(r.get(k), datetime):
                        r[k] = r[k].isoformat()
            return rows
    finally:
        conn.close()


def chat_mensagens(chat_id: int, usuario_id: int) -> list | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM chats WHERE id = %s AND usuario_id = %s", (chat_id, usuario_id))
            if not cur.fetchone():
                return None
            cur.execute(
                "SELECT pergunta, resposta, fonte, created_at FROM conversas WHERE chat_id = %s ORDER BY id ASC",
                (chat_id,)
            )
            rows = cur.fetchall()
            for r in rows:
                if isinstance(r.get('created_at'), datetime):
                    r['created_at'] = r['created_at'].isoformat()
            return rows
    finally:
        conn.close()


def chat_deletar(chat_id: int, usuario_id: int) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM chats WHERE id = %s AND usuario_id = %s", (chat_id, usuario_id))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM conversas WHERE chat_id = %s", (chat_id,))
            cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def chat_atualizar_titulo(chat_id: int, titulo: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE chats SET titulo = %s WHERE id = %s", (titulo[:200], chat_id))
        conn.commit()
    finally:
        conn.close()


def migrar_json_para_mysql(base_dir: str):
    import json
    knowledge_path = os.path.join(base_dir, 'dados', 'knowledge.json')
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            try: fatos = json.load(f)
            except: fatos = []
        if fatos:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    for fato in fatos:
                        cur.execute(
                            "INSERT INTO knowledge (pergunta, resposta, fonte, url) VALUES (%s, %s, %s, %s)",
                            (fato.get('pergunta', '')[:500], fato.get('resposta', ''), fato.get('fonte', 'web')[:50], fato.get('url', None))
                        )
                conn.commit()
            finally: conn.close()

    historico_path = os.path.join(base_dir, 'dados', 'historico.json')
    if os.path.exists(historico_path):
        with open(historico_path, 'r', encoding='utf-8') as f:
            try: historico = json.load(f)
            except: historico = []
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
            finally: conn.close()

    memoria_path = os.path.join(base_dir, 'dados', 'memoria.json')
    if os.path.exists(memoria_path):
        with open(memoria_path, 'r', encoding='utf-8') as f:
            try: dados = json.load(f)
            except: dados = {}
        if dados:
            for chave, valor in dados.items():
                if isinstance(valor, (list, dict)):
                    import json as j
                    memoria_set(chave, j.dumps(valor, ensure_ascii=False))
                else: memoria_set(chave, str(valor))


if __name__ == '__main__':
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inicializar_banco()
    if '--migrar' in sys.argv:
        migrar_json_para_mysql(base)