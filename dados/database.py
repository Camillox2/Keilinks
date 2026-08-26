"""Banco local SQLite da Keilinks.

O projeto original dependia de um MySQL local que não fazia parte do
repositório. Isso transformava o primeiro ``import api.servidor`` em falha
quando o serviço, a senha ou a porta não existiam. Esta camada usa somente a
biblioteca padrão do Python, cria a base local sob ``keilinks_data/`` e mantém
o contrato das funções que o servidor e os importadores já usam.

SQLite é adequado para a instância local/single-user: WAL permite leitores
enquanto a conversa é gravada e FTS5 fornece busca lexical. Para uma futura
implantação com várias instâncias, migre explicitamente para PostgreSQL; não
compartilhe este arquivo SQLite por uma pasta de rede.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "keilinks_data" / "keilinks.sqlite3"
_INITIALIZED_PATHS: set[Path] = set()
_INITIALIZE_LOCK = threading.RLock()


def database_path() -> Path:
    """Retorna o caminho da base local, com override explícito para testes."""

    value = os.getenv("KEILINKS_DB_PATH", str(DEFAULT_DB_PATH)).strip()
    return Path(value).expanduser().resolve()


def _connect_raw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        # O banco ainda funciona em rollback journal quando o filesystem não
        # permite WAL. Não escondemos outros erros de consulta/escrita.
        pass
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT NOT NULL,
            resposta TEXT NOT NULL,
            fonte TEXT NOT NULL DEFAULT 'web',
            categoria TEXT NOT NULL DEFAULT 'geral',
            url TEXT,
            relevancia INTEGER NOT NULL DEFAULT 0,
            acessos INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_pergunta ON knowledge(pergunta);
        CREATE INDEX IF NOT EXISTS idx_knowledge_fonte ON knowledge(fonte);

        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            senha_hash TEXT NOT NULL,
            nome TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER NOT NULL,
            titulo TEXT NOT NULL DEFAULT 'Nova conversa',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chats_usuario_atualizado
            ON chats(usuario_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS conversas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT NOT NULL,
            resposta TEXT NOT NULL,
            fonte TEXT NOT NULL DEFAULT 'desconhecido',
            chat_id INTEGER,
            usuario_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE SET NULL,
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversas_chat ON conversas(chat_id, id);

        CREATE TABLE IF NOT EXISTS crawler_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fonte TEXT NOT NULL,
            topico TEXT,
            sucesso INTEGER NOT NULL DEFAULT 1,
            fatos_novos INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS memoria (
            chave TEXT PRIMARY KEY,
            valor TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- A tabela ``memoria`` acima é o contrato legado/global. A memória
        -- usada pelo runtime V4 é separada por usuário e nunca deve misturar
        -- informações entre contas.
        CREATE TABLE IF NOT EXISTS user_memory_settings (
            user_id INTEGER PRIMARY KEY,
            memory_enabled INTEGER NOT NULL DEFAULT 0,
            history_enabled INTEGER NOT NULL DEFAULT 1,
            training_consent INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES usuarios(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS user_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL DEFAULT 'note',
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES usuarios(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_memories_user_updated
            ON user_memories(user_id, updated_at DESC, id DESC);
        """
    )
    try:
        connection.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                pergunta,
                resposta,
                content='knowledge',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, pergunta, resposta)
                VALUES (new.id, new.pergunta, new.resposta);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, pergunta, resposta)
                VALUES ('delete', old.id, old.pergunta, old.resposta);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_au
            AFTER UPDATE OF pergunta, resposta ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, pergunta, resposta)
                VALUES ('delete', old.id, old.pergunta, old.resposta);
                INSERT INTO knowledge_fts(rowid, pergunta, resposta)
                VALUES (new.id, new.pergunta, new.resposta);
            END;
            """
        )
        connection.execute(
            "INSERT INTO knowledge_fts(rowid, pergunta, resposta) "
            "SELECT id, pergunta, resposta FROM knowledge "
            "WHERE id NOT IN (SELECT rowid FROM knowledge_fts)"
        )
    except sqlite3.OperationalError:
        # FTS5 pode não estar compilado no Python de uma distribuição minimal.
        # ``knowledge_buscar`` usa LIKE como fallback seguro.
        pass


def _ensure_initialized() -> Path:
    path = database_path()
    with _INITIALIZE_LOCK:
        if path in _INITIALIZED_PATHS and path.exists():
            return path
        connection = _connect_raw(path)
        try:
            _create_schema(connection)
            connection.commit()
        finally:
            connection.close()
        _INITIALIZED_PATHS.add(path)
    return path


def get_conn() -> sqlite3.Connection:
    """Abre uma conexão SQLite pronta para uso, com linhas indexáveis por nome."""

    return _connect_raw(_ensure_initialized())


@contextmanager
def _connection_scope() -> Any:
    """Abre, confirma/retrocede e fecha a conexão em toda operação local."""

    connection = get_conn()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def inicializar_banco() -> None:
    """Cria o banco local e todas as tabelas caso ainda não existam."""

    print(f"[SQLite] Banco local pronto: {_ensure_initialized()}")


def _iso_timestamp(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "T" not in value and " " in value:
        return value.replace(" ", "T", 1) + ("" if value.endswith("Z") else "Z")
    return value


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in ("created_at", "updated_at"):
        if key in result:
            result[key] = _iso_timestamp(result[key])
    return result


def _knowledge_insert(
    connection: sqlite3.Connection,
    pergunta: str,
    resposta: str,
    fonte: str = "web",
    categoria: str = "geral",
    url: str | None = None,
    relevancia: int = 0,
) -> bool:
    question = str(pergunta or "").strip()[:500]
    answer = str(resposta or "").strip()
    if not question or not answer:
        return False
    if connection.execute(
        "SELECT id FROM knowledge WHERE pergunta = ? LIMIT 1", (question,)
    ).fetchone():
        return False
    answer_start = answer[:200]
    if answer_start and connection.execute(
        "SELECT id FROM knowledge WHERE substr(resposta, 1, 200) = ? LIMIT 1",
        (answer_start,),
    ).fetchone():
        return False
    connection.execute(
        """
        INSERT INTO knowledge (pergunta, resposta, fonte, categoria, url, relevancia)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            answer,
            str(fonte or "web")[:80],
            str(categoria or "geral")[:80],
            str(url)[:2048] if url else None,
            int(relevancia),
        ),
    )
    return True


def knowledge_adicionar(
    pergunta: str,
    resposta: str,
    fonte: str = "web",
    categoria: str = "geral",
    url: str | None = None,
    relevancia: int = 0,
) -> bool:
    with _connection_scope() as connection:
        return _knowledge_insert(
            connection, pergunta, resposta, fonte, categoria, url, relevancia
        )


def _search_terms(question: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÿ_]{2,}", question.lower())[:12]


def _search_knowledge_fts(
    connection: sqlite3.Connection, terms: list[str], limit: int
) -> list[sqlite3.Row]:
    if not terms:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    return connection.execute(
        """
        SELECT knowledge.id, knowledge.pergunta, knowledge.resposta,
               knowledge.fonte, knowledge.url
        FROM knowledge_fts
        JOIN knowledge ON knowledge.id = knowledge_fts.rowid
        WHERE knowledge_fts MATCH ?
        ORDER BY bm25(knowledge_fts), knowledge.relevancia DESC, knowledge.acessos DESC
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()


def _search_knowledge_like(
    connection: sqlite3.Connection, terms: list[str], limit: int
) -> list[sqlite3.Row]:
    if not terms:
        return []
    clauses: list[str] = []
    params: list[str | int] = []
    for term in terms:
        like = f"%{term}%"
        clauses.append("(lower(pergunta) LIKE ? OR lower(resposta) LIKE ?)")
        params.extend((like, like))
    params.append(limit)
    return connection.execute(
        "SELECT id, pergunta, resposta, fonte, url FROM knowledge "
        f"WHERE {' OR '.join(clauses)} "
        "ORDER BY relevancia DESC, acessos DESC, id DESC LIMIT ?",
        params,
    ).fetchall()


def knowledge_buscar(pergunta: str, limite: int = 1) -> list[dict[str, Any]]:
    terms = _search_terms(str(pergunta or ""))
    if not terms:
        return []
    maximum = max(1, min(int(limite), 20))
    with _connection_scope() as connection:
        try:
            rows = _search_knowledge_fts(connection, terms, maximum)
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = _search_knowledge_like(connection, terms, maximum)
        ids = [int(row["id"]) for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f"UPDATE knowledge SET acessos = acessos + 1 WHERE id IN ({placeholders})",
                ids,
            )
        return [_row_dict(row) for row in rows if row is not None]


def knowledge_existe(pergunta: str) -> bool:
    with _connection_scope() as connection:
        return bool(
            connection.execute(
                "SELECT id FROM knowledge WHERE pergunta = ? LIMIT 1",
                (str(pergunta or "").strip()[:500],),
            ).fetchone()
        )


def knowledge_total() -> int:
    with _connection_scope() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])


def knowledge_por_fonte() -> dict[str, int]:
    with _connection_scope() as connection:
        rows = connection.execute(
            "SELECT fonte, COUNT(*) AS total FROM knowledge GROUP BY fonte"
        ).fetchall()
    return {str(row["fonte"]): int(row["total"]) for row in rows}


def conversa_salvar(
    pergunta: str,
    resposta: str,
    fonte: str = "desconhecido",
    chat_id: int | None = None,
    usuario_id: int | None = None,
) -> None:
    with _connection_scope() as connection:
        connection.execute(
            """
            INSERT INTO conversas (pergunta, resposta, fonte, chat_id, usuario_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(pergunta), str(resposta), str(fonte)[:80], chat_id, usuario_id),
        )
        if chat_id is not None:
            connection.execute(
                "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (chat_id,),
            )


def conversa_historico(limite: int = 50) -> list[dict[str, Any]]:
    maximum = max(1, min(int(limite), 500))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT pergunta, resposta, fonte, created_at
            FROM conversas ORDER BY id DESC LIMIT ?
            """,
            (maximum,),
        ).fetchall()
    history = []
    for row in reversed(rows):
        item = _row_dict(row) or {}
        item["data"] = item.pop("created_at", None)
        history.append(item)
    return history


def crawler_log_salvar(
    fonte: str,
    topico: str | None = None,
    sucesso: bool = True,
    fatos_novos: int = 0,
) -> None:
    with _connection_scope() as connection:
        connection.execute(
            "INSERT INTO crawler_log (fonte, topico, sucesso, fatos_novos) VALUES (?, ?, ?, ?)",
            (
                str(fonte)[:80],
                str(topico)[:500] if topico else None,
                int(bool(sucesso)),
                int(fatos_novos),
            ),
        )


def crawler_log_recentes(limite: int = 20) -> list[dict[str, Any]]:
    maximum = max(1, min(int(limite), 200))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT fonte, topico, sucesso, fatos_novos, created_at
            FROM crawler_log ORDER BY id DESC LIMIT ?
            """,
            (maximum,),
        ).fetchall()
    return [_row_dict(row) for row in reversed(rows) if row is not None]


def memoria_get(chave: str, default: Any = None) -> Any:
    with _connection_scope() as connection:
        row = connection.execute(
            "SELECT valor FROM memoria WHERE chave = ?", (str(chave)[:100],)
        ).fetchone()
    return row["valor"] if row else default


def memoria_set(chave: str, valor: str) -> None:
    with _connection_scope() as connection:
        connection.execute(
            """
            INSERT INTO memoria (chave, valor, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chave) DO UPDATE SET
                valor = excluded.valor,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(chave)[:100], str(valor)),
        )


def memoria_todos() -> dict[str, str]:
    with _connection_scope() as connection:
        rows = connection.execute("SELECT chave, valor FROM memoria").fetchall()
    return {str(row["chave"]): str(row["valor"] or "") for row in rows}


def _auth_secret_path() -> Path:
    return database_path().with_suffix(".auth_secret")


def _auth_secret() -> bytes:
    configured = os.getenv("KEILINKS_AUTH_SECRET", "").strip()
    if configured:
        if len(configured) < 32:
            raise RuntimeError("KEILINKS_AUTH_SECRET precisa ter no mínimo 32 caracteres.")
        return configured.encode("utf-8")

    # Uma instalação local sem .env continua funcional. O segredo persiste fora
    # do Git, ao lado do banco, para que tokens sobrevivam ao restart.
    path = _auth_secret_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        value = ""
    if len(value) < 32:
        value = secrets.token_urlsafe(48)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value + "\n", encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return value.encode("utf-8")


def _hash_senha(senha: str) -> str:
    if not senha:
        raise ValueError("senha não pode estar vazia")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(senha.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return (
        "scrypt$16384$8$1$"
        + base64.b64encode(salt).decode("ascii")
        + "$"
        + base64.b64encode(derived).decode("ascii")
    )


def _verificar_senha(senha: str, armazenada: str) -> tuple[bool, bool]:
    if armazenada.startswith("scrypt$"):
        try:
            _, n, r, p, salt_b64, hash_b64 = armazenada.split("$", 6)
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(hash_b64)
            actual = hashlib.scrypt(
                senha.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p)
            )
            return hmac.compare_digest(actual, expected), False
        except (ValueError, TypeError):
            return False, False
    legacy = hashlib.sha256(senha.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, armazenada), True


def _gerar_token(username: str) -> str:
    issued_at = str(int(time.time()))
    payload = f"{username}:{issued_at}"
    signature = hmac.new(_auth_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def _normalizar_username(username: str) -> str:
    value = str(username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", value):
        raise ValueError("username deve ter 3-50 caracteres: letras, números, _, . ou -")
    return value


def usuario_criar(username: str, senha: str, nome: str | None = None) -> dict[str, Any] | None:
    username = _normalizar_username(username)
    with _connection_scope() as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO usuarios (username, senha_hash, nome) VALUES (?, ?, ?)",
                (username, _hash_senha(senha), str(nome or username)[:100]),
            )
        except sqlite3.IntegrityError:
            return None
        row = connection.execute(
            "SELECT id, username, nome FROM usuarios WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    user = _row_dict(row) or {}
    user["token"] = _gerar_token(username)
    return user


def usuario_login(username: str, senha: str) -> dict[str, Any] | None:
    try:
        username = _normalizar_username(username)
    except ValueError:
        return None
    with _connection_scope() as connection:
        row = connection.execute(
            "SELECT id, username, nome, senha_hash FROM usuarios WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        valid, needs_migration = _verificar_senha(senha, str(row["senha_hash"]))
        if not valid:
            return None
        if needs_migration:
            connection.execute(
                "UPDATE usuarios SET senha_hash = ? WHERE id = ?",
                (_hash_senha(senha), int(row["id"])),
            )
    user = {"id": int(row["id"]), "username": str(row["username"]), "nome": row["nome"]}
    user["token"] = _gerar_token(user["username"])
    return user


def usuario_por_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        username, issued_at, signature = token.rsplit(":", 2)
        issued_at_int = int(issued_at)
    except (TypeError, ValueError):
        return None
    ttl_seconds = int(
        os.getenv("KEILINKS_AUTH_TOKEN_TTL_SECONDS", str(7 * 24 * 60 * 60))
    )
    now = int(time.time())
    if (
        not username
        or ttl_seconds <= 0
        or issued_at_int > now + 60
        or now - issued_at_int > ttl_seconds
    ):
        return None
    expected = hmac.new(
        _auth_secret(), f"{username}:{issued_at}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    with _connection_scope() as connection:
        row = connection.execute(
            "SELECT id, username, nome FROM usuarios WHERE username = ?", (username,)
        ).fetchone()
    return _row_dict(row)


def usuario_por_id(user_id: int) -> dict[str, Any] | None:
    with _connection_scope() as connection:
        row = connection.execute(
            "SELECT id, username, nome FROM usuarios WHERE id = ?", (int(user_id),)
        ).fetchone()
    return _row_dict(row)


_MEMORY_CATEGORIES = {"preference", "profile", "project", "note"}
_SENSITIVE_MEMORY_PATTERN = re.compile(
    r"\b(?:senha|password|api[ _-]?key|chave[ _-]?(?:api|privada|secreta)|"
    r"token|cpf|cart[aã]o|cvv|n[uú]mero[ _-]?de[ _-]?conta)\b",
    flags=re.IGNORECASE,
)
_MEMORY_STOP_WORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "eu", "isso", "meu", "minha", "na", "nas", "no",
    "nos", "o", "os", "para", "por", "que", "se", "um", "uma", "você", "voce",
}


def _coerce_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "sim", "on"}:
            return True
        if normalized in {"0", "false", "no", "nao", "não", "off"}:
            return False
    raise ValueError(f"{field} deve ser booleano")


def _ensure_user_memory_settings(connection: sqlite3.Connection, user_id: int) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO user_memory_settings (user_id) VALUES (?)",
        (int(user_id),),
    )


def _memory_settings_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = _row_dict(row) or {}
    return {
        "memory_enabled": bool(item.get("memory_enabled")),
        "history_enabled": bool(item.get("history_enabled")),
        "training_consent": bool(item.get("training_consent")),
        "updated_at": item.get("updated_at"),
    }


def memoria_usuario_config(user_id: int) -> dict[str, Any]:
    """Retorna as escolhas de privacidade de uma única conta."""

    with _connection_scope() as connection:
        _ensure_user_memory_settings(connection, user_id)
        row = connection.execute(
            "SELECT * FROM user_memory_settings WHERE user_id = ?", (int(user_id),)
        ).fetchone()
    if row is None:  # Cobertura defensiva para um banco externo inconsistente.
        raise LookupError("configuração de memória não encontrada")
    return _memory_settings_dict(row)


def memoria_usuario_config_atualizar(
    user_id: int,
    *,
    memory_enabled: Any | None = None,
    history_enabled: Any | None = None,
    training_consent: Any | None = None,
) -> dict[str, Any]:
    """Atualiza somente as escolhas explicitamente fornecidas pela pessoa."""

    values: list[Any] = []
    assignments: list[str] = []
    for column, value in (
        ("memory_enabled", memory_enabled),
        ("history_enabled", history_enabled),
        ("training_consent", training_consent),
    ):
        if value is None:
            continue
        assignments.append(f"{column} = ?")
        values.append(int(_coerce_bool(value, column)))

    with _connection_scope() as connection:
        _ensure_user_memory_settings(connection, user_id)
        if assignments:
            assignments.append("updated_at = CURRENT_TIMESTAMP")
            connection.execute(
                "UPDATE user_memory_settings SET "
                + ", ".join(assignments)
                + " WHERE user_id = ?",
                (*values, int(user_id)),
            )
        row = connection.execute(
            "SELECT * FROM user_memory_settings WHERE user_id = ?", (int(user_id),)
        ).fetchone()
    if row is None:
        raise LookupError("configuração de memória não encontrada")
    return _memory_settings_dict(row)


def _normalize_memory_content(content: Any) -> str:
    value = re.sub(r"\s+", " ", str(content or "")).strip()
    if len(value) < 3:
        raise ValueError("memória deve ter ao menos 3 caracteres")
    if len(value) > 700:
        raise ValueError("memória pode ter no máximo 700 caracteres")
    if _SENSITIVE_MEMORY_PATTERN.search(value):
        raise ValueError(
            "não salve senhas, chaves, documentos, cartões ou outros segredos como memória"
        )
    return value


def _normalize_memory_category(category: Any) -> str:
    value = str(category or "note").strip().lower()
    if value not in _MEMORY_CATEGORIES:
        raise ValueError("categoria de memória inválida")
    return value


def memorias_usuario_listar(user_id: int, limite: int = 100) -> list[dict[str, Any]]:
    maximum = max(1, min(int(limite), 200))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT id, category, content, source, created_at, updated_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (int(user_id), maximum),
        ).fetchall()
    return [_row_dict(row) for row in rows if row is not None]


def memoria_usuario_criar(
    user_id: int,
    content: Any,
    *,
    category: Any = "note",
    source: str = "manual",
) -> dict[str, Any]:
    """Registra uma memória curta, explícita e revisável para uma conta."""

    normalized_content = _normalize_memory_content(content)
    normalized_category = _normalize_memory_category(category)
    with _connection_scope() as connection:
        _ensure_user_memory_settings(connection, user_id)
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM user_memories WHERE user_id = ?", (int(user_id),)
            ).fetchone()[0]
        )
        if total >= 200:
            raise ValueError("limite de 200 memórias por usuário atingido")
        cursor = connection.execute(
            """
            INSERT INTO user_memories (user_id, category, content, source)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), normalized_category, normalized_content, str(source)[:50]),
        )
        row = connection.execute(
            """
            SELECT id, category, content, source, created_at, updated_at
            FROM user_memories WHERE id = ? AND user_id = ?
            """,
            (cursor.lastrowid, int(user_id)),
        ).fetchone()
    return _row_dict(row) or {}


def memoria_usuario_atualizar(
    user_id: int,
    memory_id: int,
    *,
    content: Any | None = None,
    category: Any | None = None,
) -> dict[str, Any] | None:
    assignments: list[str] = []
    values: list[Any] = []
    if content is not None:
        assignments.append("content = ?")
        values.append(_normalize_memory_content(content))
    if category is not None:
        assignments.append("category = ?")
        values.append(_normalize_memory_category(category))
    if not assignments:
        return next(
            (
                item
                for item in memorias_usuario_listar(user_id, 200)
                if item["id"] == int(memory_id)
            ),
            None,
        )

    with _connection_scope() as connection:
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        connection.execute(
            "UPDATE user_memories SET "
            + ", ".join(assignments)
            + " WHERE id = ? AND user_id = ?",
            (*values, int(memory_id), int(user_id)),
        )
        row = connection.execute(
            """
            SELECT id, category, content, source, created_at, updated_at
            FROM user_memories WHERE id = ? AND user_id = ?
            """,
            (int(memory_id), int(user_id)),
        ).fetchone()
    return _row_dict(row)


def memoria_usuario_excluir(user_id: int, memory_id: int) -> bool:
    with _connection_scope() as connection:
        cursor = connection.execute(
            "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
            (int(memory_id), int(user_id)),
        )
    return cursor.rowcount > 0


def usuario_atualizar_nome(user_id: int, nome: Any) -> dict[str, Any] | None:
    normalized = re.sub(r"\s+", " ", str(nome or "")).strip()
    if not normalized:
        raise ValueError("nome não pode ficar vazio")
    if len(normalized) > 100:
        raise ValueError("nome pode ter no máximo 100 caracteres")
    with _connection_scope() as connection:
        connection.execute(
            "UPDATE usuarios SET nome = ? WHERE id = ?", (normalized, int(user_id))
        )
        row = connection.execute(
            "SELECT id, username, nome FROM usuarios WHERE id = ?", (int(user_id),)
        ).fetchone()
    return _row_dict(row)


def _memory_terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", text.lower())
        if word not in _MEMORY_STOP_WORDS
    }


def memoria_usuario_contexto(user_id: int, question: str = "", limite: int = 5) -> str:
    """Monta um contexto pequeno e relevante, sem vazar memórias de outra conta."""

    settings = memoria_usuario_config(user_id)
    if not settings["memory_enabled"]:
        return ""
    profile = usuario_por_id(user_id)
    memories = memorias_usuario_listar(user_id, 100)
    question_terms = _memory_terms(question)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for position, item in enumerate(memories):
        score = len(question_terms & _memory_terms(str(item.get("content", ""))))
        ranked.append((score, -position, item))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected = [item for score, _, item in ranked if score > 0][:limite]
    if len(selected) < limite:
        # Preferências são deliberadamente aplicáveis a quase toda resposta
        # (por exemplo, tom conciso ou idioma). Só elas podem complementar
        # uma busca sem termos coincidentes; notas e dados de projeto ficam de
        # fora até serem de fato relevantes para a pergunta atual.
        selected_ids = {int(item["id"]) for item in selected}
        for _, _, item in ranked:
            if item.get("category") != "preference" or int(item["id"]) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(int(item["id"]))
            if len(selected) >= min(limite, 2):
                break

    parts: list[str] = []
    if profile and profile.get("nome"):
        parts.append(f"nome confirmado: {profile['nome']}")
    labels = {
        "preference": "preferência",
        "profile": "perfil",
        "project": "projeto",
        "note": "nota",
    }
    for item in selected:
        label = labels.get(str(item.get("category")), "nota")
        parts.append(f"{label}: {item['content']}")
    return " | ".join(parts)[:1800]


def conversa_historico_usuario(user_id: int, limite: int = 50) -> list[dict[str, Any]]:
    maximum = max(1, min(int(limite), 500))
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT pergunta, resposta, fonte, created_at, chat_id
            FROM conversas WHERE usuario_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (int(user_id), maximum),
        ).fetchall()
    return [_row_dict(row) for row in reversed(rows) if row is not None]


def chat_criar(usuario_id: int, titulo: str = "Nova conversa") -> dict[str, Any]:
    with _connection_scope() as connection:
        cursor = connection.execute(
            "INSERT INTO chats (usuario_id, titulo) VALUES (?, ?)",
            (int(usuario_id), str(titulo or "Nova conversa")[:200]),
        )
        row = connection.execute("SELECT * FROM chats WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _row_dict(row) or {}


def chat_listar(usuario_id: int) -> list[dict[str, Any]]:
    with _connection_scope() as connection:
        rows = connection.execute(
            """
            SELECT id, titulo, created_at, updated_at FROM chats
            WHERE usuario_id = ? ORDER BY updated_at DESC, id DESC
            """,
            (int(usuario_id),),
        ).fetchall()
    return [_row_dict(row) for row in rows if row is not None]


def chat_mensagens(chat_id: int, usuario_id: int) -> list[dict[str, Any]] | None:
    with _connection_scope() as connection:
        owner = connection.execute(
            "SELECT id FROM chats WHERE id = ? AND usuario_id = ?",
            (int(chat_id), int(usuario_id)),
        ).fetchone()
        if owner is None:
            return None
        rows = connection.execute(
            """
            SELECT pergunta, resposta, fonte, created_at FROM conversas
            WHERE chat_id = ? ORDER BY id ASC
            """,
            (int(chat_id),),
        ).fetchall()
    return [_row_dict(row) for row in rows if row is not None]


def chat_deletar(chat_id: int, usuario_id: int) -> bool:
    with _connection_scope() as connection:
        owner = connection.execute(
            "SELECT id FROM chats WHERE id = ? AND usuario_id = ?",
            (int(chat_id), int(usuario_id)),
        ).fetchone()
        if owner is None:
            return False
        connection.execute("DELETE FROM conversas WHERE chat_id = ?", (int(chat_id),))
        connection.execute("DELETE FROM chats WHERE id = ?", (int(chat_id),))
    return True


def chat_atualizar_titulo(chat_id: int, titulo: str) -> None:
    with _connection_scope() as connection:
        connection.execute(
            "UPDATE chats SET titulo = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (str(titulo or "Nova conversa")[:200], int(chat_id)),
        )


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return fallback


def migrar_json_para_sqlite(base_dir: str | Path) -> dict[str, int]:
    """Importa JSONs legados uma única vez sem sobrescrever dados locais."""

    root = Path(base_dir)
    imported = {"knowledge": 0, "conversas": 0, "memoria": 0}
    facts = _read_json(root / "dados" / "knowledge.json", [])
    history = _read_json(root / "dados" / "historico.json", [])
    memory = _read_json(root / "dados" / "memoria.json", {})
    with _connection_scope() as connection:
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict) and _knowledge_insert(
                    connection,
                    str(fact.get("pergunta", "")),
                    str(fact.get("resposta", "")),
                    str(fact.get("fonte", "web")),
                    str(fact.get("categoria", "geral")),
                    fact.get("url"),
                ):
                    imported["knowledge"] += 1
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("pergunta", "")).strip()
                answer = str(item.get("resposta", "")).strip()
                if question and answer:
                    connection.execute(
                        "INSERT INTO conversas (pergunta, resposta, fonte) VALUES (?, ?, ?)",
                        (question, answer, str(item.get("fonte", "desconhecido"))[:80]),
                    )
                    imported["conversas"] += 1
        if isinstance(memory, dict):
            for key, value in memory.items():
                serialized = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else str(value)
                )
                connection.execute(
                    """
                    INSERT INTO memoria (chave, valor, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(chave) DO UPDATE SET valor = excluded.valor,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (str(key)[:100], serialized),
                )
                imported["memoria"] += 1
    return imported


def migrar_json_para_mysql(base_dir: str | Path) -> dict[str, int]:
    """Alias temporário para scripts antigos; não abre nem exige MySQL."""

    return migrar_json_para_sqlite(base_dir)


if __name__ == "__main__":
    import sys

    inicializar_banco()
    if "--migrar" in sys.argv:
        print(migrar_json_para_sqlite(PROJECT_ROOT))
