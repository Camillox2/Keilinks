CREATE DATABASE IF NOT EXISTS keilinks
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE keilinks;

-- Base de conhecimento
CREATE TABLE IF NOT EXISTS knowledge (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pergunta VARCHAR(500) NOT NULL,
    resposta TEXT NOT NULL,
    fonte ENUM('wikipedia','reddit','stackoverflow','devto','hackernews','web','conversa','ensino') DEFAULT 'web',
    categoria ENUM('tech','geral','pessoal','programacao','futebol','ciencia') DEFAULT 'geral',
    url VARCHAR(500) DEFAULT NULL,
    relevancia INT DEFAULT 0,
    acessos INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FULLTEXT idx_busca (pergunta, resposta)
) ENGINE=InnoDB;

-- Historico de conversas
CREATE TABLE IF NOT EXISTS conversas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pergunta TEXT NOT NULL,
    resposta TEXT NOT NULL,
    fonte VARCHAR(50) DEFAULT 'desconhecido',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Log do crawler
CREATE TABLE IF NOT EXISTS crawler_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fonte VARCHAR(50) NOT NULL,
    topico VARCHAR(200) DEFAULT NULL,
    sucesso TINYINT(1) DEFAULT 1,
    fatos_novos INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Memoria de longo prazo
CREATE TABLE IF NOT EXISTS memoria (
    id INT AUTO_INCREMENT PRIMARY KEY,
    chave VARCHAR(100) UNIQUE NOT NULL,
    valor TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;
