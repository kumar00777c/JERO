import sqlite3
import sys 
from pathlib import Path


class MemoryDatabase:
    """Local SQLite storage for JERO's long-term memories and chat history."""

    def __init__(self, database_path=None):
        if database_path is None:
            if getattr(sys, "frozen", False):
                database_path = Path(sys.executable).resolve().parent / "memory.db"
            else:
                database_path = Path(__file__).resolve().parents[2] / "memory.db"

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.create_table()
        self.create_conversation_tables()

    def create_table(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def create_conversation_tables(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New conversation',
                is_pinned INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """
        )
        # Migrate databases created by the first conversation patch.
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(conversations)")
        }
        if "is_pinned" not in columns:
            self.connection.execute(
                "ALTER TABLE conversations ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    # Existing long-term memory methods — retained unchanged in purpose.
    def add_memory(self, key, value):
        self.connection.execute(
            "INSERT INTO memories (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.connection.commit()

    def get_memory(self, key):
        cursor = self.connection.execute(
            "SELECT value FROM memories WHERE key = ? ORDER BY id DESC LIMIT 1",
            (key,),
        )
        row = cursor.fetchone()
        return None if row is None else row[0]

    def delete_memory(self, key):
        self.connection.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.connection.commit()

    def get_all_memories(self):
        cursor = self.connection.execute(
            "SELECT key, value FROM memories ORDER BY id ASC"
        )
        return cursor.fetchall()

    # Conversation history methods.
    def create_conversation(self, title="New conversation"):
        cursor = self.connection.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        self.connection.commit()
        return cursor.lastrowid

    def list_conversations(self):
        cursor = self.connection.execute(
            """SELECT c.id, c.title, c.updated_at,
                      (SELECT COUNT(*) FROM conversation_messages m
                       WHERE m.conversation_id = c.id) AS message_count,
                      c.is_pinned
               FROM conversations c
               ORDER BY c.is_pinned DESC, c.updated_at DESC, c.id DESC"""
        )
        return cursor.fetchall()

    def get_conversation_messages(self, conversation_id):
        cursor = self.connection.execute(
            """SELECT role, content FROM conversation_messages
               WHERE conversation_id = ? ORDER BY id ASC""",
            (conversation_id,),
        )
        return cursor.fetchall()

    def add_conversation_message(self, conversation_id, role, content):
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be 'user' or 'assistant'.")
        self.connection.execute(
            """INSERT INTO conversation_messages (conversation_id, role, content)
               VALUES (?, ?, ?)""",
            (conversation_id, role, content),
        )
        self.connection.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        self.connection.commit()

    def update_conversation_title(self, conversation_id, title):
        title = (title or "New conversation").strip()[:60]
        self.connection.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, conversation_id),
        )
        self.connection.commit()

    def set_conversation_pinned(self, conversation_id, is_pinned):
        self.connection.execute(
            "UPDATE conversations SET is_pinned = ? WHERE id = ?",
            (1 if is_pinned else 0, conversation_id),
        )
        self.connection.commit()

    def delete_conversation(self, conversation_id):
        # ON DELETE CASCADE removes only messages belonging to this conversation.
        self.connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        self.connection.commit()

    def close(self):
        self.connection.close()
