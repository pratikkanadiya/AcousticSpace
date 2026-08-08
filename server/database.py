import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "acousticspace.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(result):
    conn = get_connection()

    cursor = conn.execute("""
        INSERT INTO analysis_history
        (
            filename,
            verdict,
            confidence_pct
        )
        VALUES (?, ?, ?)
    """, (
        result["filename"],
        result["verdict"],
        result["confidence_pct"]
    ))

    conn.commit()

    history_id = cursor.lastrowid

    conn.close()

    return history_id


def get_history():
    conn = get_connection()

    rows = conn.execute("""
        SELECT
            id,
            filename,
            verdict,
            confidence_pct,
            timestamp
        FROM analysis_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def delete_history(history_id):
    conn = get_connection()

    cursor = conn.execute(
        "DELETE FROM analysis_history WHERE id = ?",
        (history_id,)
    )

    conn.commit()

    deleted = cursor.rowcount > 0

    conn.close()

    return deleted


def clear_history():
    conn = get_connection()

    conn.execute("DELETE FROM analysis_history")

    conn.commit()
    conn.close()