"""Database layer for Naexya Docs AI.

This module centralises all SQLite interactions used by the application. By
keeping the SQL logic in one place the rest of the codebase can focus on the
business workflow while delegating persistence concerns here.  Each function is
carefully documented so future contributors understand not only *what* the
function does but *why* the design decisions were made.

The helper functions below follow a handful of guiding principles:

* **Single connection helper** – ``_get_connection`` ensures every call uses
  the same connection configuration and enables ``sqlite3.Row`` mapping for
  ergonomic dictionary-style access.
* **Explicit transactions** – ``with`` blocks are used to guarantee commits and
  to automatically close connections regardless of success or failure.
* **Robust error handling** – problems are logged with contextual information
  before being re-raised, giving the caller an opportunity to surface helpful
  feedback in the UI while still capturing the original stack trace.
* **Comprehensive comments** – inline notes explain the schema, relationships,
  and reasoning so the file doubles as lightweight documentation.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# Resolve the database file relative to this module.  Placing the database in
# the repository root keeps the demo self-contained while allowing advanced
# users to supply a custom path when embedding the library elsewhere.
DATABASE_PATH = Path(__file__).resolve().parent / "naexya_docs_ai.db"

# Configure a module-specific logger so calling code can hook into the
# application's logging setup.  ``getLogger(__name__)`` ensures messages are
# namespaced to ``database`` making them easy to filter.
LOGGER = logging.getLogger(__name__)


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection with row access configured.

    Args:
        db_path: Optional custom database path.  When ``None`` the default
            ``DATABASE_PATH`` constant is used.

    Returns:
        A ``sqlite3.Connection`` instance with ``row_factory`` set to
        ``sqlite3.Row`` so query results behave like dictionaries.
    """

    connection = sqlite3.connect(db_path or DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def init_database(db_path: Optional[Path] = None) -> None:
    """Create all required tables if they do not already exist.

    The application stores projects, conversations, chat messages, and
    extracted specifications.  ``init_database`` is idempotent; running it
    multiple times simply ensures the schema remains available without wiping
    existing data.
    """

    LOGGER.debug("Initialising SQLite schema")
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()

            # ``projects`` table stores the high-level workspace definition
            # containing a human-friendly name and optional description.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ``conversations`` capture separate chat threads for each persona
            # (requirements, technical, etc.) and link back to the owning
            # project.  ``is_locked`` helps us prevent further edits once a
            # conversation has been validated.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    persona_type TEXT NOT NULL,
                    is_locked INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
                """
            )

            # ``messages`` belong to a conversation and capture the actual
            # dialog history.  ``role`` mirrors the familiar OpenAI convention
            # of ``user`` and ``assistant`` to keep the data structure flexible
            # if additional participants are ever introduced.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
                """
            )

            # ``specifications`` house the structured outputs created by the
            # AI personas.  ``status`` tracks whether an item is pending
            # validation or has been approved by a human reviewer.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS specifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    conversation_id INTEGER,
                    spec_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
                """
            )

            # ``approved_specs`` is a lightweight table dedicated to storing
            # validated specification summaries used by the Gradio interface.
            # Keeping a separate table avoids interfering with the richer
            # workflow tables above while providing a simple history for
            # export operations and demo content.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS approved_specs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Database initialisation failed: %s", error)
        raise


# ---------------------------------------------------------------------------
# Project management helpers
# ---------------------------------------------------------------------------

def create_project(name: str, description: str = "", db_path: Optional[Path] = None) -> int:
    """Insert a new project row and return its generated ID."""

    LOGGER.info("Creating project: %s", name)
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description),
            )
            conn.commit()
            project_id = cursor.lastrowid
            LOGGER.debug("Created project %s with id %s", name, project_id)
            return project_id
    except sqlite3.IntegrityError as error:
        # ``IntegrityError`` handles duplicate names and other constraint
        # violations.  Re-raising with context helps the UI provide clear
        # feedback, for example when a user accidentally creates a duplicate
        # project.
        LOGGER.exception("Failed to create project '%s': %s", name, error)
        raise


def get_projects(db_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Return all projects ordered by most recent first."""

    LOGGER.debug("Fetching project list")
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, description, created_at FROM projects ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Failed to fetch projects: %s", error)
        raise


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def create_conversation(
    project_id: int,
    persona_type: str,
    db_path: Optional[Path] = None,
) -> int:
    """Start a new conversation for the supplied project and persona."""

    LOGGER.info("Starting %s conversation for project %s", persona_type, project_id)
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (project_id, persona_type) VALUES (?, ?)",
                (project_id, persona_type),
            )
            conn.commit()
            conversation_id = cursor.lastrowid
            LOGGER.debug("Conversation %s created", conversation_id)
            return conversation_id
    except sqlite3.DatabaseError as error:
        LOGGER.exception(
            "Failed to create conversation for project %s (%s): %s",
            project_id,
            persona_type,
            error,
        )
        raise


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    db_path: Optional[Path] = None,
) -> int:
    """Persist an individual chat message belonging to a conversation."""

    LOGGER.debug("Adding %s message to conversation %s", role, conversation_id)
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
            conn.commit()
            message_id = cursor.lastrowid
            LOGGER.debug("Stored message %s", message_id)
            return message_id
    except sqlite3.DatabaseError as error:
        LOGGER.exception(
            "Failed to add message to conversation %s: %s", conversation_id, error
        )
        raise


def lock_conversation(conversation_id: int, db_path: Optional[Path] = None) -> None:
    """Mark a conversation as locked to prevent further editing."""

    LOGGER.info("Locking conversation %s", conversation_id)
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                "UPDATE conversations SET is_locked = 1 WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Failed to lock conversation %s: %s", conversation_id, error)
        raise


# ---------------------------------------------------------------------------
# Specification helpers
# ---------------------------------------------------------------------------

def create_specification(
    project_id: int,
    conversation_id: Optional[int],
    spec_type: str,
    title: str,
    content: str,
    db_path: Optional[Path] = None,
) -> int:
    """Save a generated specification in ``pending`` status."""

    LOGGER.info("Recording %s specification for project %s", spec_type, project_id)
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO specifications (
                    project_id,
                    conversation_id,
                    spec_type,
                    title,
                    content
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, conversation_id, spec_type, title, content),
            )
            conn.commit()
            specification_id = cursor.lastrowid
            LOGGER.debug("Specification %s stored", specification_id)
            return specification_id
    except sqlite3.DatabaseError as error:
        LOGGER.exception(
            "Failed to create specification for project %s: %s", project_id, error
        )
        raise


def get_pending_specifications(
    project_id: int,
    db_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Return specifications awaiting approval for the given project."""

    LOGGER.debug("Fetching pending specifications for project %s", project_id)
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, spec_type, title, content, created_at
                FROM specifications
                WHERE project_id = ? AND status = 'pending'
                ORDER BY created_at ASC
                """,
                (project_id,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.DatabaseError as error:
        LOGGER.exception(
            "Failed to retrieve pending specifications for project %s: %s",
            project_id,
            error,
        )
        raise


def approve_specification(spec_id: int, db_path: Optional[Path] = None) -> None:
    """Mark a specification as approved."""

    LOGGER.info("Approving specification %s", spec_id)
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                "UPDATE specifications SET status = 'approved' WHERE id = ?",
                (spec_id,),
            )
            conn.commit()
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Failed to approve specification %s: %s", spec_id, error)
        raise


def get_approved_specifications(
    project_id: int,
    spec_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Return approved specifications filtered by project and optional type."""

    LOGGER.debug(
        "Fetching approved specifications for project %s (type=%s)",
        project_id,
        spec_type or "*",
    )
    try:
        with _get_connection(db_path) as conn:
            if spec_type:
                rows = conn.execute(
                    """
                    SELECT id, spec_type, title, content, created_at
                    FROM specifications
                    WHERE project_id = ? AND status = 'approved' AND spec_type = ?
                    ORDER BY created_at DESC
                    """,
                    (project_id, spec_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, spec_type, title, content, created_at
                    FROM specifications
                    WHERE project_id = ? AND status = 'approved'
                    ORDER BY created_at DESC
                    """,
                    (project_id,),
                ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.DatabaseError as error:
        LOGGER.exception(
            "Failed to fetch approved specifications for project %s: %s",
            project_id,
            error,
        )
        raise


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def create_sample_data(db_path: Optional[Path] = None) -> None:
    """Populate the database with a minimal set of demo records.

    This helper is intentionally idempotent – it only inserts data when the
    database is empty.  The goal is to provide a ready-to-explore environment
    for users trying the application without configuring API keys.
    """

    LOGGER.info("Seeding sample data if database is empty")
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM projects")
            count = cursor.fetchone()["count"]
        if count:
            LOGGER.debug("Sample data already present; skipping seed")
            return

        # Create a sample project that the UI can immediately load.
        project_id = create_project(
            "Demo Product",
            "Sample workspace showcasing Naexya Docs AI capabilities",
            db_path=db_path,
        )

        # Start one conversation per persona to demonstrate the workflow.
        requirements_conv = create_conversation(
            project_id, "requirements", db_path=db_path
        )
        technical_conv = create_conversation(
            project_id, "technical", db_path=db_path
        )

        # Seed a few representative chat messages to illustrate history.
        add_message(
            requirements_conv,
            "user",
            "We need a mobile app for ordering office supplies with approval workflows.",
            db_path=db_path,
        )
        add_message(
            requirements_conv,
            "assistant",
            "Understood. I'll outline the business goals and success metrics.",
            db_path=db_path,
        )
        add_message(
            technical_conv,
            "assistant",
            "Suggesting a serverless backend with OAuth authentication and inventory sync.",
            db_path=db_path,
        )

        # Finally, add a mixture of pending and approved specifications so
        # the validation and reporting tabs have realistic content.
        spec_id = create_specification(
            project_id,
            requirements_conv,
            "Business Requirements",
            "Ordering Experience",
            "Employees can browse catalogues, submit carts, and track approvals.",
            db_path=db_path,
        )

        create_specification(
            project_id,
            technical_conv,
            "Technical Architecture",
            "Solution Overview",
            "React Native client with AWS Lambda microservices and DynamoDB storage.",
            db_path=db_path,
        )

        # Approve one specification to show both states in the UI.
        approve_specification(spec_id, db_path=db_path)

        LOGGER.info("Sample data created successfully")
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Failed to create sample data: %s", error)
        raise


# ---------------------------------------------------------------------------
# Lightweight manager used by the Gradio interface
# ---------------------------------------------------------------------------


@dataclass
class SpecificationRecord:
    """Representation of an approved specification stored for exports."""

    id: int
    title: str
    content: str
    created_at: str


class DatabaseManager:
    """Simplified database helper tailored for the Gradio UI flows."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        init_database(self.database_path)

    def save_specification(self, title: str, content: str) -> int:
        """Persist an approved specification for later browsing and export."""

        LOGGER.info("Persisting approved specification: %s", title)
        try:
            with _get_connection(self.database_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO approved_specs (title, content) VALUES (?, ?)",
                    (title, content),
                )
                conn.commit()
                spec_id = int(cursor.lastrowid)
                LOGGER.debug("Approved specification stored with id %s", spec_id)
                return spec_id
        except sqlite3.DatabaseError as error:
            LOGGER.exception("Failed to store approved specification '%s': %s", title, error)
            raise

    def fetch_recent_specifications(self, limit: int = 50) -> List[SpecificationRecord]:
        """Return the most recently stored approved specifications."""

        LOGGER.debug("Fetching up to %s approved specifications", limit)
        try:
            with _get_connection(self.database_path) as conn:
                rows = conn.execute(
                    """
                    SELECT id, title, content, created_at
                    FROM approved_specs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [
                SpecificationRecord(
                    id=int(row["id"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]
        except sqlite3.DatabaseError as error:
            LOGGER.exception("Failed to fetch approved specifications: %s", error)
            raise


# Ensure the schema exists whenever this module is imported.  This keeps the
# rest of the application simple because it can assume the tables are present.
init_database()


__all__ = [
    "DATABASE_PATH",
    "DatabaseManager",
    "SpecificationRecord",
    "init_database",
    "create_project",
    "get_projects",
    "create_conversation",
    "add_message",
    "lock_conversation",
    "create_specification",
    "get_pending_specifications",
    "approve_specification",
    "get_approved_specifications",
    "create_sample_data",
]

