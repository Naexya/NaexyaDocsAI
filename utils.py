"""Utility helpers used across the Naexya Docs AI application.

The project pulls together configuration, database persistence, and a Gradio
interface.  This module keeps shared helper functions in one place so they can
be reused by both the UI and background processes.  Each function includes
extensive documentation that explains the intended behaviour, common edge
cases, and recommended extension points.
"""

from __future__ import annotations

import html
import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:  # ``jinja2`` is optional at runtime but recommended for template rendering.
    from jinja2 import Template
except ImportError:  # pragma: no cover - executed only when dependency missing.
    Template = None  # type: ignore[misc]

from config import AI_PROVIDERS, EXPORT_TEMPLATES, SPECIFICATION_TYPES
from database import DATABASE_PATH

# Configure a dedicated logger so user action tracking and validation warnings
# can be filtered or redirected by the application-wide logging configuration.
LOGGER = logging.getLogger(__name__)

# Resolve repository root and template directory.  ``BASE_DIR`` allows us to
# construct absolute paths for template loading even when the application is
# executed from a different working directory (e.g. when packaged as a module).
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Mapping of canonical specification categories to Markdown template placeholders.
MARKDOWN_SECTION_KEYS: Dict[str, str] = {
    "User Stories": "user_stories_section",
    "Features": "features_section",
    "API Endpoints": "api_endpoints_section",
    "Database Design": "database_design_section",
    "System Architecture": "system_architecture_section",
}


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def validate_api_key(provider: str, api_key: str) -> bool:
    """Perform lightweight validation of an API key for a given provider.

    The function does **not** make external HTTP requests.  Instead it checks
    that the provider exists in :mod:`config`, ensures a key was supplied, and
    verifies that it is of a plausible length.  Applications can call this when
    a user enters credentials to provide immediate feedback before attempting a
    full API call.

    Args:
        provider: Provider identifier matching a key in ``AI_PROVIDERS``.
        api_key: Raw string provided by the end user.

    Returns:
        ``True`` when the key passes basic validation.  ``False`` is returned
        when validation fails but the caller prefers to handle messaging
        without exceptions.

    Raises:
        ValueError: If the provider is unknown or the key is blank.
    """

    if provider not in AI_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' is not recognised. Please choose one of: "
            f"{', '.join(sorted(AI_PROVIDERS))}."
        )

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API key must be a non-empty string.")

    cleaned = api_key.strip()
    if len(cleaned) < 8:
        LOGGER.warning(
            "API key for provider '%s' appears unusually short.", provider
        )
        return False

    # Many providers rely on Authorization headers containing ``{api_key}``.
    header_format = AI_PROVIDERS[provider]["headers"].get("Authorization", "")
    if "{api_key}" not in header_format:
        LOGGER.debug(
            "Provider '%s' does not use a standard Authorization header template.",
            provider,
        )

    return True


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def format_conversation_history(
    messages: Sequence[Mapping[str, Any]]
) -> str:
    """Create a readable transcript from stored conversation messages.

    Each message mapping should contain at minimum ``role`` and ``content``
    keys, with ``timestamp`` being optional.  ``format_conversation_history``
    sorts the messages chronologically (when timestamps are available) and
    returns a newline-delimited string ready for display in the Gradio
    interface or export templates.

    Args:
        messages: Iterable of dictionary-like objects representing chat turns.

    Returns:
        A human-friendly string.  When no messages are provided a helpful
        placeholder message is returned instead of an empty string.
    """

    if not messages:
        return "No conversation history available yet."

    def _sort_key(message: Mapping[str, Any]):
        ts = message.get("timestamp")
        if isinstance(ts, datetime):
            return (0, ts)
        if isinstance(ts, str):
            try:
                parsed = datetime.fromisoformat(ts)
                return (0, parsed)
            except ValueError:
                return (1, ts)
        return (2, "")

    sorted_messages = sorted(messages, key=_sort_key)

    formatted_lines = []
    for entry in sorted_messages:
        role = str(entry.get("role", "unknown")).title()
        timestamp = entry.get("timestamp")
        human_time = f" [{timestamp}]" if timestamp else ""
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        formatted_lines.append(f"{role}{human_time}:\n{content.strip()}\n")

    return "\n".join(formatted_lines).strip()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _render_template(path: Path, context: Mapping[str, Any]) -> str:
    """Render a template file using Jinja2 when available.

    This private helper keeps file reading and template rendering consistent for
    both HTML and Markdown exports.  When :mod:`jinja2` is unavailable the
    function gracefully falls back to Python's :py:meth:`str.format` syntax,
    ensuring the application still works albeit without advanced templating
    features like loops or conditionals.
    """

    if not path.exists():
        raise FileNotFoundError(f"Template file '{path}' was not found.")

    template_text = path.read_text(encoding="utf-8")

    if Template is None:
        LOGGER.warning(
            "jinja2 is not installed; falling back to basic placeholder replacement for %s",
            path,
        )
        if "{%" in template_text or "%}" in template_text:
            raise RuntimeError(
                "The export template requires jinja2 for conditional rendering. Install jinja2 to continue."
            )
        rendered = template_text
        for key, value in context.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    return Template(template_text).render(**context)


def format_prompt(prompt: str) -> str:
    """Normalise user prompts before sending them to a provider."""

    if not isinstance(prompt, str):
        raise ValueError("Prompt must be provided as a string.")

    cleaned = sanitize_input(prompt)
    if not cleaned:
        raise ValueError("Prompt cannot be empty after sanitisation.")
    return cleaned


def render_export(template_name: str, context: Mapping[str, Any]) -> str:
    """Load a template from ``templates/`` and render it with ``context``."""

    if not template_name or not isinstance(template_name, str):
        raise ValueError("Template name must be a non-empty string.")

    template_path = Path(template_name)
    if not template_path.is_absolute():
        template_path = TEMPLATES_DIR / template_path

    return _render_template(template_path, context)


def _group_specifications(
    specifications: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Mapping[str, Any]]]:
    """Organise specification rows by their ``spec_type`` value."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for spec in specifications:
        spec_type = str(spec.get("spec_type") or "Uncategorised")
        grouped.setdefault(spec_type, []).append(spec)
    return grouped


def _prepare_html_export_context(
    project_data: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble template context for the HTML export."""

    brand_name = str(project_data.get("brand_name") or "Naexya Docs AI").strip() or "Naexya Docs AI"
    project_name = str(project_data.get("name") or "Untitled Project").strip() or "Untitled Project"
    description_raw = project_data.get("description")
    project_description = (
        html.escape(str(description_raw).strip()) if description_raw else ""
    )
    project_created_at = _format_datetime_for_display(project_data.get("created_at"))
    project_identifier = html.escape(str(project_data.get("id") or "N/A"))

    grouped = _group_specifications(specifications)
    ordered_types: List[str] = list(SPECIFICATION_TYPES)
    for spec_type in grouped.keys():
        if spec_type not in ordered_types:
            ordered_types.append(spec_type)

    counts_by_type: Dict[str, int] = {spec_type: len(grouped.get(spec_type, [])) for spec_type in ordered_types}
    total_specs = sum(counts_by_type.values())

    status_counts: Dict[str, int] = defaultdict(int)
    timestamp_candidates: List[datetime] = []
    for items in grouped.values():
        for spec in items:
            status = str(spec.get("status") or "pending").strip().lower()
            status_counts[status] += 1
            parsed = _parse_datetime(spec.get("created_at"))
            if parsed is not None:
                timestamp_candidates.append(parsed)

    latest_activity = _format_datetime_for_display(max(timestamp_candidates)) if timestamp_candidates else "Not available"

    table_of_contents = _build_table_of_contents(ordered_types, counts_by_type)
    conversation_base_url = str(project_data.get("conversation_base_url") or "").strip()
    sections_html, conversation_ids = _build_specification_sections(
        grouped, ordered_types, conversation_base_url
    )
    statistics_block = _build_statistics_block(total_specs, counts_by_type, status_counts, latest_activity)
    conversation_references = _build_conversation_reference_section(
        conversation_ids, conversation_base_url
    )

    return {
        "brand_name": brand_name,
        "project_name": html.escape(project_name),
        "project_description": project_description,
        "project_created_at": project_created_at,
        "project_identifier": project_identifier,
        "specification_total": total_specs,
        "table_of_contents": table_of_contents,
        "specification_sections": sections_html,
        "statistics_block": statistics_block,
        "conversation_references": conversation_references,
        "latest_activity": latest_activity,
    }


def _build_table_of_contents(spec_types: Sequence[str], counts: Mapping[str, int]) -> str:
    """Create an ordered list linking to each specification section."""

    if not spec_types:
        return (
            "<p class=\"empty-state\">No specification categories are configured. "
            "Update SPECIFICATION_TYPES to populate the table of contents.</p>"
        )

    lines = ["<ol class=\"toc-list\">"]
    for spec_type in spec_types:
        slug = _slugify(spec_type)
        count = counts.get(spec_type, 0)
        lines.append(
            "  <li>"
            f"<a href=\"#{slug}\">"
            f"<span class=\"toc-title\">{html.escape(spec_type)}</span>"
            f"<span class=\"toc-count\">{count}</span>"
            "</a>"
            "</li>"
        )
    lines.append("</ol>")
    return "\n".join(lines)


def _build_specification_sections(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    ordered_types: Sequence[str],
    conversation_base_url: str,
) -> Tuple[str, Set[str]]:
    """Render each specification category into HTML sections."""

    sections: List[str] = []
    conversation_ids: Set[str] = set()

    if not grouped:
        sections.append(
            "<section class=\"spec-section empty\">"
            "<p>No specifications have been captured yet. Approve drafts to populate this report.</p>"
            "</section>"
        )
        return "\n".join(sections), conversation_ids

    for spec_type in ordered_types:
        items = list(grouped.get(spec_type, []))
        slug = _slugify(spec_type)
        sections.append(f"<section id=\"{slug}\" class=\"spec-section\">")
        header_html = (
            "  <header class=\"section-header\">"
            f"<h2>{html.escape(spec_type)}</h2>"
            f"<span class=\"badge\">{len(items)} items</span>"
            "</header>"
        )
        sections.append(header_html)

        if not items:
            sections.append(
                "  <p class=\"empty-state\">No specifications approved for this category yet.</p>"
            )
            sections.append("</section>")
            continue

        for spec in items:
            title = html.escape(str(spec.get("title") or "Untitled").strip() or "Untitled")
            raw_status = str(spec.get("status") or "pending").strip() or "pending"
            status_label = html.escape(raw_status.replace("_", " ").title())
            status_class = _slugify(raw_status)
            created_display = _format_datetime_for_display(spec.get("created_at"))
            conversation_id = spec.get("conversation_id")
            conversation_link = ""
            if conversation_id is not None:
                identifier = str(conversation_id)
                conversation_ids.add(identifier)
                link_href = _build_conversation_link(conversation_base_url, identifier)
                link_text = html.escape(identifier)
                conversation_link = (
                    f'<a class="conversation-link" href="{link_href}">'
                    f'View source conversation #{link_text}</a>'
                )

            body_html = _render_rich_text(str(spec.get("content") or ""))

            sections.append("  <article class=\"spec-card\">")
            sections.append(f"    <h3>{title}</h3>")
            sections.append("    <div class=\"spec-meta\">")
            sections.append(
                f"      <span class=\"status-pill status-{status_class}\">{status_label}</span>"
            )
            if created_display:
                sections.append(
                    f"      <span class=\"timestamp\">Captured: {created_display}</span>"
                )
            if conversation_link:
                sections.append(f"      {conversation_link}")
            sections.append("    </div>")
            sections.append(f"    <div class=\"spec-body\">{body_html}</div>")
            sections.append("  </article>")

        sections.append("</section>")

    return "\n".join(sections), conversation_ids


def _build_statistics_block(
    total_specs: int,
    counts_by_type: Mapping[str, int],
    status_counts: Mapping[str, int],
    latest_activity: str,
) -> str:
    """Summarise key metrics for the exported project."""

    cards: List[str] = ["<div class=\"statistics-grid\">"]
    cards.append(
        "  <div class=\"stat-card\">"
        "<span class=\"stat-label\">Total Specifications</span>"
        f"<span class=\"stat-value\">{total_specs}</span>"
        "</div>"
    )

    for spec_type, count in counts_by_type.items():
        cards.append(
            "  <div class=\"stat-card\">"
            f"<span class=\"stat-label\">{html.escape(spec_type)}</span>"
            f"<span class=\"stat-value\">{count}</span>"
            "</div>"
        )

    if status_counts:
        status_items: List[str] = []
        for status, count in sorted(status_counts.items()):
            status_items.append(
                "<li>"
                f"<span class=\"status-name\">{html.escape(status.replace('_', ' ').title())}</span>"
                f"<span class=\"status-count\">{count}</span>"
                "</li>"
            )
        cards.append(
            "  <div class=\"stat-card span-2\">"
            "<span class=\"stat-label\">By Status</span>"
            f"<ul class=\"status-list\">{''.join(status_items)}</ul>"
            "</div>"
        )

    cards.append(
        "  <div class=\"stat-card span-2\">"
        "<span class=\"stat-label\">Last Updated</span>"
        f"<span class=\"stat-value\">{latest_activity}</span>"
        "</div>"
    )

    cards.append("</div>")
    return "\n".join(cards)


def _build_conversation_reference_section(
    conversation_ids: Set[str], conversation_base_url: str
) -> str:
    """Generate a section listing links back to the originating conversations."""

    header = "<section id=\"conversation-references\" class=\"conversation-section\">"
    header += "<h2>Conversation References</h2>"

    if not conversation_ids:
        return (
            header
            + "<p>No linked conversations were captured for these specifications. Continue collaborating to enrich this section.</p>"
            + "</section>"
        )

    items: List[str] = []
    for identifier in sorted(conversation_ids, key=lambda value: (len(value), value)):
        href = _build_conversation_link(conversation_base_url, identifier)
        items.append(
            f"<li><a href=\"{href}\">Conversation #{html.escape(identifier)}</a></li>"
        )

    return header + "<ul class=\"conversation-list\">" + "".join(items) + "</ul></section>"


def _build_conversation_link(base_url: str, conversation_id: Any) -> str:
    """Return a safe hyperlink for a conversation reference."""

    identifier = str(conversation_id)
    if base_url:
        href = f"{base_url.rstrip('/')}/{identifier}"
    else:
        href = f"#conversation-{identifier}"
    return html.escape(href, quote=True)


def _render_rich_text(content: str) -> str:
    """Convert plain text into minimal HTML while preserving structure."""

    stripped = content.strip()
    if not stripped:
        return "<p><em>No additional details provided.</em></p>"

    escaped = html.escape(stripped)
    paragraphs = [para for para in escaped.split("\n\n") if para]
    if not paragraphs:
        paragraphs = [escaped]

    formatted: List[str] = []
    for paragraph in paragraphs:
        formatted.append("<p>" + paragraph.replace("\n", "<br />") + "</p>")
    return "\n".join(formatted)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Safely parse a datetime from various formats used in the database."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_datetime_for_display(value: Any) -> str:
    """Render a datetime value in a human-friendly string."""

    parsed = _parse_datetime(value)
    if parsed is None:
        if isinstance(value, str) and value.strip():
            return html.escape(value.strip())
        return ""

    display = parsed.astimezone(timezone.utc).strftime("%d %B %Y %H:%M %Z")
    return html.escape(display)


def _slugify(value: str) -> str:
    """Convert arbitrary text into an anchor-friendly slug."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    slug = slug.strip("-")
    return slug or "section"


def _build_markdown_context(
    project_data: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    """Assemble structured context for the Markdown export template."""

    project_name = (
        str(project_data.get("name") or "Untitled Project").strip() or "Untitled Project"
    )
    description_raw = project_data.get("description")
    project_description = (
        str(description_raw).strip() if description_raw and str(description_raw).strip() else "Not provided."
    )
    conversation_base_url = str(project_data.get("conversation_base_url") or "").strip()

    project_identifier = str(project_data.get("id") or "N/A")
    created_dt = _parse_datetime(project_data.get("created_at"))
    project_created_at = (
        created_dt.astimezone(timezone.utc).isoformat() if created_dt else "not_recorded"
    )

    grouped = _group_specifications(specifications)
    ordered_types: List[str] = list(SPECIFICATION_TYPES)
    for spec_type in grouped:
        if spec_type not in ordered_types:
            ordered_types.append(spec_type)

    spec_counts = [
        {"type": spec_type, "count": len(grouped.get(spec_type, []))}
        for spec_type in ordered_types
    ]
    total_specs = sum(entry["count"] for entry in spec_counts)

    status_totals: Dict[str, int] = defaultdict(int)
    conversation_links: List[Dict[str, str]] = []
    seen_conversations: Set[str] = set()
    latest_candidates: List[datetime] = []

    sections: Dict[str, str] = {
        placeholder: f"_No {category} documented yet._"
        for category, placeholder in MARKDOWN_SECTION_KEYS.items()
    }
    additional_section_blocks: List[str] = []

    for spec_type in ordered_types:
        items = list(grouped.get(spec_type, []))

        for spec in items:
            status = str(spec.get("status") or "pending").strip() or "pending"
            status_totals[status] += 1

            created = _parse_datetime(spec.get("created_at"))
            if created is not None:
                latest_candidates.append(created)

            conversation_id = spec.get("conversation_id")
            if conversation_id is not None:
                identifier = str(conversation_id)
                if identifier not in seen_conversations:
                    seen_conversations.add(identifier)
                    if conversation_base_url:
                        link_url = f"{conversation_base_url.rstrip('/')}/{identifier}"
                    else:
                        link_url = f"#conversation-{identifier}"
                    conversation_links.append({"id": identifier, "url": link_url})

        section_text = _format_markdown_section(spec_type, items, conversation_base_url)
        placeholder = MARKDOWN_SECTION_KEYS.get(spec_type)
        if placeholder:
            sections[placeholder] = section_text
        elif section_text:
            additional_section_blocks.append(f"### {spec_type}\n{section_text}")

    status_counts = [
        {"status": status.replace("_", " ").title(), "count": count}
        for status, count in sorted(status_totals.items())
    ]

    conversation_links.sort(key=lambda item: (len(item["id"]), item["id"]))
    latest_activity = (
        max(latest_candidates).astimezone(timezone.utc).isoformat()
        if latest_candidates
        else "not_recorded"
    )

    implementation_notes = str(project_data.get("implementation_notes") or "").strip()
    if not implementation_notes:
        implementation_notes = "_No implementation notes provided yet._"

    additional_sections = "\n\n".join(additional_section_blocks)

    metadata = {
        "project_id": project_identifier,
        "project_created_at": project_created_at,
        "spec_counts": spec_counts,
        "status_counts": status_counts,
        "conversation_links": conversation_links,
        "latest_activity": latest_activity,
    }

    context: Dict[str, Any] = {
        "project_name": project_name,
        "project_description": project_description,
        "generation_date": generated_at,
        "spec_count": total_specs,
        "implementation_notes": implementation_notes,
        "additional_sections": additional_sections,
        "metadata": metadata,
    }
    context.update(sections)
    return context


def _format_markdown_section(
    spec_type: str,
    items: Sequence[Mapping[str, Any]],
    conversation_base_url: str,
) -> str:
    """Render a specification collection as a YAML-like Markdown block."""

    if not items:
        return f"_No {spec_type} documented yet._"

    lines: List[str] = []
    for spec in items:
        lines.extend(_format_markdown_entry(spec, conversation_base_url))
    return "\n".join(lines)


def _format_markdown_entry(
    spec: Mapping[str, Any], conversation_base_url: str
) -> List[str]:
    """Create a structured bullet list representation for a specification."""

    title = str(spec.get("title") or "Untitled").strip() or "Untitled"
    status = str(spec.get("status") or "pending").strip() or "pending"
    status_label = status.replace("_", " ").title()

    entry: List[str] = [f"- title: {json.dumps(title)}", f"  status: {json.dumps(status_label)}"]

    spec_id = spec.get("id")
    if spec_id is not None:
        entry.append(f"  specification_id: {json.dumps(str(spec_id))}")

    conversation_id = spec.get("conversation_id")
    if conversation_id is not None:
        identifier = str(conversation_id)
        entry.append(f"  conversation_id: {json.dumps(identifier)}")
        if conversation_base_url:
            url = f"{conversation_base_url.rstrip('/')}/{identifier}"
        else:
            url = f"#conversation-{identifier}"
        entry.append(f"  conversation_url: {json.dumps(url)}")

    created = _parse_datetime(spec.get("created_at"))
    if created is not None:
        entry.append(f"  captured_at: {json.dumps(created.astimezone(timezone.utc).isoformat())}")

    body = str(spec.get("content") or "").strip()
    if body:
        entry.append("  details: |")
        for line in body.splitlines():
            entry.append(f"    {line}")
    else:
        entry.append("  details: _No additional narrative provided._")

    return entry


def generate_export_html(
    project_data: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
) -> str:
    """Generate an HTML report for a project and its specifications.

    Args:
        project_data: Metadata describing the project (name, description, etc.).
        specifications: Collection of specification records to include.

    Returns:
        Rendered HTML string ready for download.

    Raises:
        ValueError: When the HTML template configuration is missing.
    """

    html_template_meta = EXPORT_TEMPLATES.get("html")
    if not html_template_meta:
        raise ValueError("HTML export template is not configured.")

    template_path = Path(html_template_meta["path"])
    if not template_path.is_absolute():
        template_path = BASE_DIR / template_path

    context = _prepare_html_export_context(project_data, specifications)
    context["generated_at"] = get_current_timestamp()

    return _render_template(template_path, context)


def generate_export_markdown(
    project_data: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
) -> str:
    """Generate a Markdown report mirroring the HTML export."""

    md_template_meta = EXPORT_TEMPLATES.get("markdown")
    if not md_template_meta:
        raise ValueError("Markdown export template is not configured.")

    template_path = Path(md_template_meta["path"])
    if not template_path.is_absolute():
        template_path = BASE_DIR / template_path

    generated_at = get_current_timestamp()
    context = _build_markdown_context(project_data, specifications, generated_at)
    return _render_template(template_path, context)


# ---------------------------------------------------------------------------
# Security and auditing helpers
# ---------------------------------------------------------------------------

def sanitize_input(text: Optional[str]) -> str:
    """Escape potentially dangerous user input before rendering.

    This helper strips leading/trailing whitespace, normalises line endings, and
    escapes HTML-sensitive characters.  It does **not** attempt to remove
    Markdown formatting or SQL injection vectors; those concerns should be
    handled by parameterised queries and additional context-specific checks.
    """

    if text is None:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return html.escape(cleaned)


def log_user_action(action: str, details: Optional[Mapping[str, Any]] = None) -> None:
    """Record high-level user events to aid debugging and auditing.

    Args:
        action: Short description of the operation (e.g. ``"create_project"``).
        details: Optional mapping of additional metadata for structured logs.
    """

    if not action:
        raise ValueError("Action description must be provided for logging.")

    if details is None:
        details = {}

    LOGGER.info("User action: %s | Details: %s", action, details)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def get_current_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def calculate_project_stats(project_id: int) -> Dict[str, Any]:
    """Compute aggregate metrics for a single project.

    The resulting dictionary includes counts of conversations, messages, pending
    specifications, approved specifications, and the timestamp of the most
    recent activity.  These metrics power dashboards or can be surfaced in the
    "Specifications" tab to give users a quick overview of project health.
    """

    if not isinstance(project_id, int) or project_id <= 0:
        raise ValueError("project_id must be a positive integer.")

    stats = {
        "total_conversations": 0,
        "total_messages": 0,
        "pending_specifications": 0,
        "approved_specifications": 0,
        "last_activity": None,
    }

    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) AS count FROM conversations WHERE project_id = ?",
                (project_id,),
            )
            stats["total_conversations"] = cursor.fetchone()["count"]

            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.project_id = ?
                """,
                (project_id,),
            )
            stats["total_messages"] = cursor.fetchone()["count"]

            cursor.execute(
                "SELECT COUNT(*) AS count FROM specifications WHERE project_id = ? AND status = 'pending'",
                (project_id,),
            )
            stats["pending_specifications"] = cursor.fetchone()["count"]

            cursor.execute(
                "SELECT COUNT(*) AS count FROM specifications WHERE project_id = ? AND status = 'approved'",
                (project_id,),
            )
            stats["approved_specifications"] = cursor.fetchone()["count"]

            cursor.execute(
                """
                SELECT MAX(ts) AS last_activity
                FROM (
                    SELECT MAX(created_at) AS ts FROM conversations WHERE project_id = ?
                    UNION ALL
                    SELECT MAX(timestamp) AS ts FROM messages m JOIN conversations c ON c.id = m.conversation_id WHERE c.project_id = ?
                    UNION ALL
                    SELECT MAX(created_at) AS ts FROM specifications WHERE project_id = ?
                )
                """,
                (project_id, project_id, project_id),
            )
            row = cursor.fetchone()
            stats["last_activity"] = row["last_activity"] if row else None
    except sqlite3.DatabaseError as error:
        LOGGER.exception("Failed to calculate stats for project %s: %s", project_id, error)
        raise

    return stats


__all__ = [
    "validate_api_key",
    "format_prompt",
    "format_conversation_history",
    "render_export",
    "generate_export_html",
    "generate_export_markdown",
    "sanitize_input",
    "log_user_action",
    "get_current_timestamp",
    "calculate_project_stats",
]
