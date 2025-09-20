"""Gradio user interface for the Naexya Docs AI application.

This module assembles the full interactive experience for the project while
remaining intentionally high-level so future contributors can plug in real
business logic. The interface models the end-to-end workflow for capturing
project requirements, collaborating with AI personas, validating the generated
content, and exporting approved specifications.

Key features implemented below:

* Application initialization that wires together configuration, the SQLite
  database helper, and the AI client abstraction.
* Responsive Gradio ``Blocks`` interface composed of multiple tabs that mirror
  the intended product workflow (projects, conversations, validation,
  specification review, export, and settings).
* Robust state management powered by ``gr.State`` objects so interactions remain
  consistent across user actions and refreshes.
* Extensive inline comments, docstrings, and structured sections to serve as a
  living guide for engineers extending the tool.
* Demo data helpers that allow the UI to be exercised without API keys or
  external dependencies—ideal for automated tests and onboarding sessions.
"""

from __future__ import annotations

import itertools
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gradio as gr

from ai_client import AIClient
from config import AI_PROVIDERS, AppConfig
from database import DatabaseManager, SpecificationRecord
from utils import format_prompt, render_export

# ---------------------------------------------------------------------------
# Application bootstrapping
# ---------------------------------------------------------------------------

# Configure logging early so helpers can emit debug information. In production
# you might route this to structured logs or observability platforms.
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Instantiate configuration, database manager, and AI client when the module is
# imported. This ensures shared state is reused across Gradio requests.
CONFIG: AppConfig = AppConfig.from_environment()
DB_MANAGER = DatabaseManager(database_path=CONFIG.database_path)
AI = AIClient(config=CONFIG)

# Category definitions used throughout validation and reporting flows. The order
# controls how sections are rendered in the Specifications tab.
SPECIFICATION_CATEGORIES: Tuple[str, ...] = (
    "Business Requirements",
    "Functional Specifications",
    "Non-Functional Requirements",
    "Technical Architecture",
    "Validation Criteria",
)

# Create a simple counter so each pending specification has a predictable,
# unique identifier. ``itertools.count`` is lightweight and thread-safe for the
# single-worker environments common when running Gradio locally.
PENDING_ID_SEQUENCE = itertools.count(1)

# Demo specification used when users enable mock data. Keeping the structure in
# a dataclass makes the code self-documenting.
@dataclass
class DemoSpecification:
    """Structure representing mock specifications bundled with the app."""

    title: str
    category: str
    content: str


DEMO_PROJECT_NAME = "Demo Commerce Platform"
DEMO_SPECIFICATIONS: Tuple[DemoSpecification, ...] = (
    DemoSpecification(
        title="Customer Journey Overview",
        category="Business Requirements",
        content=(
            "- Describe online storefront goals.\n"
            "- Identify primary personas (shoppers, support, merchandising).\n"
            "- Highlight success metrics such as conversion rate and AOV."
        ),
    ),
    DemoSpecification(
        title="Checkout Microservice",
        category="Technical Architecture",
        content=(
            "- Python FastAPI service with PostgreSQL persistence.\n"
            "- Integrates with payment gateway via REST webhooks.\n"
            "- Includes observability hooks for latency and error tracking."
        ),
    ),
)


def _prepare_demo_database() -> None:
    """Seed the SQLite database with a small demo record if empty."""

    existing = list(DB_MANAGER.fetch_recent_specifications(limit=1))
    if existing:
        return

    LOGGER.info("Seeding demo specification records")
    for spec in DEMO_SPECIFICATIONS:
        title = f"{spec.category}::{DEMO_PROJECT_NAME}::{spec.title}"
        DB_MANAGER.save_specification(title=title, content=spec.content)


# Ensure the schema exists and optionally seed demo content. The database manager
# already creates tables on initialization; we only add demo data if none exists
# to keep the repository self-contained for new users.
_prepare_demo_database()


# ---------------------------------------------------------------------------
# Helper utilities for stateful interactions
# ---------------------------------------------------------------------------

def _ensure_project_selected(project: Optional[str]) -> None:
    """Raise an informative error when a project has not been chosen."""

    if not project:
        raise ValueError(
            "Please create or select a project on the Projects tab before using this feature."
        )


def _create_pending_entry(
    *,
    project: str,
    persona: str,
    response: str,
    category: str,
) -> Dict[str, str]:
    """Compose a dictionary representing a specification awaiting validation."""

    pending_id = next(PENDING_ID_SEQUENCE)
    title = f"{project} - {persona.title()} Draft #{pending_id}"
    return {
        "id": str(pending_id),
        "project": project,
        "persona": persona,
        "category": category,
        "title": title,
        "content": response,
    }


def _persona_prompt(persona: str, message: str) -> str:
    """Format the user message with persona-specific guidance."""

    persona_guidance = {
        "requirements": (
            "Act as a business analyst capturing stakeholder goals, user personas, and"
            " measurable outcomes."
        ),
        "technical": (
            "Act as a systems architect proposing services, integrations, and deployment"
            " considerations."
        ),
    }
    guidance = persona_guidance.get(persona, "Act as an assistant.")
    return (
        "You are collaborating on Naexya Docs AI. "
        f"{guidance}\n\nUser message:\n{message.strip()}"
    )


def _record_conversation(
    conversation_state: Dict[str, List[Tuple[str, str]]],
    persona: str,
    user_message: str,
    ai_response: str,
) -> Dict[str, List[Tuple[str, str]]]:
    """Append conversation turns and return the mutated state copy."""

    updated_history = {**conversation_state}
    history = list(updated_history.get(persona, []))
    history.append(("user", user_message))
    history.append(("assistant", ai_response))
    updated_history[persona] = history
    return updated_history


def _format_validation_queue(queue: Iterable[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Create friendly labels for pending specifications displayed in dropdowns."""

    labels = []
    for pending in queue:
        label = f"#{pending['id']} · {pending['category']} · {pending['title']}"
        labels.append((label, pending["id"]))
    return labels


def _group_approved_specifications(records: Iterable[SpecificationRecord]) -> Dict[str, List[str]]:
    """Organize approved specs by category for the Specifications tab."""

    grouped: Dict[str, List[str]] = {category: [] for category in SPECIFICATION_CATEGORIES}
    for record in records:
        if "::" in record.title:
            category, project, name = record.title.split("::", 2)
        else:
            category, project, name = "Uncategorized", "Unknown Project", record.title
        summary = f"**{project} — {name}**\n\n{record.content}".strip()
        grouped.setdefault(category, []).append(summary)
    return grouped


# ---------------------------------------------------------------------------
# Gradio callback functions (project management)
# ---------------------------------------------------------------------------

def bootstrap_application() -> Tuple[List[str], gr.Dropdown.update, str, Dict[str, List[Tuple[str, str]]], Dict[str, List[Dict[str, str]]], str]:
    """Return initial state for the interface when the app loads."""

    projects = [DEMO_PROJECT_NAME]
    current_project = DEMO_PROJECT_NAME
    conversation_state = {"requirements": [], "technical": []}
    pending_state = {"queue": []}
    if CONFIG.demo_mode:
        status = (
            "Loaded demo mode. Use the Projects tab to explore with mock data or"
            " add a project once you configure API keys."
        )
    else:
        status = (
            "Ready to collaborate. Create a project or load demo data while"
            " authenticated providers generate live specifications."
        )
    dropdown_update = gr.Dropdown.update(choices=projects, value=current_project)
    return projects, dropdown_update, current_project, conversation_state, pending_state, status


def create_project(
    project_name: str,
    projects: List[str],
    current_project: Optional[str],
) -> Tuple[List[str], gr.Dropdown.update, str, gr.Textbox.update]:
    """Create a new project and update the selection dropdown."""

    if not project_name or not project_name.strip():
        raise ValueError("Project name cannot be empty.")

    normalized_name = project_name.strip()
    if normalized_name in projects:
        raise ValueError(f"Project '{normalized_name}' already exists.")

    updated_projects = projects + [normalized_name]
    dropdown_update = gr.Dropdown.update(choices=updated_projects, value=normalized_name)
    status = f"Created project '{normalized_name}' and set it as active."
    clear_input = gr.Textbox.update(value="")
    return updated_projects, dropdown_update, status, clear_input


def select_project(project_name: str) -> Tuple[str, str]:
    """Handle project selection from the dropdown."""

    if not project_name:
        raise ValueError("Select a project to continue.")
    status = f"Active project switched to '{project_name}'."
    return project_name, status


def load_demo_data(
    projects: List[str],
    conversation_state: Dict[str, List[Tuple[str, str]]],
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[str], Dict[str, List[Tuple[str, str]]], Dict[str, List[Dict[str, str]]], gr.Dropdown.update, str]:
    """Populate application state with mock data for testing."""

    demo_projects = projects if DEMO_PROJECT_NAME in projects else projects + [DEMO_PROJECT_NAME]

    conversation_state = {
        "requirements": [
            ("user", "Outline the business goals for the ecommerce relaunch."),
            (
                "assistant",
                "Generated demo summary covering revenue targets, customer journeys, and KPIs.",
            ),
        ],
        "technical": [
            ("user", "Propose the core services and integrations we need."),
            (
                "assistant",
                "Demo architecture: API gateway, checkout service, event bus, analytics pipeline.",
            ),
        ],
    }

    queue = [
        _create_pending_entry(
            project=DEMO_PROJECT_NAME,
            persona="requirements",
            response="Demo requirements specification awaiting approval.",
            category="Business Requirements",
        ),
        _create_pending_entry(
            project=DEMO_PROJECT_NAME,
            persona="technical",
            response="Demo technical architecture overview pending validation.",
            category="Technical Architecture",
        ),
    ]

    pending_state = {"queue": queue}
    dropdown_update = gr.Dropdown.update(choices=demo_projects, value=DEMO_PROJECT_NAME)
    status = "Demo data loaded. Conversations and pending drafts now contain example content."
    return demo_projects, conversation_state, pending_state, dropdown_update, status


# ---------------------------------------------------------------------------
# Gradio callback functions (AI conversations)
# ---------------------------------------------------------------------------

def _handle_conversation(
    *,
    persona: str,
    message: str,
    project: Optional[str],
    conversation_state: Dict[str, List[Tuple[str, str]]],
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]], Dict[str, List[Dict[str, str]]], str]:
    """Core handler shared by both AI persona chat tabs."""

    _ensure_project_selected(project)
    if not message or not message.strip():
        raise ValueError("Please provide a message for the AI persona.")

    formatted_prompt = format_prompt(_persona_prompt(persona, message))

    try:
        ai_response = AI.generate_specification(
            prompt=formatted_prompt,
            persona=persona,
            user_message=message,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for API failures
        LOGGER.error("AI generation failed: %s", exc)
        LOGGER.debug("Traceback: %s", traceback.format_exc())
        raise RuntimeError("Unable to generate a response. Check provider settings.") from exc

    updated_conversation = _record_conversation(
        conversation_state=conversation_state,
        persona=persona,
        user_message=message,
        ai_response=ai_response,
    )

    category = (
        "Business Requirements"
        if persona == "requirements"
        else "Technical Architecture"
    )
    queue = list(pending_state.get("queue", []))
    queue.append(
        _create_pending_entry(
            project=project,
            persona=persona,
            response=ai_response,
            category=category,
        )
    )
    updated_pending = {"queue": queue}

    status = "Draft added to the validation queue. Review it on the Validation tab."
    return updated_conversation[persona], updated_conversation, updated_pending, status


def handle_requirements_chat(
    message: str,
    project: Optional[str],
    conversation_state: Dict[str, List[Tuple[str, str]]],
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]], Dict[str, List[Dict[str, str]]], str]:
    """Wrapper for the Requirements persona interaction."""

    return _handle_conversation(
        persona="requirements",
        message=message,
        project=project,
        conversation_state=conversation_state,
        pending_state=pending_state,
    )


def handle_technical_chat(
    message: str,
    project: Optional[str],
    conversation_state: Dict[str, List[Tuple[str, str]]],
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]], Dict[str, List[Dict[str, str]]], str]:
    """Wrapper for the Technical persona interaction."""

    return _handle_conversation(
        persona="technical",
        message=message,
        project=project,
        conversation_state=conversation_state,
        pending_state=pending_state,
    )


# ---------------------------------------------------------------------------
# Gradio callback functions (validation and approvals)
# ---------------------------------------------------------------------------

def refresh_pending_specs(pending_state: Dict[str, List[Dict[str, str]]]) -> Tuple[gr.Dropdown.update, str]:
    """Update the pending specification dropdown and display guidance."""

    queue = pending_state.get("queue", [])
    if not queue:
        return gr.Dropdown.update(choices=[], value=None), "No drafts awaiting validation."

    labels = _format_validation_queue(queue)
    first_id = queue[0]["id"]
    return gr.Dropdown.update(choices=labels, value=first_id), "Select a draft to review."


def load_pending_spec(
    spec_id: str,
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[str, str]:
    """Return the specification content for the selected pending draft."""

    queue = pending_state.get("queue", [])
    for pending in queue:
        if pending["id"] == spec_id:
            header = f"### {pending['title']}\n**Category:** {pending['category']}"
            return header, pending["content"]
    raise ValueError("Pending draft not found. Refresh the queue and try again.")


def approve_specification(
    spec_id: str,
    project: Optional[str],
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[Dict[str, List[Dict[str, str]]], str]:
    """Move a pending draft into the approved specifications list."""

    _ensure_project_selected(project)
    queue = list(pending_state.get("queue", []))
    remaining: List[Dict[str, str]] = []
    approved_entry: Optional[Dict[str, str]] = None
    for pending in queue:
        if pending["id"] == spec_id:
            approved_entry = pending
        else:
            remaining.append(pending)

    if approved_entry is None:
        raise ValueError("Unable to locate draft for approval. Refresh and retry.")

    title = f"{approved_entry['category']}::{approved_entry['project']}::{approved_entry['title']}"
    DB_MANAGER.save_specification(title=title, content=approved_entry["content"])

    updated_state = {"queue": remaining}
    status = f"Approved '{approved_entry['title']}'. It is now available on the Specifications tab."
    return updated_state, status


def reject_specification(
    spec_id: str,
    pending_state: Dict[str, List[Dict[str, str]]],
) -> Tuple[Dict[str, List[Dict[str, str]]], str]:
    """Remove a pending draft without saving it to the database."""

    queue = list(pending_state.get("queue", []))
    remaining: List[Dict[str, str]] = []
    removed: Optional[Dict[str, str]] = None
    for pending in queue:
        if pending["id"] == spec_id:
            removed = pending
        else:
            remaining.append(pending)

    if removed is None:
        raise ValueError("Draft not found. Refresh the queue and retry.")

    updated_state = {"queue": remaining}
    status = f"Rejected '{removed['title']}'. It has been removed from the queue."
    return updated_state, status


# ---------------------------------------------------------------------------
# Gradio callback functions (specifications, export, and settings)
# ---------------------------------------------------------------------------

def refresh_specifications_view() -> List[str]:
    """Retrieve approved specifications and format markdown for each category."""

    records = DB_MANAGER.fetch_recent_specifications(limit=200)
    grouped = _group_approved_specifications(records)
    rendered_sections: List[str] = []
    for category in SPECIFICATION_CATEGORIES:
        entries = grouped.get(category, [])
        if entries:
            rendered_sections.append("\n\n---\n\n".join(entries))
        else:
            rendered_sections.append("*No approved specifications yet.*")
    return rendered_sections


def export_specification(
    spec_id: str,
    export_format: str,
) -> Tuple[str, str]:
    """Render the selected specification using the HTML or Markdown template."""

    if not spec_id:
        raise ValueError("Select a specification to export.")

    records = list(DB_MANAGER.fetch_recent_specifications(limit=200))
    selected: Optional[SpecificationRecord] = None
    for record in records:
        if record.id == int(spec_id):
            selected = record
            break

    if selected is None:
        raise ValueError("Select a specification to export.")

    context = {"title": selected.title, "content": selected.content}
    template = "export_html.html" if export_format == "HTML" else "export_markdown.md"
    rendered = render_export(template_name=template, context=context)
    notice = f"Rendered {export_format} export for specification #{selected.id}."
    return rendered, notice


def list_exportable_specs() -> gr.Dropdown.update:
    """Populate the export dropdown with approved specifications."""

    records = DB_MANAGER.fetch_recent_specifications(limit=200)
    options = [(record.title, str(record.id)) for record in records]
    return gr.Dropdown.update(choices=options, value=(options[0][1] if options else None))


def summarize_settings() -> str:
    """Provide a user-friendly summary of configured providers."""

    lines: List[str] = []
    for key, credential in CONFIG.providers.items():
        display = AI_PROVIDERS.get(key, {}).get("display_name", key.title())
        lines.append(
            f"- **{display}:** {'Configured' if credential.api_key else 'Not configured'}"
        )

    if CONFIG.demo_mode:
        lines.append(
            "\nDemo mode is active because no API keys were detected."
            " You can explore the interface with deterministic mock responses."
        )
    else:
        lines.append(
            "\nAt least one provider key is configured. Update `NAEXYA_DEFAULT_PROVIDER`"
            " to control which service is used first."
        )

    if CONFIG.space_id:
        lines.append(
            "Running inside a Hugging Face Space. Persistent data is stored under `/data`."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interface construction
# ---------------------------------------------------------------------------

RESPONSIVE_CSS = """
@media (max-width: 768px) {
  .two-column {flex-direction: column !important;}
}
"""


def build_interface() -> gr.Blocks:
    """Create the Gradio Blocks interface with all workflow tabs."""

    with gr.Blocks(title="Naexya Docs AI", css=RESPONSIVE_CSS) as demo:
        gr.Markdown(
            """
            # Naexya Docs AI
            Collaborate with AI personas to capture, validate, and export rich project specifications.
            Use the tabs below to move sequentially from project setup through final export.
            """
        )

        # Shared state stores the active project, persona chat histories, pending drafts,
        # and the full list of projects available in the dropdown.
        project_list_state = gr.State([DEMO_PROJECT_NAME])
        current_project_state = gr.State(DEMO_PROJECT_NAME)
        conversation_state = gr.State({"requirements": [], "technical": []})
        pending_specs_state = gr.State({"queue": []})

        # ------------------------------------------------------------------
        # Projects tab: manage project lifecycle and demo content
        # ------------------------------------------------------------------
        with gr.TabItem("Projects"):
            gr.Markdown(
                """Use this tab to create new projects, switch context, or load demo data."""
            )
            with gr.Row(elem_classes="two-column"):
                with gr.Column():
                    project_name_input = gr.Textbox(label="New Project Name", placeholder="e.g. Mobile Banking App")
                    create_project_button = gr.Button("Create Project", variant="primary")
                with gr.Column():
                    project_dropdown = gr.Dropdown(label="Active Project", choices=[DEMO_PROJECT_NAME], value=DEMO_PROJECT_NAME)
                    select_project_button = gr.Button("Set Active Project", variant="secondary")
            demo_data_button = gr.Button("Load Demo Data", variant="secondary")
            project_status = gr.Markdown()

        # ------------------------------------------------------------------
        # Requirements Chat tab
        # ------------------------------------------------------------------
        with gr.TabItem("Requirements Chat"):
            gr.Markdown(
                """
                Chat with a business analyst persona to capture stakeholder needs, success metrics,
                and product scope. Each response is added to the validation queue.
                """
            )
            requirements_chat = gr.Chatbot(height=350)
            with gr.Row(elem_classes="two-column"):
                requirements_input = gr.Textbox(label="Message", placeholder="Describe goals, constraints, and personas...", lines=3)
                requirements_submit = gr.Button("Send", variant="primary")
            requirements_status = gr.Markdown()

        # ------------------------------------------------------------------
        # Technical Chat tab
        # ------------------------------------------------------------------
        with gr.TabItem("Technical Chat"):
            gr.Markdown(
                """
                Collaborate with a systems architect persona on integrations, services, and deployment
                considerations. Drafts also flow into the validation queue for review.
                """
            )
            technical_chat = gr.Chatbot(height=350)
            with gr.Row(elem_classes="two-column"):
                technical_input = gr.Textbox(label="Message", placeholder="Ask for architecture proposals, sequencing, or risks...", lines=3)
                technical_submit = gr.Button("Send", variant="primary")
            technical_status = gr.Markdown()

        # ------------------------------------------------------------------
        # Validation tab
        # ------------------------------------------------------------------
        with gr.TabItem("Validation"):
            gr.Markdown("""Review drafts generated by AI personas and approve or reject them.""")
            refresh_pending_button = gr.Button("Refresh Pending Drafts", variant="secondary")
            pending_dropdown = gr.Dropdown(label="Pending Drafts", choices=[], interactive=True)
            pending_header = gr.Markdown()
            pending_content = gr.Markdown()
            with gr.Row():
                approve_button = gr.Button("Approve", variant="primary")
                reject_button = gr.Button("Reject", variant="stop")
            validation_status = gr.Markdown()

        # ------------------------------------------------------------------
        # Specifications tab
        # ------------------------------------------------------------------
        with gr.TabItem("Specifications"):
            gr.Markdown("""Browse approved specifications grouped by category.""")
            refresh_specs_button = gr.Button("Refresh View", variant="secondary")
            category_outputs = []
            for category in SPECIFICATION_CATEGORIES:
                with gr.Accordion(category, open=False):
                    markdown = gr.Markdown("*No approved specifications yet.*")
                    category_outputs.append(markdown)

        # ------------------------------------------------------------------
        # Export tab
        # ------------------------------------------------------------------
        with gr.TabItem("Export"):
            gr.Markdown("""Select an approved specification and render it using the export templates.""")
            export_refresh_button = gr.Button("Refresh Approved List", variant="secondary")
            export_dropdown = gr.Dropdown(label="Approved Specifications", choices=[])
            export_format_radio = gr.Radio(["Markdown", "HTML"], value="Markdown", label="Export Format")
            export_button = gr.Button("Render Export", variant="primary")
            export_preview = gr.Code(label="Export Preview", language="markdown")
            export_status = gr.Markdown()

        # ------------------------------------------------------------------
        # Settings tab
        # ------------------------------------------------------------------
        with gr.TabItem("Settings"):
            gr.Markdown(
                """
                Configure AI providers by supplying API keys in your environment. Use this summary to
                verify which providers are currently active. Demo data remains available even without keys.
                """
            )
            settings_summary = gr.Markdown(summarize_settings())
            gr.Markdown(
                """Refer to `.env.example` for the list of supported providers and required environment variables."""
            )

        # ------------------------------------------------------------------
        # Wiring callbacks to UI interactions
        # ------------------------------------------------------------------

        # Application bootstrap when the interface loads.
        demo.load(
            fn=bootstrap_application,
            inputs=None,
            outputs=[project_list_state, project_dropdown, current_project_state, conversation_state, pending_specs_state, project_status],
        )

        # Project management actions.
        create_project_button.click(
            fn=create_project,
            inputs=[project_name_input, project_list_state, current_project_state],
            outputs=[project_list_state, project_dropdown, project_status, project_name_input],
        )

        select_project_button.click(
            fn=select_project,
            inputs=project_dropdown,
            outputs=[current_project_state, project_status],
        )

        demo_data_button.click(
            fn=load_demo_data,
            inputs=[project_list_state, conversation_state, pending_specs_state],
            outputs=[project_list_state, conversation_state, pending_specs_state, project_dropdown, project_status],
        )

        # Requirements persona interactions.
        requirements_submit.click(
            fn=handle_requirements_chat,
            inputs=[requirements_input, current_project_state, conversation_state, pending_specs_state],
            outputs=[requirements_chat, conversation_state, pending_specs_state, requirements_status],
        )

        # Technical persona interactions.
        technical_submit.click(
            fn=handle_technical_chat,
            inputs=[technical_input, current_project_state, conversation_state, pending_specs_state],
            outputs=[technical_chat, conversation_state, pending_specs_state, technical_status],
        )

        # Validation workflows.
        refresh_pending_button.click(
            fn=refresh_pending_specs,
            inputs=pending_specs_state,
            outputs=[pending_dropdown, validation_status],
        )
        pending_dropdown.change(
            fn=load_pending_spec,
            inputs=[pending_dropdown, pending_specs_state],
            outputs=[pending_header, pending_content],
        )
        approve_button.click(
            fn=approve_specification,
            inputs=[pending_dropdown, current_project_state, pending_specs_state],
            outputs=[pending_specs_state, validation_status],
        )
        reject_button.click(
            fn=reject_specification,
            inputs=[pending_dropdown, pending_specs_state],
            outputs=[pending_specs_state, validation_status],
        )

        # Approved specifications browsing.
        refresh_specs_button.click(
            fn=refresh_specifications_view,
            inputs=None,
            outputs=category_outputs,
        )

        # Export workflow.
        export_refresh_button.click(
            fn=list_exportable_specs,
            inputs=None,
            outputs=export_dropdown,
        )
        export_button.click(
            fn=export_specification,
            inputs=[export_dropdown, export_format_radio],
            outputs=[export_preview, export_status],
        )

    return demo


def main() -> None:
    """Launch the Gradio development server."""

    interface = build_interface()
    interface.launch()


if __name__ == "__main__":
    main()
