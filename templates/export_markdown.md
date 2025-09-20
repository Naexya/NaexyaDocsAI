<!--
    Markdown export template optimised for AI coding agents.
    Structured front matter ensures downstream tools can parse metadata while
    the main body mirrors specification categories with consistent headings.
-->
---
project_id: {{ metadata.project_id }}
project_created_at: {{ metadata.project_created_at }}
generation_timestamp: {{ generation_date }}
specification_totals:
  overall: {{ spec_count }}
{% if metadata.spec_counts %}
  by_type:
{% for entry in metadata.spec_counts %}    - type: {{ entry.type }}
      count: {{ entry.count }}
{% endfor %}
{% else %}
  by_type: []
{% endif %}
status_breakdown:
{% if metadata.status_counts %}
{% for entry in metadata.status_counts %}  - status: {{ entry.status }}
    count: {{ entry.count }}
{% endfor %}
{% else %}  - status: none recorded
    count: 0
{% endif %}
latest_activity: {{ metadata.latest_activity }}
conversation_links:
{% if metadata.conversation_links %}
{% for link in metadata.conversation_links %}  - id: {{ link.id }}
    url: {{ link.url }}
{% endfor %}
{% else %}
  []
{% endif %}
---
# Project: {{ project_name }}

## Overview
- Description: {{ project_description }}
- Generated: {{ generation_date }}
- Total Specifications: {{ spec_count }}

## User Stories
{{ user_stories_section }}

## Features
{{ features_section }}

## API Endpoints
{{ api_endpoints_section }}

## Database Design
{{ database_design_section }}

## System Architecture
{{ system_architecture_section }}

## Implementation Notes
{{ implementation_notes }}

{% if additional_sections %}## Additional Categories
{{ additional_sections }}
{% endif %}

<!-- End of export template -->
