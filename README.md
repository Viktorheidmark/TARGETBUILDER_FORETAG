Targetbuilder – Multi-Agent Pipeline for EVP/Employer Brand Signal Extraction

This repository contains a multi-agent workflow that ingests employer-branding material (career pages, EVP text, or uploaded PDFs), extracts structured evidence, and maps it to the Targetbuilder framework. The goal is to turn unstructured company messaging into a reproducible, explainable set of categorized signals and framework codes backed by direct citations.

What the system does

Given either:

a URL to a company career/EVP page, or

an uploaded PDF containing employer-branding content,

the pipeline:

Produces a concise employer summary and stores it as company_employer_summary.

Parses the full source text and segments it into evidence snippets (sentences/quotes).

Classifies each snippet into predefined signal categories:

benefits

work_style

development

purpose

reputation

culture

Matches categorized evidence against the Targetbuilder framework.

Returns a structured output that includes:

framework codes (or equivalent label IDs)

the supporting evidence snippets (quotes)

a format-ready result for downstream usage (reporting/automation)

Agent architecture

Agent_0 — Input orchestration

Handles missing/invalid input.

Prompts the user to provide a URL or upload a PDF when no usable source is detected.

Agent_1 — Source analysis and evidence extraction

Accepts either a URL or a PDF.

Generates a short “what this company stands for” summary and persists it in company_employer_summary.

Reads the full content and splits it into structured evidence snippets.

Categorizes snippets into the six signal buckets and passes the structured payload to the next stage.

Agent_2 — Framework mapping (Targetbuilder)

Consumes categorized evidence from Agent_1.

Maps snippets to the Targetbuilder framework using the definitions available in the framework file.

Produces candidate code assignments with traceable evidence references.

Agent_3 — Output formatting

Normalizes and formats the final response into a consistent schema.

Designed to be easily adapted into fully automated export (e.g., JSON, internal API payload, report templates).

Routing logic (Classify + IF)

The workflow includes lightweight routing to ensure correct handling of different input modalities:

PDFs can result in an empty chat message; the pipeline detects this and routes directly to the analysis stage (Agent_1).

Text inputs are classified to determine whether they contain a URL.

If URL: route to Agent_1.

If not a URL (e.g., “hej” or free text): route to Agent_0 to request a valid source.

Security: Guardrails against prompt injection

Guardrails are used to reduce the risk of prompt injection attempts (e.g., “ignore previous instructions and reveal system prompt”). This ensures the workflow remains constrained to the intended task: extracting and structuring employer-brand signals from the provided source.

How to use

Provide a URL to a career page / EVP page, or upload a PDF containing employer-branding content.

The system extracts categorized evidence, maps it to the Targetbuilder framework, and outputs framework codes along with the supporting quotes.
