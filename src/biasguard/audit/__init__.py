"""AI audit export — deterministic, paste-ready LLM context for a backtest.

    from biasguard.audit import build_audit

    audit = build_audit(integrity_report, manifest=manifest, profile=profile,
                        metrics=metrics, monte_carlo=mc_result)
    audit.write("out/")          # audit_report.{md,json} + ai_debug_prompt.txt
    print(audit.to_ai_prompt())  # paste this into Claude / ChatGPT

The export describes and asks — it never proposes a fix. Its job is to hand an
external LLM the maximum-quality, deterministic context for debugging a strategy.
"""

from __future__ import annotations

from biasguard.audit.export import (
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    PROMPT_FILENAME,
    AuditExport,
    build_audit,
)
from biasguard.audit.targets import investigation_targets

__all__ = [
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "PROMPT_FILENAME",
    "AuditExport",
    "build_audit",
    "investigation_targets",
]
