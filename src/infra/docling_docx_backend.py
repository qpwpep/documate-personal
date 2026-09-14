"""Source-text policy for the pinned, optional Docling Word backend."""

from docling.backend.msword_backend import MsWordDocumentBackend


class SourceTextWordBackend(MsWordDocumentBackend):
    def _get_paragraph_elements(self, paragraph):
        """Keep a Word paragraph whole before Docling strips its formatted runs.

        DocuMate stores visible source text, not inline style or link markup.
        Joining the original run/link text here retains both missing separators
        (e.g. a styled identifier) and literal spaces, tabs and line breaks.
        Docling still owns headings, lists, code, tables and document traversal.
        This private extension is covered by actual DOCX conversion tests and
        must be checked when upgrading the pinned Docling version.
        """
        return [(self._get_paragraph_text(paragraph), None, None)]
