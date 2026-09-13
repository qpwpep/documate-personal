"""Compatibility imports for shared attachment preparation."""

from src.app.uploads import (
    PendingUploadOperation,
    StagedUpload,
    UploadStageResult,
    UploadSyncResult,
    discard_staged_files,
    stage_uploaded_files,
    sync_uploaded_file,
)
