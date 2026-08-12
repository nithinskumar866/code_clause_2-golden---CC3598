"""
Duplicate-resume detection.

A CV indexed twice appears twice in every search result and skews any ranking it takes
part in — the pool already contained the same candidate under two IDs before this
existed. Detection is by CONTENT, because the same CV is routinely re-sent under a
different filename.
"""
import io

import pytest
from fastapi import UploadFile

from app.models.database import Resume
from app.services.resume import (
    DuplicateResumeError,
    compute_content_hash,
    find_duplicate_groups,
    validate_and_save_resume,
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(
        filename=name,
        file=io.BytesIO(content),
        headers={"content-type": DOCX_MIME},
    )


def test_identical_content_hashes_identically():
    assert compute_content_hash(b"same bytes") == compute_content_hash(b"same bytes")
    assert compute_content_hash(b"a") != compute_content_hash(b"b")


def test_same_file_uploaded_twice_is_rejected(db_session):
    body = b"PK\x03\x04 resume of a candidate"
    first = validate_and_save_resume(db_session, _upload("cv.docx", body))
    assert first.content_hash == compute_content_hash(body)

    with pytest.raises(DuplicateResumeError) as excinfo:
        validate_and_save_resume(db_session, _upload("cv.docx", body))
    assert excinfo.value.existing.id == first.id


def test_duplicate_is_caught_under_a_different_filename(db_session):
    """The real-world case: 'resume_final.docx' and 'resume_final_v2.docx'."""
    body = b"PK\x03\x04 identical bytes, different name"
    validate_and_save_resume(db_session, _upload("resume_final.docx", body))
    with pytest.raises(DuplicateResumeError):
        validate_and_save_resume(db_session, _upload("resume_final_v2.docx", body))


def test_different_resumes_are_both_accepted(db_session):
    a = validate_and_save_resume(db_session, _upload("a.docx", b"PK\x03\x04 candidate A"))
    b = validate_and_save_resume(db_session, _upload("b.docx", b"PK\x03\x04 candidate B"))
    assert a.id != b.id
    assert a.content_hash != b.content_hash


def test_duplicate_can_be_kept_deliberately(db_session):
    body = b"PK\x03\x04 keep both copies"
    validate_and_save_resume(db_session, _upload("x.docx", body))
    second = validate_and_save_resume(db_session, _upload("x.docx", body), allow_duplicate=True)
    assert second.id is not None


def test_empty_upload_is_rejected(db_session):
    from app.core.exceptions import UploadError

    with pytest.raises(UploadError):
        validate_and_save_resume(db_session, _upload("empty.docx", b""))


def test_existing_duplicates_are_reported_not_deleted(db_session):
    body = b"PK\x03\x04 duplicated candidate"
    keep = validate_and_save_resume(db_session, _upload("first.docx", body))
    dupe = validate_and_save_resume(db_session, _upload("second.docx", body), allow_duplicate=True)

    groups = find_duplicate_groups(db_session)
    assert len(groups) == 1
    assert groups[0]["count"] == 2
    assert groups[0]["keep"]["id"] == keep.id
    assert [d["id"] for d in groups[0]["duplicates"]] == [dupe.id]
    # Reporting must never remove data — which copy to drop is the operator's call.
    assert db_session.query(Resume).count() == 2


def test_duplicate_error_message_names_the_existing_resume(db_session):
    body = b"PK\x03\x04 tell me where it already is"
    validate_and_save_resume(db_session, _upload("original.docx", body))
    with pytest.raises(DuplicateResumeError) as excinfo:
        validate_and_save_resume(db_session, _upload("copy.docx", body))
    assert "original.docx" in excinfo.value.message
