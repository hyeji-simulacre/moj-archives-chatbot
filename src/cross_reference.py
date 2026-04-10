from src.data_loader import load_records, load_authorities

_record_map = None
_auth_map = None
_record_to_auth = None
_auth_to_record = None


def _build_maps():
    global _record_map, _auth_map, _record_to_auth, _auth_to_record

    records = load_records()
    authorities = load_authorities()

    _record_map = {d["uuid"]: d for d in records}
    _auth_map = {d["uuid"]: d for d in authorities}

    _record_to_auth = {}
    for r in records:
        auth_list = r.get("detail", {}).get("recordAuthList", [])
        if auth_list:
            _record_to_auth[r["uuid"]] = [
                {"uuid": a["tsnAuthUuid"], "name": a.get("tsnThsrs", ""), "type": a.get("tsiAuthTypeNm", "")}
                for a in auth_list
                if a.get("tsnAuthUuid")
            ]

    _auth_to_record = {}
    for a in authorities:
        rec_list = a.get("detail", {}).get("authRecordList", [])
        if rec_list:
            _auth_to_record[a["uuid"]] = [
                {"uuid": rec["tsnRecordUuid"], "title": rec.get("tsnRecordTitle", ""), "type": rec.get("tsiFolderTypeNm", "")}
                for rec in rec_list
                if rec.get("tsnRecordUuid")
            ]


def get_related_authorities(record_uuid: str) -> list[dict]:
    if _record_to_auth is None:
        _build_maps()
    return _record_to_auth.get(record_uuid, [])


def get_related_records(auth_uuid: str) -> list[dict]:
    if _auth_to_record is None:
        _build_maps()
    return _auth_to_record.get(auth_uuid, [])


def get_doc_by_id(doc_id: str) -> dict | None:
    if _record_map is None:
        _build_maps()

    if doc_id.startswith("record_"):
        uuid = doc_id[7:]
        return _record_map.get(uuid)
    elif doc_id.startswith("auth_"):
        uuid = doc_id[5:]
        return _auth_map.get(uuid)
    return None
