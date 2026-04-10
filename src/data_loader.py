import json
import html
import re
from pathlib import Path
from src.config import DATA_DIR, WEBSNS_TEXT_LIMIT


def _clean(text):
    if not text:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _fmt_date(d):
    if not d:
        return ""
    d = str(d)
    if len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def load_records():
    path = DATA_DIR / "records_detail.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for r in raw:
        rd = r.get("recordData", {})
        title = _clean(rd.get("tsnRecordTitle", ""))
        keywords = rd.get("keywordList", "") or ""
        desc = _clean(rd.get("tleContentsText", "") or "")
        folder_type = rd.get("tsiFolderTypeNm", "")
        folder_kind = rd.get("tsiFolderKindNm", "")
        year = rd.get("tnyFolderEndYear", "")
        date = _fmt_date(rd.get("tndRecordEndDate", ""))
        uuid = r.get("uuid", "")

        creators = []
        for c in r.get("recordCreateList", []):
            name = c.get("tsnRecordCreateuser", "")
            org = c.get("tsnRecordCurkeepOrgNm", "")
            if name and name != "미상":
                creators.append(name)
            if org:
                creators.append(org)

        auth_names = [a.get("tsnThsrs", "") for a in r.get("recordAuthList", [])]
        collections = [c.get("tsnCollectionTitle", "") for c in r.get("recordCollectionList", [])]

        embed_text = f"{title}\n키워드: {keywords}\n분류: {folder_type} > {folder_kind}\n생산일자: {date}\n생산자: {', '.join(creators)}\n관련인물: {', '.join(auth_names)}\n{desc}"

        docs.append({
            "id": f"record_{uuid}",
            "type": "record",
            "uuid": uuid,
            "title": title,
            "embed_text": embed_text,
            "metadata": {
                "type": "record",
                "uuid": uuid,
                "title": title,
                "keywords": keywords,
                "folder_type": folder_type,
                "folder_kind": folder_kind,
                "year": str(year) if year else "",
                "date": date,
                "creators": ", ".join(creators),
                "auth_names": ", ".join(auth_names),
                "collections": ", ".join(collections),
            },
            "detail": r,
        })
    return docs


def load_authorities():
    path = DATA_DIR / "authority_detail.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for a in raw:
        ad = a.get("authData", {})
        name = _clean(ad.get("tsnThsrs", ""))
        auth_type = ad.get("tsiAuthTypeNm", "")
        auth_subtype = ad.get("tsiAuthSubtypeNm", "")
        memo = _clean(ad.get("tseAuthMemo", ""))
        alt_name = ad.get("tsnThsrsUf", "") or ""
        uuid = a.get("uuid", "")
        start = _fmt_date(ad.get("tddAuthStart", ""))
        end = _fmt_date(ad.get("tddAuthEnd", ""))

        related_records = [_clean(rec.get("tsnRecordTitle", "")) for rec in a.get("authRecordList", [])]

        embed_text = f"{name} ({auth_type}, {auth_subtype})\n이칭: {alt_name}\n기간: {start} ~ {end}\n{memo}\n관련기록물: {', '.join(related_records)}"

        docs.append({
            "id": f"auth_{uuid}",
            "type": "authority",
            "uuid": uuid,
            "title": name,
            "embed_text": embed_text,
            "metadata": {
                "type": "authority",
                "uuid": uuid,
                "title": name,
                "auth_type": auth_type,
                "auth_subtype": auth_subtype,
                "alt_name": alt_name,
                "start": start,
                "end": end,
            },
            "detail": a,
        })
    return docs


def load_websns():
    path = DATA_DIR / "websns_full.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for w in raw:
        title = _clean(w.get("tsnSnsCollectDataTitle", ""))
        content = _clean(w.get("tseSnsCollectDataCont", ""))
        platform = w.get("tsnSnsPlatformNm", "")
        writer = w.get("tsnSnsCollectDataWriter", "")
        date = _fmt_date(w.get("tddSnsCollectDataWrite", ""))
        tags = w.get("tseSnsCollectDataTag", "") or ""
        uuid = w.get("tsnSnsCollectDataUuid", "")
        url = w.get("tsnCollectDataUrl", "") or ""

        truncated = content[:WEBSNS_TEXT_LIMIT] if content else ""
        embed_text = f"{title}\n{truncated}\n태그: {tags}\n플랫폼: {platform}"

        year = date[:4] if date and len(date) >= 4 else ""

        docs.append({
            "id": f"websns_{uuid}",
            "type": "websns",
            "uuid": uuid,
            "title": title,
            "embed_text": embed_text,
            "metadata": {
                "type": "websns",
                "uuid": uuid,
                "title": title[:200],
                "platform": platform,
                "writer": writer,
                "year": year,
                "date": date,
                "tags": tags[:500],
                "url": url,
            },
            "full_content": content,
        })
    return docs


def load_all():
    records = load_records()
    authorities = load_authorities()
    websns = load_websns()
    return {"records": records, "authorities": authorities, "websns": websns}
