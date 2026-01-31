import os
import re
import json
import requests
import feedparser
import pandas as pd

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from pathlib import Path
from dotenv import load_dotenv

# ✅ app/.env 를 파일 위치 기준으로 확실히 로드
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"   # app/.env
load_dotenv(dotenv_path=ENV_PATH, override=False)


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
router = APIRouter()
    
class ReportState(TypedDict, total=False):
    input_prompt: str
    extra_prompt: str
    outline: str
    draft: str
    review: str

    anomalies: list
    data_snapshot: dict

    social_issues: list
    official_docs: list

    reflect_count: int
    max_reflect: int
    next_step: str

    pdf_path: str

    # PDF 표 렌더링용
    summary_rows: list
    top_rows: list

UA = {"User-Agent": "Mozilla/5.0"}

class ReportRequest(BaseModel):
    input_prompt: str
    extra_prompt: str = ""
    reflect_count: int = 0
    max_reflect: int = 2
    pdf_path: str = "output/audit_report.pdf"

class ReportResponse(BaseModel):
    review: str = ""
    draft: str = ""
    pdf_path: str = ""

def mock_fetch_anomalies_only() -> list:
    return [
        {
            "station_id": "S-001",
            "charger_id": "C-001",
            "type": "현장(청결)",
            "level": "medium",
            "issue": "청결 불량 징후 감지",
            "cause_candidates": [],
        },
        {
            "station_id": "S-001",
            "charger_id": "C-002",
            "type": "안전(화재)",
            "level": "high",
            "issue": "화재 위험 신호 감지",
            "cause_candidates": ["국부 과열", "내부 배선 열화"],
        },
        {
            "station_id": "S-001",
            "charger_id": "C-002",
            "type": "설비(고장)",
            "level": "high",
            "issue": "고장 예상(이상 징후)",
            "cause_candidates": ["커넥터 접촉 불량", "통신 모듈 불안정"],
        },

        # -------------------------
        # 추가 예시 1) 화재 + 청결
        # -------------------------
        {
            "station_id": "S-002",
            "charger_id": "C-011",
            "type": "안전(화재)+현장(청결)",
            "level": "high",
            "issue": "충전기 주변 가연성 폐기물 적치 및 내부 과열 경보가 함께 감지됨",
            "cause_candidates": [
                "환기구/팬 흡입부 이물 축적에 따른 방열 성능 저하",
                "케이블 릴/커넥터 주변 오염으로 인한 접점 발열 증가",
                "충전기 주변 가연물·먼지 누적으로 작은 스파크도 화재로 확산 가능",
            ],
        },

        # -------------------------
        # 추가 예시 2) 고장
        # -------------------------
        {
            "station_id": "S-003",
            "charger_id": "C-005",
            "type": "설비(고장)",
            "level": "medium",
            "issue": "충전 세션 중 빈번한 중단 및 통신 타임아웃으로 이용 불편 민원이 증가함",
            "cause_candidates": [
                "전원 품질 저하(순간 전압 강하/서지)로 인한 제어 보드 리셋",
                "LTE/유선 백홀 불안정으로 OCPP 통신 끊김(네트워크 품질 문제)",
                "접지 저항 증가 또는 누설전류 보호장치 오동작으로 세션 강제 종료",
            ],
        },
    ]

def build_snapshot(anomalies: list) -> dict:
    level_dist = {}
    type_dist = {}
    for a in anomalies:
        lvl = (a.get("level") or "medium").strip()
        t = (a.get("type") or "기타").strip()
        level_dist[lvl] = level_dist.get(lvl, 0) + 1
        type_dist[t] = type_dist.get(t, 0) + 1
    return {
        "record_count": len(anomalies),
        "flag_counts": level_dist,
        "type_distribution": type_dist,
    }

# 3개월 간 한국의 주요 이슈 수집

def fetch_social_issues_korea_3m(extra_prompt: str, max_keep: int = 12) -> List[Dict[str, str]]:
    p = (extra_prompt or "").strip()
    if not p:
        return []

    # “수집하지 마/제외/불필요” 같은 부정어가 있으면 수집 안함
    deny = ["수집하지", "수집 금지", "제외", "빼", "하지마", "하지 마", "불필요", "필요없"]
    if any(d in p for d in deny):
        return []

    # extra_prompt에 사회/이슈/뉴스/최근 등 단서가 없으면 이슈 수집 안함
    keywords = ["사회", "이슈", "최근", "뉴스", "기사", "트렌드", "핫이슈", "주요 이슈", "이슈 반영", "이슈 포함"]
    if not any(k in p for k in keywords):
        return []

    queries = [
        "대한민국 주요 이슈 when:3m",
        "대형 사고 when:3m",
        "재난 안전 when:3m",
        "공공기관 감사 결과 when:3m",
        "정부 발표 점검 when:3m",
        "사회적 논란 when:3m",
        "공공시설 안전 점검 when:3m",
        "대형 화재 사고 when:3m",
    ]

    items: List[Dict[str, str]] = []
    for q in queries:
        url = "https://news.google.com/rss/search?q=" + requests.utils.quote(q) + "&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r = requests.get(url, headers=UA, timeout=10)
            feed = feedparser.parse(r.text)
        except Exception:
            continue

        for e in getattr(feed, "entries", []) or []:
            title_raw = (getattr(e, "title", "") or "").strip()
            link = (getattr(e, "link", "") or "").strip()
            pub = (getattr(e, "published", "") or "").strip()

            if not title_raw:
                continue

            title = title_raw
            publisher = ""
            if " - " in title_raw:
                a, b = title_raw.rsplit(" - ", 1)
                title = a.strip()
                publisher = b.strip()

            items.append(
                {"title": title, "publisher": publisher, "published": pub, "link": link, "source_type": "google_news_rss"}
            )

    # 제목 기준 중복 제거
    seen = set()
    uniq = []
    for it in items:
        t = (it.get("title") or "").strip()
        key = re.sub(r"\s+", " ", t).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return uniq[:max_keep]

# 관련 공문서 수집

def fetch_official_docs_ev(max_keep: int = 10) -> List[Dict[str, str]]:
    base = "https://www.law.go.kr"
    queries = [
        "전기자동차 충전",
        "환경친화적 자동차",
        "전기설비 안전",
        "전기안전관리",
        "소방 안전 점검",
        "재난 안전관리",
        "시설물 유지관리",
        "공공시설 안전점검",
    ]

    def extract_from_html(html: str) -> List[Dict[str, str]]:
        hrefs = re.findall(r'href="(/(?:lsInfoP|admRulInfoP)\.do\?[^"]+)"', html)

        seen_href = set()
        cleaned = []
        for h in hrefs:
            if h in seen_href:
                continue
            seen_href.add(h)
            cleaned.append(h)

        out = []
        for h in cleaned:
            # title 속성 우선
            m = re.search(r'href="' + re.escape(h) + r'"[^>]*title="([^"]{2,200})"', html)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()
            else:
                # 없으면 anchor 텍스트 추정
                m2 = re.search(r'href="' + re.escape(h) + r'"[^>]*>(.*?)</a>', html, re.DOTALL)
                raw = (m2.group(1) if m2 else "")
                raw = re.sub(r"<[^>]+>", " ", raw)
                title = re.sub(r"\s+", " ", raw).strip()

            if not title:
                continue

            out.append(
                {
                    "title": title,
                    "link": base + h,
                    "source_type": "law.go.kr" if "lsInfoP.do" in h else "law.go.kr_admin_rule",
                }
            )
        return out

    items: List[Dict[str, str]] = []
    for q in queries:
        # 법령
        try:
            r = requests.get(base + "/lsSc.do", params={"query": q}, headers=UA, timeout=10)
            r.raise_for_status()
            items.extend(extract_from_html(r.text))
        except Exception:
            pass

        # 행정규칙
        try:
            r = requests.get(base + "/admRulSc.do", params={"query": q}, headers=UA, timeout=10)
            r.raise_for_status()
            items.extend(extract_from_html(r.text))
        except Exception:
            pass

    # 제목 기준 중복 제거
    seen = set()
    uniq = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(it)

    return uniq[:max_keep]

def load_data(state: ReportState) -> ReportState:
    state.setdefault("source_status", {})
    state.setdefault("source_errors", [])

    # 1) 이상 징후 데이터(목업)
    raw_anomalies = mock_fetch_anomalies_only()

    # 2) ✅ type 분리 확장
    # - "안전(화재)+현장(청결)" 같은 케이스는
    #   동일 station/charger/level/issue를 유지한 채
    #   type만 나눠서 2개의 레코드로 만든다.
    anomalies = []
    for a in raw_anomalies:
        t = (a.get("type") or "").strip()
        if "+" in t:
            parts = [x.strip() for x in t.split("+") if x.strip()]
            if parts:
                for part in parts:
                    a2 = dict(a)
                    a2["type"] = part
                    anomalies.append(a2)
            else:
                anomalies.append(a)
        else:
            anomalies.append(a)

    state["anomalies"] = anomalies

    # 3) 집계 스냅샷 (분리된 anomalies 기준으로 집계)
    level_dist = {"high": 0, "medium": 0, "low": 0}
    type_dist = {}

    for a in anomalies:
        lv = (a.get("level") or "").strip().lower()
        if lv in level_dist:
            level_dist[lv] += 1

        t = (a.get("type") or "").strip()
        if t:
            type_dist[t] = type_dist.get(t, 0) + 1

    state["data_snapshot"] = {
        "record_count": len(anomalies),
        "flag_counts": level_dist,
        "type_distribution": type_dist,
    }

    # 4) 사회 이슈(조건부) - 기존 유지
    try:
        extra_prompt = (state.get("extra_prompt") or "").strip()
        social = fetch_social_issues_korea_3m(extra_prompt, max_keep=12)
        state["social_issues"] = social
        state["source_status"]["social_issues_count"] = len(social)
    except Exception as e:
        state["social_issues"] = []
        state["source_errors"].append(f"social_issues_fetch_failed: {str(e)}")
        state["source_status"]["social_issues_count"] = 0

    # 5) 공식 문서 후보(공문/법령 제목 리스트) - 기존 유지
    try:
        official = fetch_official_docs_ev(max_keep=10)
        state["official_docs"] = official
        state["source_status"]["official_docs_count"] = len(official)
    except Exception as e:
        state["official_docs"] = []
        state["source_errors"].append(f"official_docs_fetch_failed: {str(e)}")
        state["source_status"]["official_docs_count"] = 0

    return state

def build_outline(state: ReportState) -> ReportState:
    outline = {
        "sections": [
            "0. 감사실시 개요",
            "1. 종합 요약",
            "2. 공문/법령 취지 기반 예상 지적사항 및 미비점",
            "3. 우선순위 실행항목",
            "4. 최근 3개월 대한민국 주요 사회 이슈 요약 및 감사 연계",
            "부록. 이상 징후 표(PDF에만 포함)"
        ]
    }
    state["outline"] = json.dumps(outline, ensure_ascii=False)
    return state

def write_report(state: ReportState) -> ReportState:
    input_prompt = (state.get("input_prompt") or "").strip()
    extra_prompt = (state.get("extra_prompt") or "").strip()

    anomalies = state.get("anomalies") or []
    snapshot = state.get("data_snapshot") or {}
    social = state.get("social_issues") or []
    official = state.get("official_docs") or []


    # PDF 요약표
    type_dist = {} 
    for a in anomalies:
        t = (a.get("type") or "").strip()
        if t:
            type_dist[t] = type_dist.get(t, 0) + 1

    summary_rows = []
    for k, v in type_dist.items():
        summary_rows.append([str(k), str(v), ""])

    if not summary_rows:
        summary_rows = [["해당 없음", "0", ""]]

    state["summary_rows"] = summary_rows

    # PDF 상세표
    official_titles = []
    for x in official:
        official_titles.append(
            {
                "title": x.get("title", ""),
                "source_type": x.get("source_type", ""),
            }
        )

    # -------------------------
    # (C) PDF 표용 상세 테이블 데이터
    # - 근거 후보 문서: 제목 매칭(간단)
    # - 증빙 항목: LLM(에이전트)이 anomaly를 보고 생성
    # -------------------------
    top_rows = []

    # 증빙 생성용 시스템 프롬프트(짧고 엄격하게)
    evidence_system_prompt = """
너는 ‘종합감사 대비 사전점검’에서 각 문제항목에 대해 "감사 대응 시 확보해야 할 증빙(자료/기록)"을 도출하는 담당자다.

출력 규칙:
- 반드시 한국어
- 4~8개 항목만
- 목록 기호(-, *, •) 금지
- 마크다운(#, **, ``` , |) 금지
- URL/링크 금지
- 한 줄에 "항목1; 항목2; 항목3 ..." 형태로 세미콜론(;)으로만 구분
- 과도한 단정/원인 확정 금지(증빙은 일반적으로 요구되는 기록/문서 중심)
""".strip()

    for i, a in enumerate(anomalies, start=1):
        a_type = (a.get("type") or "")
        a_level = (a.get("level") or "")
        a_issue = (a.get("issue") or "")
        causes = a.get("cause_candidates") or []
        causes_text = ", ".join(causes) if causes else "-"

        text = f"{a_type} {a_issue}"

        # 1) 근거 후보 문서(제목) 간단 매칭
        kw_groups = []
        if ("화재" in text) or ("안전" in text) or ("과열" in text) or ("그을림" in text):
            kw_groups += ["소방", "화재", "안전", "전기안전", "재난", "시설", "점검", "관리", "과열"]
        if ("고장" in text) or ("설비" in text) or ("통신" in text) or ("중단" in text) or ("장애" in text):
            kw_groups += ["점검", "유지", "보수", "설비", "전기설비", "운영", "장애", "기록", "통신"]
        if ("청결" in text) or ("쓰레기" in text) or ("오염" in text) or ("벌레" in text) or ("이물" in text):
            kw_groups += ["청결", "위생", "환경", "폐기물", "청소", "시설", "관리"]

        scored = []
        for doc in official:
            title = (doc.get("title") or "")
            score = 0
            for kw in kw_groups:
                if kw and (kw in title):
                    score += 1
            if score > 0:
                scored.append((score, title))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sources = [t for _, t in scored[:3]]
        sources_text = ", ".join(top_sources) if top_sources else "-"

        # 2) ✅ 증빙 항목: 에이전트(LLM)로 생성
        #    - official_titles(제목만) + anomaly 정보를 함께 주고
        #      “감사 대응 시 확보할 증빙”을 뽑게 함
        evidence_human_prompt = f"""
[문제 항목]
구분: {a_type}
위험도: {a_level}
관측된 현상: {a_issue}
원인 후보(단정 금지): {causes_text}

[근거 후보 문서(제목만)]
{json.dumps(official_titles, ensure_ascii=False)}

[요청]
위 문제 항목을 감사 관점에서 확인하기 위해 일반적으로 요구될 수 있는 증빙(기록/문서/로그/사진/점검표/계약/SLA/조치이력 등)을 4~8개 도출하라.
출력은 반드시 한 줄, 세미콜론으로만 구분하라.
""".strip()

        try:
            ev_msg = llm.invoke(
                [
                    SystemMessage(content=evidence_system_prompt),
                    HumanMessage(content=evidence_human_prompt),
                ]
            )
            evidence_text = (ev_msg.content or "").strip()
        except Exception:
            evidence_text = "점검기록; 유지보수 이력; 장애/통신 로그; 사진 증빙; 조치 결과보고"

        # 후처리: 금지 문자/형식 정리(혹시라도 튀는 경우 방어)
        evidence_text = re.sub(r"https?://\S+", "", evidence_text)
        evidence_text = evidence_text.replace("```", "").replace("|", "").replace("**", "")
        evidence_text = evidence_text.replace("\n", " ").strip()

        # 너무 길면 줄이기(표 가독성)
        if len(evidence_text) > 240:
            evidence_text = evidence_text[:240].rstrip()

        top_rows.append(
            [
                str(i),
                a.get("station_id", ""),
                a.get("charger_id", ""),
                a_type,
                a_level,
                a_issue,
                causes_text,
                sources_text,    # 근거 후보 문서
                evidence_text,   # ✅ 에이전트가 생성한 증빙 항목
            ]
        )

    state["top_rows"] = top_rows

    # -------------------------
    # (D) 보고서 본문 생성(기존 로직 유지)
    # -------------------------
    social_titles = []
    for x in social:
        social_titles.append({"title": x.get("title", ""), "published": x.get("published", "")})

    system_prompt = """
너는 ‘전기차 충전소 관리자 종합감사 대비 사전점검 보고서’ 작성자다.

최우선 규칙
1) 사용자의 추가 프롬프트가 항상 최우선이다. 충돌 시 추가 프롬프트를 따른다.
2) 이상 데이터는 징후/단서이며, 지적사항은 공문/법령/행정규칙의 취지 관점에서 구체적으로 확장한다.
3) 원인 후보는 단정하지 말고 가능성으로만 표현한다.

출력 금지
1) 마크다운 문법(#, **, |, ```), 하이픈/별표 목록, 본문 URL/링크/출처 나열
2) 텍스트 표(문자 기반 표) 및 표를 본문에 재현하는 행위

출력 형식
1. 종합 요약:
3~6문장.

2. 공문/법령 취지 기반 예상 지적사항 및 미비점:
(1)~(8) 번호를 붙여 작성. 각 항목 3~6문장.
‘현상 → 왜 지적되는지(취지) → 확인해야 할 증빙(개념적으로) → 즉시조치 → 재발방지’ 순서 유지.

3. 우선순위 실행항목:
(1)~(5), 각 1~2문장.

4. 최근 3개월 대한민국 주요 사회 이슈 요약 및 감사 연계:
사회 이슈가 없으면 생략. (5~8줄)

주의: 본문에는 표를 넣지 마라. 표는 PDF 부록으로만 제공된다.
""".strip()

    human_prompt = f"""
[사용자 요청]
{input_prompt}

[추가 프롬프트]
{extra_prompt}

[이상 데이터]
{json.dumps(anomalies, ensure_ascii=False)}

[집계 스냅샷]
{json.dumps(snapshot, ensure_ascii=False)}

[공식문서 후보(제목만)]
{json.dumps(official_titles, ensure_ascii=False)}

[사회이슈 후보(제목/시점만)]
{json.dumps(social_titles, ensure_ascii=False)}
""".strip()

    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    draft = (msg.content or "").strip()

    draft = re.sub(r"https?://\S+", "", draft)
    
    # ✅ 화살표/유사 기호 제거(강제)
    draft = draft.replace("→", " ")
    draft = draft.replace("⇒", " ")
    draft = draft.replace("→", " ")
    draft = draft.replace("->", " ")
    draft = draft.replace("=>", " ")

    # (혹시 유니코드 화살표류가 더 섞이면 같이 제거)
    for sym in ["➜", "➔", "➡", "⟶", "⟹", "⟶", "⟵", "⟶"]:
        draft = draft.replace(sym, " ")

    # 기존 방어
    draft = draft.replace("```", "").replace("|", "").replace("**", "")

    # 공백 정리
    draft = re.sub(r"[ \t]{2,}", " ", draft)
    draft = re.sub(r"\n{3,}", "\n\n", draft)

    state["draft"] = draft.strip()

    return state

def review_report(state: ReportState) -> ReportState:
    draft = (state.get("draft") or "").strip()
    anomalies = state.get("anomalies") or []
    official_docs = state.get("official_docs") or []
    social_issues = state.get("social_issues") or []

    reflect_count = int(state.get("reflect_count") or 0)
    max_reflect = int(state.get("max_reflect") or 0)

    problems = []

    if len(draft) > 2200:
        problems.append("length_over")
    if re.search(r"https?://", draft):
        problems.append("contains_url")
    if any(x in draft for x in ["```", "|", "**"]):
        problems.append("markdown_fragments")
    if re.search(r"(?m)^\s*[-*]\s+", draft):
        problems.append("bullet_list")
    if re.search(r"(?m)^\s*[A-Z]\.\s*", draft):
        problems.append("alpha_heading")

    for a in anomalies:
        iss = (a.get("issue") or "").strip()
        if iss and (iss not in draft):
            problems.append("missing_anomaly_issue")
            break

    official_titles = []
    for d in official_docs:
        title = (d.get("title") or "").strip()
        if title:
            official_titles.append(title)
    official_titles = official_titles[:12]

    if official_titles:
        hit = 0
        used = set()
        for t in official_titles:
            if t in draft and t not in used:
                used.add(t)
                hit += 1
        if hit < min(3, len(official_titles)):
            problems.append("official_docs_not_reflected")

    social_titles = []
    for s in social_issues:
        tt = (s.get("title") or "").strip()
        if tt:
            social_titles.append(tt)
    social_titles = social_titles[:12]

    if social_titles:
        need = min(6, len(social_titles))
        hit = 0
        used = set()
        for t in social_titles:
            if t in draft and t not in used:
                used.add(t)
                hit += 1
        if hit < need:
            problems.append("social_issues_not_reflected")

    if problems and reflect_count < max_reflect:
        state["reflect_count"] = reflect_count + 1
        state["next_step"] = "write_report"

        ip = (state.get("input_prompt") or "").strip()
        ip += "\n\n재작성 지시: 형식(1~4, 2번은 (1)~(8))을 유지하고, 제공된 사례(충전소/충전기/이슈)를 빠짐없이 반영하며, 공식 문서 제목과 사회 이슈 제목을 규칙대로 본문에 포함하라. 마크다운/링크/목록/영문머릿말 금지."
        state["input_prompt"] = ip

        state["review"] = "REWRITE"
        state["review_problems"] = problems
        return state

    state["next_step"] = "generate_pdf"
    state["review"] = "OK" if not problems else "OK_WITH_WARN"
    state["review_problems"] = problems
    return state


def should_reflect(state: ReportState) -> str:
    return state.get("next_step", "generate_pdf")

def generate_pdf(state: ReportState) -> ReportState:

    pdf_path = (state.get("pdf_path") or "audit_report.pdf").strip()

    folder = os.path.dirname(pdf_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    font_name = "Helvetica"
    try:
        malgun = r"C:\Windows\Fonts\malgun.ttf"
        if os.path.exists(malgun):
            pdfmetrics.registerFont(TTFont("Malgun", malgun))
            font_name = "Malgun"
    except Exception:
        font_name = "Helvetica"

    styles = getSampleStyleSheet()

    base = ParagraphStyle(
        "base",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=14,
        leading=18,
        spaceAfter=10,
    )

    # ✅ 표 셀용(작게 + 줄바꿈 안정화)
    cell = ParagraphStyle(
        "cell",
        parent=base,
        fontName=font_name,
        fontSize=8.5,
        leading=10.5,
        spaceAfter=0,
    )

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    story.append(Paragraph("전기차 충전소 관리자 종합감사 대비 사전점검 보고서", title_style))
    story.append(Spacer(1, 6))

    draft = (state.get("draft") or "").strip()
    for line in draft.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(line, base))

    story.append(Spacer(1, 10))
    story.append(Paragraph("이상 징후 요약", ParagraphStyle("h", parent=base, fontSize=11, leading=15, spaceAfter=6)))

    summary_rows = state.get("summary_rows") or []
    summary_table = Table([["구분", "건수", "비고"]] + summary_rows, colWidths=[90, 60, 330])
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONT", (0, 0), (-1, -1), font_name),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("주요 이상 징후 상세", ParagraphStyle("h2", parent=base, fontSize=11, leading=15, spaceAfter=6)))

    # ✅ Paragraph로 감싸서 자동 줄바꿈
    top_rows = state.get("top_rows") or []
    header = ["충전소", "충전기", "등급", "구분", "이슈", "원인 후보", "필요 증빙"]

    wrapped = []
    for r in top_rows:
        wrapped.append(
            [
                Paragraph(str(r[0]), cell),
                Paragraph(str(r[1]), cell),
                Paragraph(str(r[2]), cell),
                Paragraph(str(r[3]), cell),
                Paragraph(str(r[4]), cell),
                Paragraph(str(r[5]), cell),
                Paragraph(str(r[6]), cell),
            ]
        )

    # ✅ 폭 조정(겹침 방지): 이슈/증빙을 넓히고 나머지 축소
    top_table = Table(
        [[Paragraph(h, cell) for h in header]] + wrapped,
        colWidths=[45, 40, 32, 38, 135, 110, 140],
        repeatRows=1,
    )

    top_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(top_table)

    doc.build(story)
    state["pdf_path"] = pdf_path
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(ReportState)

    graph.add_node("load_data", load_data)
    graph.add_node("build_outline", build_outline)
    graph.add_node("write_report", write_report)
    graph.add_node("review_report", review_report)
    graph.add_node("generate_pdf", generate_pdf)

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "build_outline")
    graph.add_edge("build_outline", "write_report")
    graph.add_edge("write_report", "review_report")

    graph.add_conditional_edges(
        "review_report",
        should_reflect,
        {"write_report": "write_report", "generate_pdf": "generate_pdf"},
    )

    graph.add_edge("generate_pdf", END)
    return graph

app = build_graph().compile()

if __name__ == "__main__":
    out = app.invoke({
        "input_prompt": "전기차 충전소 관리자 종합감사 대비 사전점검 보고서 작성",
        "extra_prompt": "최근 3개월 대한민국 주요 사회 이슈를 먼저 수집하고, 전기차 충전소와 연결 가능할 때만 추가 점검으로 포함. 공문/법령 취지 기반으로 부족한 점을 최대한 상세히 지적. 마크다운/링크/목록/텍스트표 금지.",
        "reflect_count": 0,
        "max_reflect": 2,
        "pdf_path": "C:/test/test.pdf",
    })

    print(out.get("review", ""))
    print(out.get("draft", "")[:1200])
    print("PDF:", out.get("pdf_path"))

@router.post("/report", response_model=ReportResponse)
def generate_report(payload: ReportRequest):
    try:
        result = app.invoke(payload.dict())
        return ReportResponse(
            review=result.get("review", ""),
            draft=result.get("draft", ""),
            pdf_path=result.get("pdf_path", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
