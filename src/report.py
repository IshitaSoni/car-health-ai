"""
car-health-ai - Week 5: inspection report generator
===================================================
Turns one car's assessment + explanations into a short written report.

Design (the language model never sees raw data and never decides anything):
  1. build_payload()  collects the FACTS: score, rating, flagged parts, risks,
                      main drivers. Every number the report may use is in here.
  2. generate_report() asks a language model to WRITE from those facts only.
                      The model is any function  facts_json -> text. For the
                      LangChain path use langchain_llm_fn(chat_model).
  3. check_report()   validates the text: every flagged part named, no numbers
                      that are not in the facts, no absolute claims, the
                      simulated-data note present.
  4. If there is no model, it raises, or its text fails the check, the
     deterministic template (render_template) is used instead, so a report is
     always produced and always grounded.
"""
import json
import re
from health_score import (COMPONENTS, FLAG_THRESHOLD, SAFETY_CRITICAL, assess)
from explain import predict_risk, explain_car, top_drivers

HORIZON_KM = 10000
SIM_NOTE = ("These figures come from a model trained on simulated cars, so treat "
            "them as a screening aid, not a diagnosis.")
SYSTEM_PROMPT = (
    "You write short used-car inspection reports for a buyer. Use ONLY the facts "
    "in the JSON the user sends. Do not invent numbers, parts, repairs or prices. "
    "Name every flagged part with its risk percentage and its main drivers. "
    "Never say a part will fail; say it has a risk. Keep it under 220 words. "
    "End with this sentence exactly: " + SIM_NOTE
)


def build_payload(car, artifacts, background, car_id=None):
    risks = {c: float(predict_risk(artifacts[c], car)[0]) for c in COMPONENTS}
    a = assess(risks)
    flagged = a["flagged"]
    expl = explain_car(car, artifacts, background, components=flagged)
    capped = (a["rating"] == "Fair" and a["health_score"] >= 90)
    return {
        "car_id": car_id if car_id is not None else str(car["car_id"].iloc[0]),
        "n_flagged": len(flagged),
        "health_score": a["health_score"], "score_max": 100, "rating": a["rating"],
        "rating_capped_by_safety_part": capped,
        "horizon_km": HORIZON_KM, "flag_threshold_pct": int(round(FLAG_THRESHOLD * 100)),
        "flagged": [{
            "component": c, "risk_pct": int(round(risks[c] * 100)),
            "safety_critical": c in SAFETY_CRITICAL,
            "drivers": top_drivers(expl[c], car)} for c in flagged],
        "other_parts": [{"component": c, "risk_pct": int(round(risks[c] * 100))}
                        for c in sorted(COMPONENTS, key=lambda k: -risks[k]) if c not in flagged],
    }


def render_template(p):
    L = [f"# Inspection report: car {p['car_id']}", "",
         f"**Health Score {p['health_score']} out of {p['score_max']}, rated {p['rating']}.**"]
    if p["flagged"]:
        names = ", ".join(f["component"] for f in p["flagged"])
        n = p["n_flagged"]
        L.append(f"{n} {'part reaches' if n == 1 else 'parts reach'} the {p['flag_threshold_pct']}% "
                 f"risk level for failing within the next {p['horizon_km']:,} km: {names}.")
        if p["rating_capped_by_safety_part"]:
            L.append("The rating is held at Fair because a safety-critical part is flagged; "
                     "the score alone would rate it Good.")
        L += ["", "## Parts to check before buying"]
        for f in p["flagged"]:
            L += ["", f"### {f['component'].capitalize()}: {f['risk_pct']}% risk"]
            if f["drivers"]:
                for d in f["drivers"]:
                    pts = "point" if d["effect_pp"] == 1 else "points"
                    L.append(f"- {d['value_text']}: {d['direction']} the risk by about "
                             f"{d['effect_pp']} percentage {pts}")
            else:
                L.append("- no single input stands out")
    else:
        L.append(f"No part reaches the {p['flag_threshold_pct']}% risk level for failing within "
                 f"the next {p['horizon_km']:,} km.")
    if p["other_parts"]:
        L += ["", "## Other parts",
              ", ".join(f"{o['component']} {o['risk_pct']}%" for o in p["other_parts"]) + "."]
    L += ["", SIM_NOTE]
    return "\n".join(L)


# ---------- grounding check ----------
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_ABSOLUTE = re.compile(r"\b(will fail|guarantee[ds]?|certain to|definitely|for sure)\b", re.I)


def _numbers(text):
    out = set()
    for m in _NUM.findall(text):
        m = m.replace(",", "").rstrip(".")
        if m:
            out.add(float(m))
    return out


def check_report(text, p):
    """List of problems; empty list means the report is grounded."""
    problems = []
    low = text.lower()
    for f in p["flagged"]:
        stem = f["component"].rstrip("s")        # tires / tire
        if stem not in low:
            problems.append(f"flagged part not mentioned: {f['component']}")
    allowed = _numbers(json.dumps(p))
    extra = sorted(n for n in _numbers(text) - allowed)
    if extra:
        problems.append(f"numbers not in the facts: {extra}")
    if _ABSOLUTE.search(text):
        problems.append("absolute claim about failure")
    if "simulated" not in low:
        problems.append("simulated-data note missing")
    if f"{p['health_score']}" not in text:
        problems.append("health score not stated")
    return problems


def generate_report(p, llm_fn=None, max_tries=2):
    """Returns {text, source: 'llm'|'template', problems: [...]}."""
    log = []
    if llm_fn is not None:
        facts = json.dumps(p, indent=1)
        for _ in range(max_tries):
            try:
                text = llm_fn(facts)
            except Exception as e:                       # network, key, quota...
                log.append(f"llm error: {type(e).__name__}")
                break
            probs = check_report(text, p)
            if not probs:
                return {"text": text, "source": "llm", "problems": log}
            log += probs
    return {"text": render_template(p), "source": "template", "problems": log}


def langchain_llm_fn(chat_model):
    """Wrap any LangChain chat model as facts_json -> text.
    Example (needs ANTHROPIC_API_KEY and `pip install langchain-anthropic`):
        from langchain_anthropic import ChatAnthropic
        llm_fn = langchain_llm_fn(ChatAnthropic(model="claude-sonnet-5-5", temperature=0))
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Facts (JSON):\n{facts}")])
    chain = prompt | chat_model | StrOutputParser()
    return lambda facts: chain.invoke({"facts": facts})