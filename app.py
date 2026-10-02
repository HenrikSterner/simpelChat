"""Studievejleder-chat med profil, anbefalinger og favoritter."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st


MODEL = "phi3.5"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
AREAS = [
    "Teknologi og IT", "Matematik", "Naturvidenskab", "Medicin og sundhed",
    "Psykologi", "Mennesker og sociale forhold", "Samfund og politik",
    "Økonomi og business", "Jura", "Sprog og kommunikation",
    "Historie og kultur", "Kreativitet og design", "Medier og kommunikation",
    "Miljø og klima",
]
ABILITIES = [
    "Matematiske problemer", "Logisk tænkning", "Analyse", "Problemløsning",
    "Kreativ tænkning", "Skriftlig kommunikation", "Mundtlig kommunikation",
    "At lære nye ting", "At arbejde selvstændigt", "At arbejde sammen med andre",
    "Teknologi og computere", "At arbejde med data",
]
WORK_VALUES = [
    "Høj løn", "Jobsikkerhed", "Fleksibilitet", "Mulighed for hjemmearbejde",
    "Kreativ frihed", "At hjælpe andre mennesker", "At arbejde med teknologi",
    "At arbejde internationalt", "Gode muligheder for karriereudvikling",
    "God balance mellem arbejde og fritid",
]
SUBJECTS = [
    "Dansk", "Engelsk", "Matematik", "Fysik", "Kemi", "Biologi", "Bioteknologi",
    "Geografi", "Informatik", "Teknikfag", "Samfundsfag", "Historie", "Psykologi",
    "International økonomi", "Virksomhedsøkonomi", "Afsætning", "Erhvervsjura",
    "Sprog", "Mediefag", "Design", "Kunstneriske fag", "Andet",
]
JOB_FIELDS = [
    "Teknologi / IT", "Ingeniørarbejde", "Sundhed", "Forskning", "Økonomi",
    "Business", "Jura", "Undervisning", "Psykologi", "Kommunikation", "Design",
    "Medier", "Politik / samfund", "Natur / miljø", "Noget andet",
]
WORKDAY = [
    "Arbejde ved computer", "Analysere information", "Løse problemer",
    "Arbejde med mennesker", "Skabe/design", "Arbejde med tal/data",
    "Bygge eller udvikle produkter", "Forske/undersøge", "Skrive og kommunikere",
    "Planlægge og organisere", "Noget andet",
]
DISLIKES = [
    "Teknologi og IT", "Matematik", "Naturvidenskab", "Medicin og sundhed",
    "Psykologi", "Mennesker og sociale forhold", "Samfund og politik",
    "Økonomi og business", "Jura", "Sprog og kommunikation",
    "Historie og kultur", "Kreativitet og design", "Medier og kommunikation",
    "Miljø og klima", "Andet",
]
SYSTEM_PROMPT = (
    "Du er en venlig, neutral og realistisk studievejleder for gymnasieelever i Danmark. "
    "Svar på dansk. Brug elevens profil som kontekst, forklar dine begrundelser, "
    "stil højst ét afklarende spørgsmål ad gangen, og pres aldrig eleven mod ét valg. "
    "Anbefalinger er inspiration, ikke en beslutning. Skeln tydeligt mellem kendte fakta "
    "og usikkerhed. Opfind aldrig adgangskrav, uddannelser, links eller jobmuligheder."
)


st.set_page_config(page_title="Studievejleder", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 8% 0%, #edf2ff 0, transparent 30rem), #f8fafc; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 5rem; }
    h1 { letter-spacing: -.035em; }
    [data-testid="stChatMessage"] { background: white; border: 1px solid #e1e7f0; border-radius: 16px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_url(path: str) -> str:
    return f"{OLLAMA_URL}{path}"


def ollama_models() -> list[str]:
    response = requests.get(api_url("/api/tags"), timeout=5)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


def model_is_installed(models: list[str]) -> bool:
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in models)


def pull_model(status: Any, progress: Any) -> None:
    with requests.post(
        api_url("/api/pull"), json={"name": MODEL, "stream": True},
        stream=True, timeout=(10, 900),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(event["error"])
            status.update(label=event.get("status", "Henter model …").capitalize(), state="running")
            total, completed = event.get("total"), event.get("completed")
            if total and completed is not None:
                progress.progress(min(completed / total, 1.0), text=f"Henter {MODEL} · {completed / total:.0%}")


def warm_up_model() -> None:
    response = requests.post(
        api_url("/api/generate"),
        json={"model": MODEL, "prompt": "", "stream": False, "keep_alive": "30m"},
        timeout=300,
    )
    response.raise_for_status()


def ollama_chat(messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.35},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(api_url("/api/chat"), json=payload, timeout=(10, 600))
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("message", {}).get("content", "")


def stream_chat(messages: list[dict[str, str]]) -> Iterator[str]:
    system_content = SYSTEM_PROMPT
    if messages and messages[0].get("role") == "system":
        system_content += "\n\n" + messages[0].get("content", "")
        messages = messages[1:]
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_content}, *messages],
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.6},
    }
    with requests.post(api_url("/api/chat"), json=payload, stream=True, timeout=(10, 600)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(event["error"])
            part = event.get("message", {}).get("content", "")
            if part:
                yield part


def friendly_error(error: Exception) -> str:
    if isinstance(error, requests.ConnectionError):
        return "Jeg kan ikke få forbindelse til Ollama. Start Ollama-programmet (eller kør ollama serve), og prøv igen."
    if isinstance(error, requests.Timeout):
        return "Ollama svarede ikke i tide. Prøv igen om et øjeblik."
    if isinstance(error, requests.HTTPError):
        try:
            detail = error.response.json().get("error", "") if error.response is not None else ""
        except (ValueError, AttributeError):
            detail = error.response.text[:200] if error.response is not None else ""
        return f"Ollama returnerede en fejl{f': {detail}' if detail else '.'}"
    return f"Noget gik galt: {error}"


def empty_profile() -> dict[str, Any]:
    return {
        "uddannelse": "STX", "klassetrin": "1.g",
        "interesser": {area: 3 for area in AREAS}, "topinteresser": [],
        "undgaa": [], "undgaa_andet": "", "bedste_fag": [], "mindst_fag": [],
        "staerkeste_fag": [], "universitetsfag_svar": "Ved ikke", "universitetsfag": "",
        "evner": {item: 3 for item in ABILITIES}, "arbejdsform": 3, "teori_praksis": 3,
        "forudsigelighed": 3, "kreativitet_vigtigt": 3, "mennesker_vigtigt": 3,
        "teknologi_vigtigt": 3, "data_vigtigt": 3, "jobomraader": [],
        "jobvaerdier": {item: 3 for item in WORK_VALUES}, "uddannelser_tanker_svar": "Ved ikke",
        "uddannelser_tanker": "", "byer": "", "geografi_vigtigt": 3,
        "laengde": "Ved ikke", "jobmuligheder_vigtigt": 3, "arbejdsdag": [],
        "undgaa_job": "", "andet": "",
    }


def profile_context() -> str:
    return json.dumps(st.session_state.profile, ensure_ascii=False, indent=2)


def run_followup_analysis() -> None:
    prompt = (
        "Vurder elevens spørgeskemasvar. Stil kun opfølgende spørgsmål ved uklare, "
        "modstridende eller væsentligt manglende oplysninger. Returnér JSON med format "
        '{"questions":["..."]}. Maksimalt 5 korte, konkrete spørgsmål; returnér en tom liste, '
        "hvis profilen er tilstrækkelig. Profil:\n" + profile_context()
    )
    raw = ollama_chat([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    st.session_state.followup_questions = [str(q) for q in questions[:5] if str(q).strip()]
    st.session_state.followup_answers = {q: "" for q in st.session_state.followup_questions}
    st.session_state.stage = "opfølgning" if st.session_state.followup_questions else "gennemgang"


def generate_recommendations() -> None:
    prompt = (
        "Foreslå 3-5 relevante videregående uddannelser i Danmark ud fra elevprofilen "
        "og eventuelle opfølgende svar. Returnér kun gyldig JSON med nøglen "
        '"recommendations", som er en liste af objekter med felterne: name, institution, '
        "description, why_match, matches (liste), watch_out, admission (string; hvis usikkert "
        "skriv at det skal kontrolleres), official_url (kun en officiel URL du er sikker på, "
        "ellers tom streng), length, ratings (objekt med Teknologi, Kreativitet, Matematik, "
        "Menneskekontakt, Data; værdier Høj/Middel/Lav). Vælg reelle uddannelser og opfind "
        "ikke fakta. Vær tydelig om, at adgangskrav og detaljer skal kontrolleres på den "
        "officielle side. Profil:\n" + profile_context()
        + "\nOpfølgende svar:\n" + json.dumps(st.session_state.followup_answers, ensure_ascii=False)
    )
    raw = ollama_chat([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)
    items = data.get("recommendations", [])
    if not isinstance(items, list) or not items:
        raise ValueError("AI'en returnerede ingen uddannelsesforslag.")
    st.session_state.recommendations = items[:5]
    st.session_state.stage = "anbefalinger"


def render_advisor_chat() -> None:
    """Vis profilbaseret chat på både gennemgangs- og anbefalingssiden."""
    st.subheader("Spørg din studievejleder")
    st.caption("Assistenten bruger din profil, dine afklarende svar og eventuelle uddannelsesforslag som kontekst.")
    if not st.session_state.chat_messages:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "Tak, jeg har gemt dine svar. Jeg bruger dem til at hjælpe dig med at udforske "
                "uddannelser, der kan passe til dine interesser og ønsker. Du kan spørge mig "
                "om dine fag, arbejdsformer eller mulige uddannelsesvalg – eller fortsætte "
                "til anbefalingerne, når du er klar."
            ),
        })
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Spørg om interesser, fag eller uddannelsesvalg …",
        disabled=not st.session_state.model_ready,
        key="advisor_chat_input",
    )
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        context = (
            "Elevprofil:\n" + profile_context()
            + "\nAfklarende svar:\n" + json.dumps(st.session_state.followup_answers, ensure_ascii=False)
            + "\nAktuelle uddannelsesforslag:\n"
            + json.dumps(st.session_state.recommendations, ensure_ascii=False, indent=2)
        )
        messages = [{"role": "system", "content": "Brug denne profil og disse forslag som kontekst.\n" + context}]
        messages.extend(st.session_state.chat_messages)
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_chat(messages))
                st.session_state.chat_messages.append({"role": "assistant", "content": answer or ""})
            except (requests.RequestException, RuntimeError, ValueError) as error:
                st.error(friendly_error(error))


if "profile" not in st.session_state:
    st.session_state.profile = empty_profile()
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "stage" not in st.session_state:
    st.session_state.stage = "spørgeskema"
if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []
if "followup_answers" not in st.session_state:
    st.session_state.followup_answers = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "previous_recommendations" not in st.session_state:
    st.session_state.previous_recommendations = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pending_page" in st.session_state:
    st.session_state.selected_page = st.session_state.pop("pending_page")


st.title("🎓 Din studievejleder")
st.caption("Udforsk uddannelser med udgangspunkt i dine interesser og ønsker. Anbefalingerne er vejledning og inspiration – det er dig, der vælger.")

with st.sidebar:
    st.subheader("Din profil")
    st.caption("Du kan til enhver tid ændre svarene under Spørgeskema.")
    pages = ["Spørgeskema", "Gennemgang", "Anbefalinger", "Favoritter"]
    selected_page = st.radio("Gå til", pages, label_visibility="collapsed", key="selected_page")
    st.divider()
    if st.session_state.model_ready:
        st.success(f"Ollama · {MODEL} er klar")
    else:
        st.info("AI-funktionerne kræver Ollama.")
        if st.button("Indlæs AI", type="primary", use_container_width=True):
            try:
                with st.status("Gør AI klar …", expanded=True) as load_status:
                    models = ollama_models()
                    progress = st.progress(0, text="Kontrollerer model …")
                    if model_is_installed(models):
                        progress.progress(1, text="Modellen er klar")
                    else:
                        load_status.write(f"Henter {MODEL}. Første gang kan det tage et par minutter.")
                        pull_model(load_status, progress)
                        progress.progress(1, text="Modellen er hentet")
                    load_status.update(label="Indlæser modellen …", state="running")
                    warm_up_model()
                    st.session_state.model_ready = True
                    load_status.update(label="AI'en er klar", state="complete", expanded=False)
                st.rerun()
            except (requests.RequestException, RuntimeError, ValueError) as error:
                st.error(friendly_error(error))
    if st.button("Start spørgeskema forfra", use_container_width=True):
        st.session_state.profile = empty_profile()
        st.session_state.followup_questions = []
        st.session_state.followup_answers = {}
        st.session_state.recommendations = []
        st.session_state.previous_recommendations = []
        st.session_state.chat_messages = []
        st.session_state.stage = "spørgeskema"
        for key in list(st.session_state.keys()):
            if key.startswith("q_"):
                del st.session_state[key]
        st.rerun()


profile = st.session_state.profile

if selected_page == "Spørgeskema":
    st.header("Fortæl lidt om dig selv")
    st.write("Du kan udfylde profilen i dit eget tempo. Svarene gemmes i denne session og bruges til at tilpasse vejledningen.")
    with st.form("profile_form"):
        st.subheader("Din gymnasietid")
        c1, c2 = st.columns(2)
        with c1:
            uddannelse = st.selectbox("1. Hvilken gymnasial uddannelse går du på?", ["STX", "HHX", "HTX", "HF", "Andet"], index=["STX", "HHX", "HTX", "HF", "Andet"].index(profile["uddannelse"]), key="q_uddannelse")
        with c2:
            klassetrin = st.selectbox("2. Hvilket klassetrin går du på?", ["1.g", "2.g", "3.g", "Andet"], index=["1.g", "2.g", "3.g", "Andet"].index(profile["klassetrin"]), key="q_klassetrin")

        st.subheader("Interesser")
        st.caption("3. Vurdér din interesse fra 1 (lav) til 5 (høj).")
        interests = {}
        cols = st.columns(2)
        for i, area in enumerate(AREAS):
            with cols[i % 2]:
                interests[area] = st.slider(area, 1, 5, int(profile["interesser"].get(area, 3)), key=f"q_interest_{i}")
        top_interests = st.multiselect("4. Hvilke områder interesserer dig mest? Vælg op til 3.", AREAS, default=profile["topinteresser"], max_selections=3, key="q_top")
        avoid_areas = st.multiselect("5. Er der områder, du helst ikke vil arbejde med?", DISLIKES, default=profile["undgaa"], key="q_avoid")
        avoid_other = st.text_input("Hvis du valgte Andet, hvad tænker du på?", value=profile["undgaa_andet"], key="q_avoid_other")

        st.subheader("Fag")
        subjects = st.multiselect("6. Hvilke fag kan du bedst lide? (op til 5)", SUBJECTS, default=profile["bedste_fag"], max_selections=5, key="q_best_subjects")
        least_subjects = st.multiselect("7. Hvilke fag synes du er mindst interessante? (op til 5)", SUBJECTS, default=profile["mindst_fag"], max_selections=5, key="q_least_subjects")
        strong_subjects = st.multiselect("8. Hvilke fag føler du dig fagligt stærkest i? (op til 5)", SUBJECTS, default=profile["staerkeste_fag"], max_selections=5, key="q_strong_subjects")
        uni_study = st.radio("9. Er der fag, du gerne vil arbejde videre med på universitetet?", ["Ja", "Nej", "Ved ikke"], index=["Ja", "Nej", "Ved ikke"].index(profile["universitetsfag_svar"]), horizontal=True, key="q_uni_yes")
        uni_subjects = st.text_input("Hvis ja: hvilke fag?", value=profile["universitetsfag"], key="q_uni_subjects")

        st.subheader("Styrker og arbejdsform")
        st.caption("10. Hvor godt synes du, du er til følgende? 1 (i mindre grad) til 5 (i høj grad).")
        abilities = {}
        cols = st.columns(2)
        for i, ability in enumerate(ABILITIES):
            with cols[i % 2]:
                abilities[ability] = st.slider(ability, 1, 5, int(profile["evner"].get(ability, 3)), key=f"q_ability_{i}")
        st.caption("11–13. Flyt skyderne mellem de to ender af hver skala.")
        workstyle = st.slider("Primært alene  ←→  Primært sammen med andre", 1, 5, int(profile["arbejdsform"]), key="q_workstyle")
        theory_practice = st.slider("Teoretisk arbejde  ←→  Praktisk arbejde", 1, 5, int(profile["teori_praksis"]), key="q_theory")
        predictable = st.slider("Faste og forudsigelige opgaver  ←→  Varierede og skiftende opgaver", 1, 5, int(profile["forudsigelighed"]), key="q_variety")

        st.subheader("Hvad betyder noget i arbejdet?")
        cols = st.columns(2)
        value_controls = [
            ("Kreativitet i arbejdet", "kreativitet_vigtigt"),
            ("At arbejde med mennesker", "mennesker_vigtigt"),
            ("At arbejde med teknologi", "teknologi_vigtigt"),
            ("At arbejde med data og information", "data_vigtigt"),
        ]
        scalar_values = {}
        for i, (label, field) in enumerate(value_controls):
            with cols[i % 2]:
                scalar_values[field] = st.slider(f"{label} (1–5)", 1, 5, int(profile[field]), key=f"q_{field}")
        job_fields = st.multiselect("18. Hvilke områder kunne du forestille dig at arbejde indenfor?", JOB_FIELDS, default=profile["jobomraader"], key="q_jobareas")
        job_values = {}
        st.caption("19. Hvor vigtigt er følgende for dit fremtidige arbejde? 1 (lav betydning) til 5 (høj betydning).")
        cols = st.columns(2)
        for i, value in enumerate(WORK_VALUES):
            with cols[i % 2]:
                job_values[value] = st.slider(value, 1, 5, int(profile["jobvaerdier"].get(value, 3)), key=f"q_jobvalue_{i}")

        st.subheader("Uddannelsesønsker")
        study_ideas_yes = st.radio("20. Har du allerede nogle uddannelser i tankerne?", ["Ja", "Nej", "Ved ikke"], index=["Ja", "Nej", "Ved ikke"].index(profile["uddannelser_tanker_svar"]), horizontal=True, key="q_study_yes")
        study_ideas = st.text_area("Hvis ja: hvilke uddannelser?", value=profile["uddannelser_tanker"], key="q_study_ideas")
        cities = st.text_input("21. Har du universiteter eller byer i tankerne?", value=profile["byer"], key="q_cities")
        geo_importance = st.slider("22. Hvor vigtigt er det, hvor i Danmark uddannelsen ligger? (1–5)", 1, 5, int(profile["geografi_vigtigt"]), key="q_geo")
        length = st.radio("23. Hvor lang må uddannelsen gerne være?", ["Åben for alle længder", "Helst kortere", "Helst længere", "Ved ikke"], index=["Åben for alle længder", "Helst kortere", "Helst længere", "Ved ikke"].index(profile["laengde"]), horizontal=True, key="q_length")
        job_importance = st.slider("24. Hvor vigtigt er det, at uddannelsen fører til bestemte jobs? (1–5)", 1, 5, int(profile["jobmuligheder_vigtigt"]), key="q_job_importance")
        workday = st.multiselect("25. Hvad ville du helst lave en typisk arbejdsdag? (vælg op til 3)", WORKDAY, default=profile["arbejdsdag"], max_selections=3, key="q_workday")
        avoid_job = st.text_area("26. Hvad vil du helst undgå i dit fremtidige arbejde?", value=profile["undgaa_job"], key="q_avoid_job")
        other_info = st.text_area("27. Er der andet, systemet bør vide, når det finder uddannelser?", value=profile["andet"], key="q_other")

        submitted = st.form_submit_button(
            "Gem profil og fortsæt",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        st.session_state.profile = {
            "uddannelse": uddannelse, "klassetrin": klassetrin, "interesser": interests,
            "topinteresser": top_interests, "undgaa": avoid_areas, "undgaa_andet": avoid_other,
            "bedste_fag": subjects, "mindst_fag": least_subjects, "staerkeste_fag": strong_subjects,
            "universitetsfag_svar": uni_study, "universitetsfag": uni_subjects, "evner": abilities,
            "arbejdsform": workstyle, "teori_praksis": theory_practice, "forudsigelighed": predictable,
            **scalar_values, "jobomraader": job_fields, "jobvaerdier": job_values,
            "uddannelser_tanker_svar": study_ideas_yes, "uddannelser_tanker": study_ideas,
            "byer": cities, "geografi_vigtigt": geo_importance, "laengde": length,
            "jobmuligheder_vigtigt": job_importance, "arbejdsdag": workday,
            "undgaa_job": avoid_job, "andet": other_info,
        }
        if st.session_state.recommendations:
            st.session_state.previous_recommendations = st.session_state.recommendations.copy()
        st.session_state.recommendations = []
        st.session_state.chat_messages = []
        if st.session_state.model_ready:
            try:
                with st.spinner("AI'en ser efter uklare eller modstridende svar …"):
                    run_followup_analysis()
            except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
                st.session_state.stage = "gennemgang"
                st.warning(f"Profilen er gemt. Opfølgningen kunne ikke analyseres: {friendly_error(error)}")
        else:
            st.session_state.stage = "gennemgang"
            st.info("Profilen er gemt. Indlæs AI i sidepanelet, hvis du vil have AI-opfølgning og anbefalinger.")
        st.session_state.pending_page = "Gennemgang"
        st.rerun()


elif selected_page == "Gennemgang":
    st.header("Gennemgå dine svar")
    st.write("Du kan ændre eller supplere profilen under Spørgeskema. Når du er klar, kan du få anbefalinger.")
    if st.session_state.followup_questions:
        st.subheader("Afklarende spørgsmål")
        with st.form("followup_form"):
            answers = {}
            for index, question in enumerate(st.session_state.followup_questions):
                answers[question] = st.text_input(question, value=st.session_state.followup_answers.get(question, ""), key=f"followup_{index}")
            if st.form_submit_button("Gem svar", type="primary"):
                st.session_state.followup_answers = answers
                st.session_state.followup_questions = []
                st.rerun()
    left, right = st.columns(2)
    with left:
        st.subheader("Uddannelse og fag")
        st.write(f"**{profile['uddannelse']} · {profile['klassetrin']}**")
        st.write("**Interesser:** " + (", ".join(profile["topinteresser"]) or "Ikke valgt specifikt"))
        st.write("**Fag du bedst kan lide:** " + (", ".join(profile["bedste_fag"]) or "Ikke angivet"))
        st.write("**Faglige styrker:** " + (", ".join(profile["staerkeste_fag"]) or "Ikke angivet"))
        st.write("**Fag du gerne vil undgå:** " + (", ".join(profile["undgaa"]) or "Ikke angivet"))
        st.write("**Fag på universitetet:** " + (profile["universitetsfag"] or profile["universitetsfag_svar"]))
    with right:
        st.subheader("Ønsker til arbejdsliv")
        st.write("**Mulige arbejdsområder:** " + (", ".join(profile["jobomraader"]) or "Ikke angivet"))
        st.write("**Uddannelser i tankerne:** " + (profile["uddannelser_tanker"] or profile["uddannelser_tanker_svar"]))
        st.write("**Byer/institutioner:** " + (profile["byer"] or "Ikke angivet"))
        st.write("**Foretrukken arbejdsdag:** " + (", ".join(profile["arbejdsdag"]) or "Ikke angivet"))
        st.write("**Vil helst undgå:** " + (profile["undgaa_job"] or "Ikke angivet"))
    with st.expander("Se alle skalasvar"):
        st.json(profile, expanded=True)
    if not st.session_state.model_ready:
        st.info("Indlæs AI i sidepanelet for at få anbefalinger.")
    if st.button("Lav mine anbefalinger", type="primary", disabled=not st.session_state.model_ready, use_container_width=True):
        try:
            with st.spinner("Finder uddannelser, der matcher din profil …"):
                generate_recommendations()
            st.rerun()
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(friendly_error(error))
    st.divider()
    if not st.session_state.model_ready:
        st.info("Indlæs AI i sidepanelet for at chatte med studievejlederen.")
    render_advisor_chat()


elif selected_page == "Anbefalinger":
    st.header("Dine uddannelsesforslag")
    st.caption("Forslagene er inspiration. Kontrollér altid aktuelle adgangskrav, indhold og frister hos uddannelsesstedet.")
    if not st.session_state.recommendations:
        st.info("Gennemgå først din profil, og vælg Lav mine anbefalinger.")
        if st.button("Gå til gennemgang"):
            st.rerun()
    else:
        recommendations = st.session_state.recommendations
        if st.session_state.previous_recommendations:
            previous_names = {item.get("name", "Uddannelse") for item in st.session_state.previous_recommendations}
            current_names = {item.get("name", "Uddannelse") for item in recommendations}
            newly_suggested = current_names - previous_names
            no_longer_suggested = previous_names - current_names
            with st.expander("Sådan ændrede forslagene sig efter din profil blev opdateret"):
                st.write("**Nye forslag:** " + (", ".join(sorted(newly_suggested)) or "Ingen – listen blev justeret i begrundelser eller prioritering."))
                st.write("**Forslag fra før, som ikke længere vises:** " + (", ".join(sorted(no_longer_suggested)) or "Ingen."))
                st.caption("Sammenligningen ser på uddannelsesnavne. Begrundelser og match kan også være ændret.")
        labels = [f"{item.get('name', 'Uddannelse')} · {item.get('institution', 'Institution ikke angivet')}" for item in recommendations]
        for index, item in enumerate(recommendations, 1):
            title = item.get("name", "Uddannelse")
            institution = item.get("institution", "Institution ikke angivet")
            with st.container(border=True):
                st.subheader(f"{index}. {title}")
                st.caption(f"{institution} · {item.get('length', 'Studielængde ikke angivet')}")
                st.write(item.get("description", ""))
                st.markdown("**Hvorfor den kan passe til dig**")
                st.write(item.get("why_match", ""))
                if item.get("matches"):
                    st.write("Matcher blandt andet: " + ", ".join(map(str, item["matches"])))
                st.markdown("**Vær opmærksom på**")
                st.write(item.get("watch_out", ""))
                st.markdown("**Adgangskrav**")
                st.write(item.get("admission", "Kontrollér de aktuelle adgangskrav hos institutionen."))
                url = item.get("official_url", "")
                if isinstance(url, str) and url.startswith("https://"):
                    st.link_button("Officiel information", url)
                else:
                    st.caption("Officielt link ikke verificeret af AI'en. Søg efter uddannelsen på institutionens hjemmeside.")
                if st.button("★ Fjern fra favoritter" if item in st.session_state.favorites else "☆ Gem som favorit", key=f"fav_{index}"):
                    if item in st.session_state.favorites:
                        st.session_state.favorites.remove(item)
                    else:
                        st.session_state.favorites.append(item)
                    st.rerun()
        st.subheader("Sammenlign uddannelser")
        chosen = st.multiselect("Vælg 2–3 forslag", labels, max_selections=3, key="compare_choices")
        if len(chosen) >= 2:
            picked = [recommendations[labels.index(label)] for label in chosen]
            rows = []
            for feature, key in [
                ("Teknologi", "Teknologi"), ("Kreativitet", "Kreativitet"),
                ("Matematik", "Matematik"), ("Menneskekontakt", "Menneskekontakt"),
                ("Data", "Data"),
            ]:
                rows.append({"Område": feature, **{item.get("name", "Uddannelse"): item.get("ratings", {}).get(key, "Ikke angivet") for item in picked}})
            rows.append({"Område": "Studielængde", **{item.get("name", "Uddannelse"): item.get("length", "Ikke angivet") for item in picked}})
            st.dataframe(rows, hide_index=True, use_container_width=True)
        elif chosen:
            st.caption("Vælg mindst to forslag for at sammenligne.")

        st.divider()
        render_advisor_chat()


else:
    st.header("Dine favoritter")
    if not st.session_state.favorites:
        st.info("Du har ikke gemt nogen uddannelser endnu. Gem forslag fra siden Anbefalinger.")
    else:
        for index, item in enumerate(st.session_state.favorites):
            with st.container(border=True):
                st.subheader(item.get("name", "Uddannelse"))
                st.caption(item.get("institution", "Institution ikke angivet"))
                st.write(item.get("why_match", item.get("description", "")))
                if st.button("Fjern favorit", key=f"remove_favorite_{index}"):
                    st.session_state.favorites.pop(index)
                    st.rerun()
        if len(st.session_state.favorites) >= 2:
            st.subheader("Sammenlign favoritter")
            favorite_labels = [f"{item.get('name', 'Uddannelse')} · {item.get('institution', 'Institution ikke angivet')}" for item in st.session_state.favorites]
            chosen_favorites = st.multiselect("Vælg 2–3 favoritter", favorite_labels, max_selections=3, key="favorite_compare_choices")
            if len(chosen_favorites) >= 2:
                picked_favorites = [st.session_state.favorites[favorite_labels.index(label)] for label in chosen_favorites]
                favorite_rows = []
                for feature in ["Teknologi", "Kreativitet", "Matematik", "Menneskekontakt", "Data"]:
                    favorite_rows.append({"Område": feature, **{item.get("name", "Uddannelse"): item.get("ratings", {}).get(feature, "Ikke angivet") for item in picked_favorites}})
                favorite_rows.append({"Område": "Studielængde", **{item.get("name", "Uddannelse"): item.get("length", "Ikke angivet") for item in picked_favorites}})
                st.dataframe(favorite_rows, hide_index=True, use_container_width=True)
    if st.session_state.profile:
        st.subheader("Samlet overblik")
        top_areas = sorted(profile["interesser"].items(), key=lambda pair: pair[1], reverse=True)[:3]
        st.write("**Vigtigste interesser:** " + (", ".join(profile["topinteresser"]) or ", ".join(f"{name} ({score}/5)" for name, score in top_areas)))
        st.write("**Faglige præferencer:** bedst kan lide: " + (", ".join(profile["bedste_fag"]) or "ikke angivet") + "; stærkest i: " + (", ".join(profile["staerkeste_fag"]) or "ikke angivet") + "; mindst interessante: " + (", ".join(profile["mindst_fag"]) or "ikke angivet"))
        st.write("**Relevante uddannelser:** " + (", ".join(item.get("name", "Uddannelse") for item in st.session_state.recommendations) or "Lav anbefalinger fra din profil for at se forslag."))
        st.write("**Gemte favoritter:** " + (", ".join(item.get("name", "Uddannelse") for item in st.session_state.favorites) or "Ingen endnu"))
        st.write("Du kan ændre profilen under **Spørgeskema** og få nye anbefalinger under **Gennemgang**.")
