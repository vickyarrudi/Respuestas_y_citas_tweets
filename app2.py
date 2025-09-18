import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from apify_client import ApifyClient

# (opcional) config de página
st.set_page_config(
    page_title="Twitter Scraper · Análisis de Respuestas y Citas",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Gemini (opcional)
# ---------------------------
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------
# Helpers
# ---------------------------
APIFY_CLIENT = None  # lo seteo luego tras leer el token

def read_secret_safe(key: str, env_key: str):
    """Busca credenciales en session_state, variables de entorno o st.secrets."""
    val = st.session_state.get(key)
    if val:
        return val
    val = os.getenv(env_key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return None

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]

# Regex para IDs de tweets (x.com / twitter.com)
_TWEET_ID_PATTERNS = [
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/(\d+)", re.I),
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/i/(?:web/)?status/(\d+)", re.I),
]

def extract_tweet_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip()
    if s.isdigit():
        return s
    for rx in _TWEET_ID_PATTERNS:
        m = rx.search(s)
        if m:
            return m.group(1)
    return None

# ---------------------------
# Scrapers (cacheados por ID)
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_replies(tweet_id: str) -> pd.DataFrame:
    """Obtiene replies del hilo de conversation_id = tweet_id."""
    try:
        global APIFY_CLIENT
        actor_id = "kaitoeasyapi/twitter-reply"
        run_input = {"conversation_ids": [str(tweet_id)], "maxItems": 1000}
        run = APIFY_CLIENT.actor(actor_id).call(run_input=run_input)
        items = APIFY_CLIENT.dataset(run["defaultDatasetId"]).list_items().items or []
        df = pd.DataFrame(items)

        # Derivados de autor
        if "author" in df.columns:
            df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
            df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
            df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

        # Numéricos / fechas
        for c in ["viewCount","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","author/followers"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

        cols = [
            "author/profilePicture","text","createdAt","author/userName","author/followers",
            "url","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","viewCount","id"
        ]
        df = _ensure_cols(df, cols)
        df["tipo"] = "reply"
        return df
    except Exception as e:
        st.error(f"Error al obtener respuestas: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_quotes(tweet_id: str) -> pd.DataFrame:
    """Obtiene quote tweets que citan el tweet_id."""
    try:
        global APIFY_CLIENT
        actor_id = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
        run_input = {"filter:quote": True, "quoted_tweet_id": str(tweet_id), "maxItems": 1000}
        run = APIFY_CLIENT.actor(actor_id).call(run_input=run_input)
        items = APIFY_CLIENT.dataset(run["defaultDatasetId"]).list_items().items or []
        df = pd.DataFrame(items)

        if "author" in df.columns:
            df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
            df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
            df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

        for c in ["viewCount","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","author/followers"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

        cols = [
            "author/profilePicture","text","createdAt","author/userName","author/followers",
            "url","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","viewCount","id"
        ]
        df = _ensure_cols(df, cols)
        df["tipo"] = "quote"
        return df
    except Exception as e:
        st.error(f"Error al obtener citas: {e}")
        return pd.DataFrame()

# ---------------------------
# IA (opcionales)
# ---------------------------
def clasificar_tweet(model, texto: str, contexto: str) -> str:
    if not model or not isinstance(texto, str) or not texto.strip():
        return "NEUTRO"
    prompt = (
        f"CONTEXTO: {contexto}\n"
        "Clasifica el sentimiento del siguiente tweet en POSITIVO, NEGATIVO o NEUTRO.\n"
        "Responde únicamente con POSITIVO, NEGATIVO o NEUTRO.\n"
        f'Tweet: "{texto}"\nSentimiento:'
    )
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return (resp.text or "").strip().upper()[:8] or "NEUTRO"
    except Exception:
        return "NEUTRO"

def extraer_temas_con_ia(model, textos: list[str], sentimiento: str, contexto: str, num_temas: int = 3) -> str:
    if not model:
        return "El modelo de IA no está disponible."
    textos = [t for t in textos if isinstance(t, str) and t.strip()]
    if not textos:
        return "No hay tweets suficientes."
    texto_join = "\n".join(textos[:500])
    prompt = f"""CONTEXTO: {contexto}
Aquí hay tweets clasificados como {sentimiento}. Extrae los {num_temas} temas principales.
Tweets:
{texto_join}"""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return (resp.text or "").strip()
    except Exception as e:
        return f"No se pudieron extraer temas. Error: {e}"

# ---------------------------
# App principal
# ---------------------------
def main_app():
    st.image("https://publicalab.com/assets/imgs/logo-publica-blanco.svg", width=200)
    st.markdown(
        "<h1 style='text-align:center;'>Twitter Scraper & Análisis de Respuestas y Citas · Streamlit</h1>",
        unsafe_allow_html=True
    )

    # Credenciales
    apify_token = read_secret_safe("apify_token", "APIFY_TOKEN")
    gemini_api_key = read_secret_safe("gemini_api_key", "GEMINI_API_KEY")
    if not apify_token:
        st.error("Falta APIFY_TOKEN (env o .streamlit/secrets.toml).")
        st.stop()

    # Inicializar clientes globales
    global APIFY_CLIENT
    APIFY_CLIENT = ApifyClient(apify_token)

    model = None
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            st.warning(f"No se pudo inicializar Gemini: {e}")
    else:
        st.info("Gemini no configurado. El análisis de IA será limitado.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        url_input = st.text_input(
            "URL del Tweet",
            placeholder="https://x.com/usuario/status/1234567890123456789",
            help="Pega la URL completa del tweet."
        )
        contexto = st.text_area("Contexto para el análisis de sentimiento (opcional)")
        ejecutar = st.button("🚀 Ejecutar")

    # Estado persistente
    for key, default in [
        ("tweet_id", None),
        ("df_replies", pd.DataFrame()),
        ("df_quotes", pd.DataFrame()),
        ("data_loaded", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    parsed_id = extract_tweet_id_from_url(url_input) if url_input else None

    # --- Al hacer clic en Ejecutar
    if ejecutar:
        if not parsed_id:
            st.error("No pude extraer un ID válido de esa URL. Verificá /status/<número>.")
            st.stop()

        st.session_state["tweet_id"] = parsed_id
        st.subheader("📥 Descargando datos de X…")

        # Descarga
        df_replies = get_replies(parsed_id)
        df_quotes  = get_quotes(parsed_id)

        # Limpieza (excluir original y duplicados por URL si aplica)
        if "id" in df_replies.columns:
            df_replies = df_replies[df_replies["id"].astype(str) != str(parsed_id)]
        if "id" in df_quotes.columns:
            df_quotes  = df_quotes[df_quotes["id"].astype(str) != str(parsed_id)]
        if "url" in df_replies.columns:
            df_replies = df_replies.drop_duplicates(subset=["url"]).reset_index(drop=True)
        if "url" in df_quotes.columns:
            df_quotes  = df_quotes.drop_duplicates(subset=["url"]).reset_index(drop=True)

        # Guardar en sesión
        st.session_state["df_replies"] = df_replies
        st.session_state["df_quotes"]  = df_quotes
        st.session_state["data_loaded"] = True

        st.success(f"✅ {len(df_replies)} respuestas y {len(df_quotes)} citas descargadas.")

    # --- Render si hay datos cargados
    if st.session_state["data_loaded"]:
        df_replies = st.session_state["df_replies"]
        df_quotes  = st.session_state["df_quotes"]
        df_all = pd.concat([df_replies, df_quotes], ignore_index=True)

        # Métricas
        candidate_view_cols = ["viewCount", "views", "impressions", "impression_count", "public_metrics.impression_count"]
        views_col = next((c for c in candidate_view_cols if c in df_all.columns), None)
        total_views = pd.to_numeric(df_all[views_col], errors="coerce").fillna(0).sum() if views_col else 0

        inter_cols = [c for c in ["likeCount","replyCount","retweetCount","quoteCount","bookmarkCount"] if c in df_all.columns]
        total_interacciones = (
            df_all[inter_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
            if inter_cols else 0
        )

        st.markdown(f"""
        <div style="text-align: center; padding: 12px 0;">
            <div style="font-size: 1.2em; font-weight: 700; color: #16a34a;">
                📈 Alcance Total: {int(total_views):,}
            </div>
            <div style="font-size: 1.2em; font-weight: 700; color: #2563eb;">
                💬 Interacciones Totales: {int(total_interacciones):,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Previews con imagen
        if not df_replies.empty:
            st.write("### Algunas Respuestas")
            st.data_editor(
                df_replies.head(5),
                column_config={
                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                    "url": st.column_config.LinkColumn("Tweet"),
                    "author/userName": st.column_config.TextColumn("Usuario"),
                    "text": st.column_config.TextColumn("Contenido"),
                },
                hide_index=True,
                use_container_width=True
            )
        if not df_quotes.empty:
            st.write("### Algunas Citas")
            st.data_editor(
                df_quotes.head(5),
                column_config={
                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                    "url": st.column_config.LinkColumn("Tweet"),
                    "author/userName": st.column_config.TextColumn("Usuario"),
                    "text": st.column_config.TextColumn("Contenido"),
                },
                hide_index=True,
                use_container_width=True
            )

        # --------- IA (opcional) ----------
        st.subheader("🤖 Análisis con Gemini")
        if st.button("Analizar Conversación con IA"):
            if not model:
                st.error("Configura `GEMINI_API_KEY` para usar el análisis con IA.")
                st.stop()

            df_todos = df_all.copy()
            if df_todos.empty:
                st.warning("No hay tweets para analizar.")
                st.stop()

            # Temas principales
            with st.spinner("Extrayendo temas principales…"):
                textos = df_todos["text"].dropna().astype(str).tolist() if "text" in df_todos.columns else []
                resultado_temas = extraer_temas_con_ia(model, textos, "mixto", contexto, 5)
            st.markdown("### Temas principales detectados")
            st.write(resultado_temas)

    else:
        # Solo guía si hay texto pero aún no ejecutaste
        if (url_input or url_input == "") and not ejecutar:
            st.info("Pegá una URL válida y presioná **🚀 Ejecutar** para descargar los datos.")

# ---------------------------
# Entrada
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True  # ajustá según tu auth

if st.session_state["logged_in"]:
    main_app()
