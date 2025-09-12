import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from apify_client import ApifyClient

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

def logout():
    st.session_state["logged_in"] = False
    st.rerun()

def login_page():
    st.title("Iniciar sesión")
    if st.button("Login demo"):
        st.session_state["logged_in"] = True
        st.rerun()

def read_secret_safe(key: str, env_key: str):
    """
    Busca credenciales en este orden:
      1) st.session_state (si ya las guardaste desde la UI)
      2) Variable de entorno
      3) st.secrets (si existe; no crashea si no hay secrets.toml)
    """
    val = st.session_state.get(key)
    if val:
        return val
    val = os.getenv(env_key)
    if val:
        return val
    try:
        return st.secrets[key]  # puede lanzar si no hay secrets; capturamos arriba
    except Exception:
        return None

def sanitize_tweet_id(raw: str | int | None) -> str | None:
    """ Devuelve solo dígitos; None si no es un ID válido. """
    if raw is None:
        return None
    s = str(raw).strip()
    # quita todo lo que no sean dígitos (por ej. espacios, saltos de línea)
    s = "".join(ch for ch in s if ch.isdigit())
    return s if s.isdigit() else None

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]

# Soporta x.com y twitter.com, con o sin /i/web/, query params, etc.
_TWEET_ID_PATTERNS = [
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/(\d+)", re.I),
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/i/(?:web/)?status/(\d+)", re.I),
]

def extract_tweet_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip()
    # Si pegaron un ID por error, igual funciona
    if s.isdigit():
        return s
    for rx in _TWEET_ID_PATTERNS:
        m = rx.search(s)
        if m:
            return m.group(1)
    return None

# =============================================================================
# App principal
# =============================================================================
def main_app():
    st.image("https://publicalab.com/assets/imgs/logo-publica-blanco.svg", width=200)
    st.markdown("<h1 class='big-title'> Análisis de respuestas y citas de Tweets </h1>", unsafe_allow_html=True)

    # --- Credenciales ---
    apify_token    = read_secret_safe("apify_token", "APIFY_TOKEN")
    gemini_api_key = read_secret_safe("gemini_api_key", "GEMINI_API_KEY")

    if not apify_token:
        st.error("Falta APIFY_TOKEN. Configúralo vía variable de entorno o `.streamlit/secrets.toml`.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        url_input = st.text_input(
            "URL del Tweet",
            placeholder="https://x.com/usuario/status/1234567890123456789",
            help="Pega la URL completa del tweet."
        )
        contexto = st.text_area(
            "Contexto para el análisis de sentimiento (opcional)",
            help="Ej: 'Opiniones de clientes sobre un nuevo producto financiero'."
        )
       
        #st.markdown("---")
        #if st.button("Cerrar Sesión", key="logout_sidebar"):
            #logout()

    # --- Validación de APIFY ---
    if not apify_token:
        st.error("Necesito APIFY_TOKEN (por UI, variable de entorno o secrets.toml).")
        st.stop()

    # --- Inicializar clientes ---
    apify_client = ApifyClient(apify_token)

    model = None
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            st.warning(f"No se pudo inicializar Gemini: {e}")
    else:
        st.info("Gemini no configurado. El análisis de IA será limitado.")

    # =============================================================================
    # Scrapers 
    # =============================================================================
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_replies(tweet_id: str) -> pd.DataFrame:
        """
        Obtiene respuestas a partir del ID del tweet usando conversation_ids=[<id>].
        """
        try:
            if not (isinstance(tweet_id, str) and tweet_id.isdigit()):
                st.error("El Tweet ID debe ser numérico (solo dígitos).")
                return pd.DataFrame()

            actor_id = "kaitoeasyapi/twitter-reply"
            run_input = {
                "conversation_ids": [str(tweet_id)],   # <-- ¡array de strings!
                "maxItems": 1000                       # usar si el actor lo soporta
            }

            run = apify_client.actor(actor_id).call(run_input=run_input)
            items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
            if not items:
                return pd.DataFrame()

            df = pd.DataFrame(items)
            # después de: df = pd.DataFrame(items)
            if 'id' in df.columns:
                df = df[df['id'].astype(str) != str(tweet_id)]

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
                "author/profilePicture","text","createdAt","author/userName","author/followers","url","likeCount",
                "replyCount","retweetCount","quoteCount","bookmarkCount","viewCount"
            ]
            df = _ensure_cols(df, cols)
            df["tipo"] = "reply"
            return df

        except Exception as e:
            st.error(f"Error al obtener respuestas: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_quotes(tweet_id: str) -> pd.DataFrame:
        """
        Obtiene citas (quote tweets) por ID del tweet citado.
        """
        try:
            if not (isinstance(tweet_id, str) and tweet_id.isdigit()):
                st.error("El Tweet ID debe ser numérico (solo dígitos).")
                return pd.DataFrame()

            actor_id = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
            run_input = {
                "filter:quote": True,
                "quoted_tweet_id": str(tweet_id),
                "maxItems": 1000,
            }
            run = apify_client.actor(actor_id).call(run_input=run_input)
            items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
            if not items:
                return pd.DataFrame()

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
                "author/profilePicture","text","createdAt","author/userName","author/followers","url","likeCount",
                "replyCount","retweetCount","quoteCount","bookmarkCount","viewCount"
            ]
            df = _ensure_cols(df, cols)
            df["tipo"] = "quote"
            return df

        except Exception as e:
            st.error(f"Error al obtener citas: {e}")
            return pd.DataFrame()

    # =============================================================================
    # IA
    # =============================================================================
    def clasificar_tweet(texto: str, contexto: str) -> str:
        if not model or not isinstance(texto, str) or not texto.strip():
            return "NEUTRO"
        prompt = (
            f"CONTEXTO: {contexto}\n"
            "Clasifica el sentimiento del siguiente tweet en POSITIVO, NEGATIVO o NEUTRO.\n"
            "Responde únicamente con POSITIVO, NEGATIVO o NEUTRO. Ninguna palabra más.\n"
            f'Tweet: "{texto}"\nSentimiento:'
        )
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
            return (resp.text or "").strip().upper()[:8] or "NEUTRO"
        except Exception:
            return "NEUTRO"

    def extraer_temas_con_ia(textos: list[str], sentimiento: str, contexto: str, num_temas: int = 3) -> str:
        if not model:
            return "El modelo de IA no está disponible para extraer temas."
        textos = [t for t in textos if isinstance(t, str) and t.strip()]
        if not textos:
            return "No hay tweets suficientes para extraer temas."
        texto_join = "\n".join(textos[:500])
        prompt = f"""CONTEXTO: {contexto}
Aquí hay tweets clasificados como {sentimiento}. Extrae los {num_temas} temas principales, explicando brevemente cada uno y dando un ejemplo.
Formato:
Tema: [nombre del tema 1]
Explicación: [breve explicación]
Ejemplo: "[tweet de ejemplo relevante]"

Tema: [nombre del tema 2]
Explicación: [breve explicación]
Ejemplo: "[tweet de ejemplo relevante]"

Tweets:
{texto_join}
"""
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.4})
            return (resp.text or "").strip()
        except Exception as e:
            return f"No se pudieron extraer temas. Error: {e}"


    tweet_id = extract_tweet_id_from_url(url_input) if url_input else None

    if tweet_id:
        st.subheader("📥 Descargando datos de X/Twitter…")
        df_replies = get_replies(tweet_id)
        # 🔴 Excluir el tweet original de replies (si tu actor lo incluye)
        if "id" in df_replies.columns:
            df_replies = df_replies[df_replies["id"].astype(str) != str(tweet_id)]
    
        if "url" in df_replies.columns:
            df_replies = df_replies.drop_duplicates(subset=["url"]).reset_index(drop=True)
    
        df_quotes = get_quotes(tweet_id)
        # 🔴 Excluir el tweet original de quotes también (por si apareciera)
        if "id" in df_quotes.columns:
            df_quotes = df_quotes[df_quotes["id"].astype(str) != str(tweet_id)]
    
        if "url" in df_quotes.columns:
            df_quotes = df_quotes.drop_duplicates(subset=["url"]).reset_index(drop=True)

        st.success(f"✅ {len(df_replies)} respuestas y {len(df_quotes)} citas descargadas.")

        # Previews
        if not df_replies.empty:
            st.write("### Algunas Respuestas")
            st.dataframe(df_replies.head(5), use_container_width=True)
        if not df_quotes.empty:
            st.write("### Algunas Citas")
            st.dataframe(df_quotes.head(5), use_container_width=True)

        # ---------- Análisis con IA ----------
        st.subheader("🤖 Análisis con Gemini")
        if st.button("Analizar Conversación con IA"):
            if not model:
                st.error("Configura `GEMINI_API_KEY` (UI/env/secrets) para análisis con IA.")
                st.stop()

            df_todos = pd.concat([df_replies, df_quotes], ignore_index=True)
            if df_todos.empty:
                st.warning("No hay tweets para analizar.")
                st.stop()

            # Temas generales (mixto)
            with st.spinner("Extrayendo temas principales…"):
                textos_todos = df_todos["text"].dropna().astype(str).tolist() if "text" in df_todos.columns else []
                resultados = extraer_temas_con_ia(textos_todos, sentimiento="mixto", contexto=contexto, num_temas=5)
                st.markdown("### Temas principales detectados")
                st.write(resultados)

            # Clasificación de sentimientos (concurrency moderada)
            #st.subheader("🧠 Clasificando Sentimientos…")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            df_todos = df_todos.copy()
            resultados_sent = ["NEUTRO"] * len(df_todos)
            tweets_validos = []
            if "text" in df_todos.columns:
                tweets_validos = [(i, t) for i, t in enumerate(df_todos["text"]) if pd.notna(t) and str(t).strip()]

            if not tweets_validos:
                st.info("No hay texto para clasificar.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total = len(tweets_validos)
                done = 0

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(clasificar_tweet, t, contexto): i for i, t in tweets_validos}
                    for f in as_completed(futures):
                        idx = futures[f]
                        try:
                            resultados_sent[idx] = f.result()
                        except Exception:
                            resultados_sent[idx] = "NEUTRO"
                        done += 1
                        progress_bar.progress(min(done / total, 1.0))
                        status_text.text(f"Clasificando: {done}/{total}")

                progress_bar.empty()
                status_text.empty()
                df_todos["sentimiento"] = resultados_sent
                st.success("✅ Clasificación completada.")

            # ----- TOP 10 por vistas -----
            st.markdown("---")
            st.subheader("🔥 Top 10 Tweets Más Vistos")
            if "viewCount" in df_todos.columns:
                df_views = df_todos.dropna(subset=["viewCount"])
                if not df_views.empty:
                    top_10_views = df_views.sort_values("viewCount", ascending=False).head(10).copy()
                    top_10_views_display = top_10_views.copy()
                    top_10_views_display["viewCount"] = top_10_views_display["viewCount"].apply(lambda x: f"{int(x):,}")
                    st.dataframe(
                        top_10_views_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "author/profilePicture": st.column_config.ImageColumn("Foto"),
                            "url": st.column_config.LinkColumn("URL"),
                            "viewCount": st.column_config.NumberColumn("Vistas", format="%d"),
                            "createdAt": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                            "author/userName": st.column_config.TextColumn("Usuario"),
                            "author/followers": st.column_config.NumberColumn("Seguidores", format="%d"),
                            "likeCount": st.column_config.NumberColumn("Likes", format="%d"),
                            "replyCount": st.column_config.NumberColumn("Respuestas", format="%d"),
                            "retweetCount": st.column_config.NumberColumn("Retweets", format="%d"),
                            "quoteCount": st.column_config.NumberColumn("Citas", format="%d"),
                            "bookmarkCount": st.column_config.NumberColumn("Guardados", format="%d"),
                            "text": st.column_config.TextColumn("Contenido"),
                            "sentimiento": st.column_config.TextColumn("Sentimiento"),
                            "tipo": st.column_config.TextColumn("Tipo"),
                        },
                    )
                else:
                    st.info("No hay datos de visualizaciones disponibles.")
            else:
                st.info("No se encontró la columna `viewCount`.")

            # ----- TOP 10 usuarios por seguidores -----
            st.markdown("---")
            st.subheader("👑 Top 10 Usuarios con Más Seguidores")
            if {"author/followers", "author/userName"}.issubset(df_todos.columns):
                df_users = df_todos.dropna(subset=["author/followers", "author/userName"])
                if not df_users.empty:
                    top_users = (
                        df_users.groupby("author/userName")
                        .agg(**{
                            "author/followers": pd.NamedAgg(column="author/followers", aggfunc="max"),
                            "author/profilePicture": pd.NamedAgg(column="author/profilePicture", aggfunc="first"),
                        })
                        .reset_index()
                        .sort_values("author/followers", ascending=False)
                        .head(10)
                    )
                    top_users_display = top_users.copy()
                    top_users_display["author/followers"] = top_users_display["author/followers"].apply(lambda x: f"{int(x):,}")
                    st.dataframe(
                        top_users_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "author/profilePicture": st.column_config.ImageColumn("Foto"),
                            "author/userName": st.column_config.TextColumn("Usuario"),
                            "author/followers": st.column_config.NumberColumn("Seguidores", format="%d"),
                        },
                    )
                else:
                    st.info("No hay datos de seguidores para mostrar.")
            else:
                st.info("No se encontraron columnas de usuario/seguidores.")

            # ----- Distribución de sentimientos -----
            st.subheader("📊 Distribución de Sentimientos")
            if "sentimiento" in df_todos.columns and not df_todos["sentimiento"].dropna().empty:
                counts = df_todos["sentimiento"].value_counts().reset_index()
                counts.columns = ["Sentimiento", "Cantidad"]
                counts["Porcentaje"] = counts["Cantidad"] / counts["Cantidad"].sum() * 100

                st.markdown("### Resumen Rápido")
                summary_df = counts[["Sentimiento", "Porcentaje"]].copy()
                summary_df["Porcentaje"] = summary_df["Porcentaje"].round(2).astype(str) + "%"
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                fig = px.pie(
                    counts,
                    values="Cantidad",
                    names="Sentimiento",
                    title="Distribución de Sentimientos de los Tweets",
                    hover_data=["Porcentaje"],
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aún no hay sentimientos clasificados.")

            # ----- Temas por sentimiento -----
            st.subheader("🔍 Temas Principales por Sentimiento")
            if "sentimiento" in df_todos.columns and "text" in df_todos.columns:
                for tipo in ["POSITIVO", "NEGATIVO", "NEUTRO"]:
                    subset = df_todos.loc[df_todos["sentimiento"] == tipo, "text"].dropna().astype(str).tolist()
                    if subset:
                        with st.expander(f"Mostrar temas **{tipo}** ({len(subset)} tweets)"):
                            with st.spinner(f"Extrayendo temas {tipo}…"):
                                resumen = extraer_temas_con_ia(subset, tipo, contexto)
                            st.markdown(resumen.replace("---", "---\n"))
                    else:
                        st.info(f"No hay tweets clasificados como **{tipo}**.")
            else:
                st.info("No hay datos para extraer temas por sentimiento.")

            # ----- Evolución temporal -----
            if "createdAt" in df_todos.columns:
                df_time = df_todos.copy()
                df_time["createdAt"] = pd.to_datetime(df_time["createdAt"], errors="coerce", utc=True)
                df_time = df_time.dropna(subset=["createdAt"])
                if not df_time.empty:
                    min_date = df_time["createdAt"].min().date()
                    max_date = df_time["createdAt"].max().date()
                    date_range_days = (max_date - min_date).days

                    if date_range_days <= 3:
                        df_time["time_bucket"] = df_time["createdAt"].dt.strftime("%Y-%m-%d %H:00")
                        xaxis_label = "Hora"
                    elif date_range_days <= 150:
                        df_time["time_bucket"] = df_time["createdAt"].dt.date
                        xaxis_label = "Fecha"
                    else:
                        df_time["time_bucket"] = df_time["createdAt"].dt.to_period("M").astype(str)
                        xaxis_label = "Mes"

                    st.markdown("---")
                    st.subheader("📈 Evolución de Tweets en el Tiempo")
                    timeline = df_time.groupby("time_bucket").size().reset_index(name="Cantidad de Tweets")
                    fig_tl = px.line(timeline, x="time_bucket", y="Cantidad de Tweets",
                                     title=f"Cantidad de Tweets por {xaxis_label}", markers=True)
                    fig_tl.update_layout(
                        xaxis_title=xaxis_label, yaxis_title="Número de Tweets",
                        margin=dict(t=40, b=0, l=0, r=0), yaxis_range=[0, None]
                    )
                    st.plotly_chart(fig_tl, use_container_width=True)

            # ----- Descarga -----
            st.markdown("---")
            # ========= PDF directo desde el mismo HTML =========
            st.markdown("---")
            st.info("✨ Aplicación creada con Streamlit, Apify y Google Gemini.")            

    elif url_input:
        st.error("No pude extraer un ID válido de esa URL. Revisa que tenga el formato /status/<número>.")

# --- Entrada a la app ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True  # ajusta según tu auth

if st.session_state["logged_in"]:
    main_app()
else:
    login_page()












