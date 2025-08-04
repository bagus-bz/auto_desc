import streamlit as st
import pandas as pd
import requests
import random
import time
import re

import yandex_cloud_ml_sdk
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.auth import IAMTokenAuth

# --- Функция для поиска "О компании" страниц ---
def find_about_links(md_links_section):
    # Ищем строки вида: - [caption](url)
    pattern = re.compile(r'- \[(.*?)\]\((.*?)\)', re.IGNORECASE)
    about_keywords = [
        'о компании', 'о нас', 'about', 'about us', 'about-company', 'aboutus', 'about_company', 'aboutus', 'about.html', 'about.php', 'about.aspx', 'aboutus.html', 'aboutus.php', 'aboutus.aspx'
    ]
    results = []
    for match in pattern.finditer(md_links_section):
        caption, url = match.group(1).strip().lower(), match.group(2).strip().lower()
        if any(kw in caption for kw in about_keywords) or any(kw in url for kw in about_keywords):
            results.append({'caption': match.group(1), 'url': match.group(2)})
    return results

st.set_page_config(page_title="Генератор описаний поставщика", layout="wide", page_icon="🤖")

# --- Загрузка данных ---
df = pd.read_csv('df_Company.csv')
filled_site = df['Site'].notna().sum()
missing_site = df['Site'].isna().sum()
filled_df = df[df['Site'].notna()][['Name', 'Site']].reset_index(drop=True)
if 'subset_df' not in st.session_state:
    st.session_state['subset_df'] = filled_df.sample(n=min(10, len(filled_df)), random_state=random.randint(0, 100000))
if 'site_input' not in st.session_state:
    st.session_state['site_input'] = ''
if 'jina_md' not in st.session_state:
    st.session_state['jina_md'] = ''
if 'gpt_resp' not in st.session_state:
    st.session_state['gpt_resp'] = ''
if 'last_dropdown_site' not in st.session_state:
    st.session_state['last_dropdown_site'] = ''

col1, col2 = st.columns([2,3])

# --- Верхний ряд: Информация, выбор сайта, кнопка "В Markdown" ---
with col1:
    # Настройки YandexGPT
    YC_FOLDER_ID = 'b1g1u3uo289nf62q3n08'
    YC_IAM_TOKEN = st.text_input('Токен YandexGPT', value='', type='password')
    if YC_IAM_TOKEN:
        sdk = YCloudML(folder_id=YC_FOLDER_ID.strip(), auth=IAMTokenAuth(YC_IAM_TOKEN.strip()))
        model = sdk.models.completions("yandexgpt-lite")
    else:
        model = None

    st.markdown(
        f":green-badge[:material/check_circle: Сайт заполнен: {filled_site}] :gray-badge[:material/do_not_disturb_on: Сайт пропущен: {missing_site}]"
    )
    subset_df = st.session_state['subset_df']
    site_options = [f"{row['Name']} | {row['Site']}" for _, row in subset_df.iterrows()]
    st.markdown('<span style="font-size:14px;">Выберите сайт из списка:</span>', unsafe_allow_html=True)
    col_dropdown, col_button = st.columns([0.88, 0.12], gap='small')

    with col_button:
        refresh_button = st.button('', icon=':material/refresh:', use_container_width=True)
        if refresh_button:
            st.session_state['subset_df'] = filled_df.sample(n=min(10, len(filled_df)), random_state=random.randint(0, 100000))
            st.session_state['site_input'] = ''
            st.session_state['last_dropdown_site'] = ''
            st.session_state['dropdown_site'] = ''

    with col_dropdown:
        selected_option = st.selectbox(
            'Выберите сайт из списка',
            [''] + site_options,
            key='dropdown_site',
            label_visibility="collapsed",
        )

    # Logic for updating site_input only if a new selection is made
    if selected_option:
        selected_site = selected_option.split('|', 1)[1].strip()
        # Only update if user hasn't typed manually or if last_dropdown_site matches previous
        if (not st.session_state['site_input']) or (st.session_state['site_input'] == st.session_state['last_dropdown_site']):
            st.session_state['site_input'] = selected_site
        st.session_state['last_dropdown_site'] = selected_site
    else:
        st.session_state['last_dropdown_site'] = ''
    site = st.text_input('URL сайта', key='site_input')
    col_md, col_about = st.columns([0.4, 0.6], gap='small')
    with col_md:
        md_button = st.button('В Markdown', type='primary', icon=':material/subdirectory_arrow_right:')
    with col_about:
        about_checkbox = st.checkbox('Искать "О компании"', key='about_checkbox')

# --- Логика для обычного Markdown ---
if md_button:
    if site:
        headers = {
            "Content-Type": "application/json",
            "X-Engine": "direct",
            "X-Md-Link-Style": "referenced",
            "X-Retain-Images": "none",
            "X-With-Links-Summary": "all"
        }
        data = {"url": site}
        try:
            start_time = time.perf_counter()
            resp = requests.post("https://r.jina.ai/", headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            md_text = resp.text
            elapsed = time.perf_counter() - start_time

            about_found = False
            # --- Check for about page if checkbox is active ---
            if about_checkbox:
                links_phrase = 'Links/Buttons:'
                if links_phrase in md_text:
                    md_links_section = md_text.split(links_phrase, 1)[1]
                else:
                    md_links_section = ''
                about_links = find_about_links(md_links_section)
                if about_links:
                    # Fetch about page markdown
                    about_url = about_links[0]['url']
                    about_data = {"url": about_url}
                    about_start = time.perf_counter()
                    about_resp = requests.post("https://r.jina.ai/", headers=headers, json=about_data, timeout=10)
                    about_resp.raise_for_status()
                    md_text = about_resp.text
                    elapsed = time.perf_counter() - about_start  # Optionally, use about page timing
                    about_found = True

            st.session_state['jina_md'] = md_text
            st.session_state['jina_time'] = elapsed
            st.session_state['about_found'] = about_found
            st.rerun()
        except Exception as e:
            st.session_state['jina_md'] = f"Ошибка: {e}"
            st.session_state['jina_time'] = None
            st.session_state['about_found'] = False
            st.rerun()
    else:
        st.session_state['jina_md'] = 'Пожалуйста, введите URL сайта.'
        st.session_state['jina_time'] = None
        st.session_state['about_found'] = False
        st.rerun()

# --- col2: Markdown output, badges, and about-page button ---
with col2:
    md_text = st.session_state.get('jina_md', '')
    st.text_area('Markdown от Jina Reader API', value=md_text, height=350, disabled=True)
    jina_time = st.session_state.get('jina_time', None)
    # --- Получаем секцию Links/Buttons ---
    links_phrase = 'Links/Buttons:'
    if links_phrase in md_text:
        md_links_section = md_text.split(links_phrase, 1)[1]
    else:
        md_links_section = ''
    # --- Ищем "О компании" страницы ---
    about_links = find_about_links(md_links_section)
    links_phrase = 'Links/Buttons:'
    if links_phrase in md_text:
        md_clean = md_text.split(links_phrase, 1)[0]
    else:
        md_clean = md_text
    if md_clean and model:
        tokens_count = len(model.tokenize(md_text))
        col_timer, col_tokens, col_ygpt, col_qwen = st.columns([0.15, 0.2, 0.2, 0.45], gap='small')
        with col_timer:
            if jina_time is not None:
                st.badge(
                    f"{jina_time:.3f}s",
                    icon=":material/timer:"
                )
        with col_tokens:
            st.badge(
                f"{tokens_count} токенов",
                color='orange',
                icon=":material/link:"
            )
        with col_ygpt:
            st.badge(
                "YandexGPT",
                color='green' if tokens_count <= 32000 else 'red',
                icon=":material/check_circle:" if tokens_count <= 32000 else ':material/block:'
            )
        with col_qwen:
            st.badge(
                "Qwen3 235B",
                color='green' if tokens_count <= 256000 else 'red',
                icon=":material/check_circle:" if tokens_count <= 256000 else ':material/block:'
            )

st.divider()

# --- Нижний ряд: YC настройки, промпты, кнопка и ответ YandexGPT ---
col3, col4 = st.columns([2,3])

with col3:
    def_sys = "Ты — полезный ассистент, который отвечает на вопросы о деятельности компании в четком и структурированном виде."
    def_user = "Чем занимается эта компания?\n---\n{desc}"
    sys_prompt = st.text_area('System Prompt', value=def_sys, height=68)
    user_prompt = st.text_area('User Prompt', value=def_user, height=80)
    if st.button('В описание', type='primary', icon=':material/subdirectory_arrow_right:'):
        error_msg = None
        gpt_text = ''
        desc = st.session_state.get('jina_md', '')
        if not site:
            error_msg = 'Пожалуйста, введите URL сайта.'
        elif not desc or desc.startswith('Ошибка') or desc.startswith('Пожалуйста'):
            error_msg = 'Нет валидного описания сайта для отправки в YandexGPT.'
        else:
            try:
                user_prompt_filled = user_prompt.replace('{desc}', desc)
                start_time = time.perf_counter()
                result = model.configure(temperature=1).run([
                    {"role": "system", "text": sys_prompt},
                    {"role": "user",   "text": user_prompt_filled}
                ])
                elapsed = time.perf_counter() - start_time
                gpt_text = result[0].text if result and hasattr(result[0], 'text') else str(result)
                st.session_state['gpt_resp'] = gpt_text
                st.session_state['gpt_time'] = elapsed
            except Exception as e:
                st.session_state['gpt_resp'] = f"Ошибка YandexGPT: {e}"
                st.session_state['gpt_time'] = None
        if error_msg:
            st.session_state['gpt_resp'] = error_msg
            st.session_state['gpt_time'] = None

with col4:
    st.text_area('Ответ от YandexGPT', value=st.session_state.get('gpt_resp', ''), height=350, disabled=True)
    gpt_time = st.session_state.get('gpt_time', None)
    if gpt_time is not None:
        st.badge(f"{gpt_time:.3f}s", icon=":material/timer:")
    # st.markdown(st.session_state.get('gpt_resp', ''))  # Можно раскомментировать для Markdown, но без скролла 