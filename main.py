import streamlit as st
import pandas as pd
import requests
import random

import yandex_cloud_ml_sdk
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.auth import IAMTokenAuth

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
        f":green-badge[:material/check_circle: Сайт заполнен: {filled_site}] :red-badge[:material/block: Сайт пропущен: {missing_site}]"
    )
    subset_df = st.session_state['subset_df']
    site_options = [f"{row['Name']} | {row['Site']}" for _, row in subset_df.iterrows()]
    selected_option = st.selectbox('Выберите сайт из списка:', [''] + site_options, key='dropdown_site')

    # Логика подстановки в site_input только если пользователь не редактировал вручную
    if selected_option:
        selected_site = selected_option.split('|', 1)[1].strip()
        if (not st.session_state['site_input']) or (st.session_state['site_input'] == st.session_state['last_dropdown_site']):
            st.session_state['site_input'] = selected_site
        st.session_state['last_dropdown_site'] = selected_site
    else:
        st.session_state['last_dropdown_site'] = ''
    if st.button('Обновить выборку'):
        st.session_state['subset_df'] = filled_df.sample(n=min(10, len(filled_df)), random_state=random.randint(0, 100000))
        st.session_state['site_input'] = ''
        st.session_state['last_dropdown_site'] = ''
    site = st.text_input('URL сайта', key='site_input')
    if st.button('В Markdown'):
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
                resp = requests.post("https://r.jina.ai/", headers=headers, json=data, timeout=10)
                resp.raise_for_status()
                md_text = resp.text
                st.session_state['jina_md'] = md_text
            except Exception as e:
                st.session_state['jina_md'] = f"Ошибка: {e}"
        else:
            st.session_state['jina_md'] = 'Пожалуйста, введите URL сайта.'

with col2:
    md_text = st.session_state.get('jina_md', '')
    st.text_area('Markdown от Jina Reader API', value=md_text, height=350, disabled=True)
    # st.markdown(st.session_state.get('jina_md', ''))  # Можно раскомментировать для Markdown, но без скролла
    # Примерный лимит контекста для yandexgpt-lite (8000 символов)
    # Обрезаем по 'Links/Buttons:' если есть
    links_phrase = 'Links/Buttons:'
    if links_phrase in md_text:
        md_clean = md_text.split(links_phrase, 1)[0]
    else:
        md_clean = md_text
    if md_clean and model:
        tokens_count = len(model.tokenize(md_text))
        col_a, col_b, col_c = st.columns([0.6, 0.2, 0.2], gap='small')
        with col_a:
            st.badge(
                f"Токенов до раздела\nLinks/Buttons: {tokens_count}",
                color='orange',
                icon=":material/link:"
            )
        with col_b:
            st.badge(
                "YandexGPT",
                color='green' if tokens_count <= 32000 else 'red',
                icon=":material/check_circle:" if tokens_count <= 32000 else ':material/block:'
            )
        with col_c:
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
    if st.button('В описание'):
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
                result = model.configure(temperature=1).run([
                    {"role": "system", "text": sys_prompt},
                    {"role": "user",   "text": user_prompt_filled}
                ])
                gpt_text = result[0].text if result and hasattr(result[0], 'text') else str(result)
                st.session_state['gpt_resp'] = gpt_text
            except Exception as e:
                st.session_state['gpt_resp'] = f"Ошибка YandexGPT: {e}"
        if error_msg:
            st.session_state['gpt_resp'] = error_msg

with col4:
    st.text_area('Ответ от YandexGPT', value=st.session_state.get('gpt_resp', ''), height=350, disabled=True)
    # st.markdown(st.session_state.get('gpt_resp', ''))  # Можно раскомментировать для Markdown, но без скролла 