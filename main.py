import streamlit as st
import pandas as pd
import requests
import random
import time
import re
import json
import markdown

import yandex_cloud_ml_sdk
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.auth import IAMTokenAuth

from dynamic_models import DynamicModelGenerator, FieldConfigManager


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

def main():
    st.set_page_config(page_title="Генератор описаний поставщика", layout="wide", page_icon="🤖")

    # --- Инициализация компонентов ---
    if 'model_generator' not in st.session_state:
        st.session_state['model_generator'] = DynamicModelGenerator()
    if 'field_manager' not in st.session_state:
        st.session_state['field_manager'] = FieldConfigManager()
    if 'custom_fields' not in st.session_state:
        st.session_state['custom_fields'] = st.session_state['field_manager'].default_fields.copy()
    if 'parsed_result' not in st.session_state:
        st.session_state['parsed_result'] = None

    # --- Функции для работы с полями ---
    def remove_field(index: int):
        """Удаляет поле по индексу"""
        if 0 <= index < len(st.session_state['custom_fields']):
            st.session_state['custom_fields'].pop(index)
            st.rerun()

    def reset_to_default():
        """Сбрасывает поля к значениям по умолчанию"""
        st.session_state['custom_fields'] = st.session_state['field_manager'].default_fields.copy()
        st.rerun()

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
    if 'structured_description' not in st.session_state:
        st.session_state['structured_description'] = ''

    # --- Нижний ряд: Промпты, поля модели и ответ YandexGPT ---
    def_sys = "Ты — эксперт по анализу компаний и извлечению структурированной информации. Твоя задача - проанализировать информацию о компании и заполнить все необходимые поля в JSON формате согласно заданной схеме."
    def_user = """Проанализируй информацию о компании и заполни все поля согласно схеме.

        Информация о компании:
        {desc}

        ВАЖНО:
        1. Заполни ВСЕ обязательные поля (required: true)
        2. Для опциональных полей используй null если информация недоступна
        3. Используй точные данные из текста
        4. Для числовых полей используй только числа (без текста)
        5. Для булевых полей используй true/false
        6. Для списков используй массив: [\"item1\", \"item2\"], либо пустой массив [].
        7. Для словарей используй объект: {{\"key\": \"value\"}}

        Отвечай ТОЛЬКО JSON объектом с данными. Придерживайся схемы данных, даже если какое-то значение не найдено."""
    # --- Основной интерфейс ---
    col1, col2 = st.columns([1, 1])

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
            refresh_button = st.button('', icon=':material/refresh:', use_container_width=True, key="refresh_sites_button")
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
            if (not st.session_state['site_input']) or (st.session_state['site_input'] == st.session_state['last_dropdown_site']):
                st.session_state['site_input'] = selected_site
            st.session_state['last_dropdown_site'] = selected_site
        else:
            st.session_state['last_dropdown_site'] = ''
        site = st.text_input('URL сайта', key='site_input')
        about_checkbox = st.checkbox('Искать "О компании"', key='about_checkbox')
        md_button = st.button('В Markdown', type='primary', icon=':material/subdirectory_arrow_right:', key='markdown_button')

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
        st.markdown('<span style="font-size:14px; color: #6c757d;">Markdown от Jina Reader API:</span>', unsafe_allow_html=True)
        st.text_area('Markdown от Jina Reader API', value=md_text, height=350, disabled=True, help="", label_visibility="collapsed")
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
                    f"{tokens_count} токен(ов)",
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

    # --- Вторая секция: Работа с YandexGPT ---
    # 1. System Prompt, User Prompt, Поля Pydantic модели (все collapsible, collapsed by default)
    # 2. Структурированное описание (text_area), затем collapsible raw output
    with st.container():
        col_prompts, col_output = st.columns([1, 1])
        with col_prompts:
            with st.expander('System Prompt', expanded=False):
                st.text_area(
                    "System Prompt",
                    value=def_sys,
                    height=150,
                    key="system_prompt",
                    label_visibility="collapsed"
                )
            with st.expander('User Prompt', expanded=False):
                st.text_area(
                    "User Prompt",
                    value=def_user,
                    height=200,
                    key="user_prompt",
                    label_visibility="collapsed"
                )
            with st.expander('Поля Pydantic модели', expanded=True):
                if st.session_state['custom_fields']:
                    fields_container = st.container()
                    with fields_container:
                        for i, field in enumerate(st.session_state['custom_fields']):
                            field_name = field['name']
                            field_type = field['type']
                            field_desc = field.get('description', '')
                            display_text = f"{field_desc} [{field_type}]"
                            with st.container():
                                fcol1, fcol2 = st.columns([0.9, 0.1])
                                with fcol1:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: #262730;
                                        border: 1px solid #6b7280;
                                        border-radius: 8px;
                                        padding: 8px 12px;
                                        margin: 4px 0;
                                        color: #FFFFFF;
                                        font-weight: 500;
                                        font-size: 15px;
                                    ">
                                        {display_text}
                                    </div>
                                    """, unsafe_allow_html=True)
                                with fcol2:
                                    if st.button("×", key=f"remove_field_{i}", help=f"Удалить поле {field_name}"):
                                        remove_field(i)
                                        st.rerun()
                # Кнопки управления полями
                fcol_add, fcol_reset, fcol_export = st.columns(3)
                with fcol_add:
                    if st.button("Добавить поле", type="secondary", key="add_field_button", use_container_width=True):
                        st.session_state['show_add_field'] = True
                with fcol_reset:
                    if st.button("Сбросить", type="secondary", key="reset_to_default_button", use_container_width=True):
                        reset_to_default()
                with fcol_export:
                    if st.button("Экспорт", type="secondary", key="export_config_button", use_container_width=True):
                        config_json = json.dumps(st.session_state['custom_fields'], ensure_ascii=False, indent=2)
                        st.download_button(
                            label="Скачать JSON",
                            data=config_json,
                            file_name="field_config.json",
                            mime="application/json"
                        )
                if st.session_state.get('show_add_field', False):
                    st.markdown("#### Добавить новое поле")
                    if 'new_field_name' not in st.session_state:
                        st.session_state['new_field_name'] = ''
                    if 'new_field_type' not in st.session_state:
                        st.session_state['new_field_type'] = 'text'
                    if 'new_field_description' not in st.session_state:
                        st.session_state['new_field_description'] = ''
                    if 'clear_form' not in st.session_state:
                        st.session_state['clear_form'] = False
                    if st.session_state['clear_form']:
                        st.session_state['new_field_name'] = ''
                        st.session_state['new_field_type'] = 'text'
                        st.session_state['new_field_description'] = ''
                        st.session_state['clear_form'] = False
                    addcol1, addcol2 = st.columns(2)
                    with addcol1:
                        field_name = st.text_input("Название поля", key="new_field_name", placeholder="Например: Контакты")
                    with addcol2:
                        field_type = st.selectbox("Тип поля", st.session_state['field_manager'].get_field_types(), key="new_field_type")
                    field_description = st.text_area("Описание поля", key="new_field_description", placeholder="Описание назначения поля", height=60)
                    addcol_save, addcol_cancel = st.columns(2)
                    with addcol_save:
                        if st.button("Сохранить", type="primary", key="save_field_button"):
                            if field_name and field_type:
                                new_field = {
                                    "name": field_name,
                                    "type": field_type,
                                    "description": field_description
                                }
                                if st.session_state['field_manager'].validate_field_config(new_field):
                                    st.session_state['custom_fields'].append(new_field)
                                    st.session_state['clear_form'] = True
                                    st.session_state['show_add_field'] = False
                                    st.rerun()
                            else:
                                st.error("Пожалуйста, заполните название и тип поля")
                    with addcol_cancel:
                        if st.button("Отмена", type="secondary", key="cancel_field_button"):
                            st.session_state['show_add_field'] = False
                            st.rerun()

            # Кнопка "В описание"
            if st.button('В описание', type='primary', icon=':material/subdirectory_arrow_right:', key='description_button'):
                error_msg = None
                gpt_text = ''
                desc = st.session_state.get('jina_md', '')
                if not site:
                    error_msg = 'Пожалуйста, введите URL сайта.'
                elif not desc or desc.startswith('Ошибка') or desc.startswith('Пожалуйста'):
                    error_msg = 'Нет валидного описания сайта для отправки в YandexGPT.'
                else:
                    try:
                        sys_prompt = st.session_state.get('system_prompt', def_sys)
                        user_prompt = st.session_state.get('user_prompt', def_user)
                        user_prompt_filled = user_prompt.replace('{desc}', desc)
                        if st.session_state['custom_fields']:
                            model_class = st.session_state['model_generator'].create_dynamic_model(
                                st.session_state['custom_fields'], 
                                "DynamicCompanyDescription"
                            )
                            parser = st.session_state['model_generator'].create_parser(model_class)
                            format_instructions = parser.get_format_instructions()
                            field_names = [field['name'] for field in st.session_state['custom_fields']]
                            field_descriptions = [field.get('description', '') for field in st.session_state['custom_fields']]
                            field_instructions = "\n\nПроанализируй информацию о компании и заполни следующие поля в JSON формате:\n"
                            for i, (name, desc) in enumerate(zip(field_names, field_descriptions)):
                                if desc:
                                    field_instructions += f"- {name}: {desc}\n"
                                else:
                                    field_instructions += f"- {name}\n"
                            enhanced_prompt = f"{user_prompt_filled}\n{field_instructions}\n\n{format_instructions}"
                            start_time = time.perf_counter()
                            result = model.configure(temperature=0.7).run([
                                {"role": "system", "text": sys_prompt},
                                {"role": "user", "text": enhanced_prompt}
                            ])
                            elapsed = time.perf_counter() - start_time
                            st.session_state['yandex_time'] = elapsed
                            gpt_text = result[0].text if result and hasattr(result[0], 'text') else str(result)
                            parsed_result = st.session_state['model_generator'].parse_llm_response(gpt_text, model_class)
                            if parsed_result:
                                st.session_state['parsed_result'] = parsed_result
                                st.session_state['gpt_resp'] = gpt_text
                                result_data = parsed_result.model_dump()
                                structured_text = ""
                                for field_name, field_value in result_data.items():
                                    if field_value is not None and field_value != "":
                                        field_config = next((field for field in st.session_state['custom_fields'] if field['name'] == field_name), None)
                                        display_name = field_config['description'] if field_config else field_name.replace('_', ' ').title()
                                        if isinstance(field_value, list):
                                            structured_text += f"**{display_name}:**\n"
                                            for item in field_value:
                                                structured_text += f"• {item}\n"
                                            structured_text += "\n"
                                        elif isinstance(field_value, dict):
                                            structured_text += f"**{display_name}:**\n"
                                            for key, value in field_value.items():
                                                structured_text += f"• {key}: {value}\n"
                                            structured_text += "\n"
                                        else:
                                            structured_text += f"**{display_name}:**\n{field_value}\n\n"
                                st.session_state['structured_description'] = structured_text
                            else:
                                st.session_state['gpt_resp'] = f"❌ Ошибка парсинга. Исходный ответ:\n\n{gpt_text}"
                                st.session_state['structured_description'] = "Не удалось сформировать структурированное описание"
                        else:
                            start_time = time.perf_counter()
                            result = model.configure(temperature=1).run([
                                {"role": "system", "text": sys_prompt},
                                {"role": "user", "text": user_prompt_filled}
                            ])
                            elapsed = time.perf_counter() - start_time
                            st.session_state['yandex_time'] = elapsed
                            gpt_text = result[0].text if result and hasattr(result[0], 'text') else str(result)
                            st.session_state['gpt_resp'] = gpt_text
                            st.session_state['structured_description'] = gpt_text
                    except Exception as e:
                        st.session_state['gpt_resp'] = f"Ошибка YandexGPT: {e}"
                if error_msg:
                    st.session_state['gpt_resp'] = error_msg

        # --- Вторая секция: Выводы ---
        # Структурированное описание (text_area), затем collapsible raw output
        with col_output:
            st.markdown('<span style="font-size:14px; color: #6c757d;">Структурированное описание:</span>', unsafe_allow_html=True)
            structured_content = st.session_state.get('structured_description', '')
            if structured_content:
                html_content = markdown.markdown(structured_content, extensions=['nl2br', 'sane_lists'])
                st.markdown(f"""
                <div style="
                    background-color: #262730;
                    border-radius: 8px;
                    padding: 16px;
                    height: 400px;
                    overflow-y: auto;
                    font-family: 'Source Sans Pro', sans-serif;
                    line-height: 1.6;
                    color: #7B7B80;
                    font-size: 15px;
                ">
                    {html_content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background-color: #262730;
                    border-radius: 8px;
                    padding: 16px;
                    height: 400px;
                    color: #7B7B80;
                    font-style: italic;
                    font-size: 15px;
                ">
                    Структурированное описание появится здесь после обработки
                </div>
                """, unsafe_allow_html=True)
            st.write('') # пустая строка для отступа
            # --- Badges for YandexGPT ---
            yandex_time = st.session_state.get('yandex_time', None)
            gpt_resp = st.session_state.get('gpt_resp', '')
            tokens_count = None
            if gpt_resp and model:
                try:
                    tokens_count = len(model.tokenize(gpt_resp))
                except Exception:
                    tokens_count = None
            badge_cols = st.columns([0.2, 0.25, 0.55], gap='small')
            with badge_cols[0]:
                if yandex_time is not None:
                    st.badge(
                        f"{yandex_time:.3f}s",
                        icon=":material/timer:"
                    )
            with badge_cols[1]:
                if tokens_count is not None:
                    st.badge(
                        f"{tokens_count} токен(ов)",
                        color='orange',
                        icon=":material/link:"
                    )

            with st.expander('Ответ от YandexGPT (JSON)', expanded=False):
                st.text_area(
                    "Ответ от YandexGPT",
                    value=st.session_state.get('gpt_resp', ''),
                    height=410,
                    key="gpt_response",
                    disabled=True,
                    help="",
                    label_visibility="collapsed"
                )

if __name__ == "__main__":
    main() 