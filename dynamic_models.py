from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, create_model
import json

class FieldConfigManager:
    """Менеджер для управления конфигурацией полей"""
    
    def __init__(self):
        # Пресет заполненных полей по умолчанию
        self.default_fields = [
            {
                "name": "company_info",
                "type": "text",
                "description": "Краткое инфо о компании"
            },
            {
                "name": "service_types",
                "type": "list",
                "description": "Типы услуг или товаров"
            },
            {
                "name": "products_services",
                "type": "list",
                "description": "Список товаров и услуг"
            },
            {
                "name": "contact_info",
                "type": "text",
                "description": "Контактная информация"
            },
            {
                "name": "activity_regions",
                "type": "list",
                "description": "Регионы деятельности"
            }
        ]
    
    def get_field_types(self) -> List[str]:
        """Возвращает доступные типы полей"""
        return ["text", "number", "integer", "boolean", "list", "dict"]
    
    def validate_field_config(self, field_config: Dict[str, Any]) -> bool:
        """Валидирует конфигурацию поля"""
        required_keys = ["name", "type"]
        if not all(key in field_config for key in required_keys):
            return False
        
        if field_config["type"] not in self.get_field_types():
            return False
        
        return True

class DynamicModelGenerator:
    """Генератор динамических Pydantic моделей"""
    
    def __init__(self):
        self.field_manager = FieldConfigManager()
    
    def _get_python_type(self, field_type: str) -> type:
        """Преобразует строковый тип в Python тип (всегда Optional)"""
        type_mapping = {
            "text": Optional[str],
            "number": Optional[float],
            "integer": Optional[int],
            "boolean": Optional[bool],
            "list": Optional[List[str]],
            "dict": Optional[Dict[str, Any]]
        }
        return type_mapping.get(field_type, Optional[str])
    
    def _create_field_annotation(self, field_config: Dict[str, Any]) -> tuple:
        """Создает аннотацию поля для Pydantic модели"""
        python_type = self._get_python_type(field_config["type"])
        
        # Создаем Field с описанием и default=None для всех полей
        field_args = {}
        if field_config.get("description"):
            field_args["description"] = field_config["description"]
        
        # Все поля теперь опциональные с default=None
        field_args["default"] = None
        return (python_type, Field(**field_args))
    
    def create_dynamic_model(self, fields_config: List[Dict[str, Any]], model_name: str = "DynamicModel") -> type:
        """Создает динамическую Pydantic модель на основе конфигурации полей"""
        
        # Создаем словарь полей для модели
        model_fields = {}
        
        for field_config in fields_config:
            if self.field_manager.validate_field_config(field_config):
                field_name = field_config["name"]
                field_annotation = self._create_field_annotation(field_config)
                model_fields[field_name] = field_annotation
        
        # Создаем модель
        dynamic_model = create_model(model_name, **model_fields)
        return dynamic_model
    
    def create_parser(self, model_class: type):
        """Создает парсер для LLM ответов"""
        return PydanticOutputParser(pydantic_object=model_class)
    
    def parse_llm_response(self, response_text: str, model_class: type) -> Optional[BaseModel]:
        """Парсит ответ LLM в Pydantic модель"""
        try:
            # Извлекаем JSON из ответа
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("JSON не найден в ответе")
                return None
            
            json_str = response_text[json_start:json_end]
            
            # Попытка парсинга JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Ошибка JSON парсинга: {e}")
                print(f"Проблемный JSON: {json_str}")
                return None
            
            # Проверяем, что это не схема, а данные
            if 'properties' in data or 'type' in data and data.get('type') == 'object':
                print("Обнаружена JSON схема вместо данных")
                return None
            
            # Обрабатываем пустые значения
            processed_data = {}
            for key, value in data.items():
                if value == "NULL" or value == "":
                    processed_data[key] = None
                else:
                    processed_data[key] = value
            
            # Создаем экземпляр модели
            try:
                return model_class(**processed_data)
            except Exception as e:
                print(f"Ошибка создания модели: {e}")
                print(f"Данные: {processed_data}")
                return None
                
        except Exception as e:
            print(f"Общая ошибка парсинга: {e}")
            return None

class PydanticOutputParser:
    """Парсер для извлечения Pydantic объектов из LLM ответов"""
    
    def __init__(self, pydantic_object: type):
        self.pydantic_object = pydantic_object
    
    def get_format_instructions(self) -> str:
        """Возвращает инструкции по форматированию для LLM"""
        try:
            schema = self.pydantic_object.model_json_schema()
            
            # Извлекаем только нужные поля из схемы
            properties = schema.get('properties', {})
            required_fields = schema.get('required', [])
            
            # Создаем простую схему для LLM
            simple_schema = {}
            for field_name, field_info in properties.items():
                field_type = field_info.get('type', 'string')
                description = field_info.get('description', '')
                is_required = field_name in required_fields
                
                simple_schema[field_name] = {
                    'type': field_type,
                    'description': description,
                    'required': is_required
                }
            
            instructions = f"""
Ты должен проанализировать информацию о компании и заполнить следующие поля в JSON формате:

{json.dumps(simple_schema, indent=2, ensure_ascii=False)}

ВАЖНО: 
1. Отвечай ТОЛЬКО JSON объектом с данными, НЕ схемой
2. Заполни ВСЕ поля - если информация недоступна, используй пустую строку "" для текстовых полей
3. Используй точные данные из текста
4. Для числовых полей используй только числа (без текста) или null если нет данных
5. Для булевых полей используй true/false
6. Для списков: ["item1", "item2"] или [] если нет данных
7. Для словарей: {{"key": "value"}} или {{}} если нет данных
8. Если поле не найдено в тексте - используй пустую строку "" для текста, null для чисел
9. Для полей типа "text" возвращай строку, а не объект
10. Для полей типа "dict" возвращай объект {{"key": "value"}}
11. Для полей типа "list" НИКОГДА не используй null - используй пустой список []

Пример ответа:
{{
  "company_info": "Название компании",
  "service_types": ["тип1", "тип2"],
  "products_services": ["товар1", "товар2"],
  "contact_info": "Телефон: +7 123 456 78 90, Email: info@company.ru",
  "activity_regions": []
}}
"""
            return instructions
        except Exception as e:
            # Fallback инструкции если схема не может быть сгенерирована
            return """
Ты должен проанализировать информацию о компании и ответить в формате JSON.

ВАЖНО: 
1. Отвечай ТОЛЬКО JSON объектом с данными
2. Заполни ВСЕ поля - если информация недоступна, используй пустую строку "" для текстовых полей
3. Используй точные данные из текста
4. Для числовых полей используй только числа (без текста) или null если нет данных
5. Для булевых полей используй true/false
6. Для списков: ["item1", "item2"] или [] если нет данных
7. Для словарей: {"key": "value"} или {} если нет данных
8. Если поле не найдено в тексте - используй пустую строку "" для текста, null для чисел
9. Для полей типа "text" возвращай строку, а не объект
10. Для полей типа "dict" возвращай объект {"key": "value"}
11. Для полей типа "list" НИКОГДА не используй null - используй пустой список []

Пример ответа:
{
  "company_info": "Название компании",
  "service_types": ["тип1", "тип2"],
  "products_services": ["товар1", "товар2"],
  "contact_info": "Телефон: +7 123 456 78 90, Email: info@company.ru",
  "activity_regions": []
}
""" 