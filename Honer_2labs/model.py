import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


# Загрузка конфигурации из файла
def load_config(config_file="config.json"):
    """
    Чтение конфигурационного файла JSON и возврат его содержимого как словаря.
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        st.error("Конфигурационный файл не найден!")
        return None
    except json.JSONDecodeError:
        st.error("Ошибка в формате конфигурационного файла!")
        return None

# Функция для генерации текста


def generate_text(model_name, prompt, max_length, num_return_sequences, temperature, top_p):
    """
    Генерация текста на основе заданной модели.

    :param model_name: Название модели для генерации текста.
    :param prompt: Исходный текст-запрос.
    :param max_length: Максимальная длина сгенерированного текста.
    :param num_return_sequences: Количество генерируемых вариантов.
    :param temperature: Параметр случайности для генерации.
    :param top_p: Параметр выбора токенов по вероятности.
    :return: Сгенерированный текст.
    """
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Преобразуем текст в токены
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Генерация текста
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=temperature,
        do_sample=True,
        top_p=top_p
    )

    # Декодируем токены обратно в текст
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# Основная функция Streamlit
def main():
    # Загружаем конфигурацию
    config = load_config()

    if config is None:
        return  # Прекращаем выполнение, если конфигурация не загружена

    st.title("Генератор текста на основе GPT-2")

    # Текстовое поле для ввода запроса
    prompt = st.text_area("Введите текст для генерации:", height=150)

    # Кнопка генерации
    if st.button("Генерировать текст"):
        if prompt.strip():
            try:
                # Генерация текста с параметрами из конфигурации
                generated_text = generate_text(
                    model_name=config["model_name"],
                    prompt=prompt,
                    max_length=config["max_length"],
                    num_return_sequences=config["num_return_sequences"],
                    temperature=config["temperature"],
                    top_p=config["top_p"]
                )

                # Отображение сгенерированного текста
                st.subheader("Результат генерации:")
                st.write(generated_text)
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
        else:
            st.warning("Пожалуйста, введите текст для генерации.")


if __name__ == "__main__":
    main()
