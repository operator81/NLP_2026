import os
import logging
import tempfile
from pathlib import Path
import asyncio

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import whisper
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import ffmpeg
import re

# ffmpeg путь
os.environ["PATH"] += os.pathsep + r"C:\Users\user\Desktop\NLP"

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class V_M_Class:
    def __init__(self, model_path, use_heuristics=True):
        """
        Классификатор голосовых сообщений
        
        Args:
            model_path: путь к обученной модели
            use_heuristics: использовать эвристики для явных случаев
        """
        self.use_heuristics = use_heuristics
        
        # Загружаем модель классификации текста
        logger.info(f"Загружаю модель из {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Загружаем модель Whisper для распознавания речи
        logger.info("Загружаю Whisper для распознавания речи...")
        self.asr_model = whisper.load_model("tiny")  # можно "small" для лучшего качества
        
        # Настройка устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Модель загружена на устройство: {self.device}")
    
    def conv_audio(self, input_path, output_path="converted.wav"):
        """Конвертирует аудио в формат WAV (16kHz, mono)"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except Exception as e:
            logger.error(f"Ошибка конвертации аудио: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        """Распознавание речи из аудиофайла"""
        try:
            result = self.asr_model.transcribe(audio_path, language="ru", fp16=False)
            text = result["text"].strip()
            logger.info(f"Распознанный текст: {text}")
            return text
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e}")
            return None
    
    

    def check_oficial(self, text):
        """Проверка на явно официальный текст"""
        official_patterns = [
            # Обращения
            r'уважаем(?:ые|ый)\s+(?:коллеги|сотрудники|партнеры|клиенты|участники)',
            r'глубокоуважаем(?:ый|ая)',
            r'дорог(?:ой|ая)\s+(?:коллега|сотрудник)',
            
            # Юридические/деловые конструкции
            r'в соответствии с (?:договором|пунктом|статьей|решением|приказом|протоколом)',
            r'настоящим (?:уведомляем|сообщаем|информируем|подтверждаем|доводим)',
            r'согласно (?:договору|приказу|решению|положению)',
            r'на основании (?:приказа|решения|протокола|договора)',
            r'во исполнение (?:приказа|решения|распоряжения)',
            
            # Документы
            r'(?:протокол|приказ|договор|отчет|акт|заявление)\s*(?:№?\d+|от\s+\d{2}\.\d{2}\.\d{4})',
            r'исх\.\s*№?\d+',
            r'исходящий\s*(?:№?\d+|от\s+\d{2}\.\d{2}\.\d{4})',
            
            # Просьбы и поручения
            r'прошу (?:предоставить|рассмотреть|ознакомиться|направить|согласовать|утвердить)',
            r'просим (?:предоставить|рассмотреть|ознакомиться|направить)',
            r'необходимо (?:предоставить|представить|согласовать|утвердить)',
            r'требуется (?:предоставить|представить|согласовать)',
            
            # Деловые фразы
            r'доводим до вашего сведения',
            r'ставим вас в известность',
            r'информируем вас о',
            r'сообщаем следующее',
            r'обращаем ваше внимание',
            r'напоминаем о необходимости',
            r'информируем вас, что',
            
            # Финансовые/бухгалтерские
            r'счет(?:-фактура)?\s*(?:№?\d+|от\s+\d{2}\.\d{2}\.\d{4})',
            r'оплата должна быть произведена',
            r'в течение банковских дней',
            r'банковские реквизиты',
            r'налоговая декларация',
            
            # Формальные даты и номера
            r'\d{2}\.\d{2}\.\d{4}\s*(?:года|г\.)',
            r'№\s*\d+[-/]\d+',
            r'от\s+\d{2}\.\d{2}\.\d{4}',
            r'за\s+№?\d+',
            
            # Организационные
            r'в рамках (?:проекта|программы|мероприятия)',
            r'с целью (?:дальнейшего|последующего)',
            r'в связи с (?:проведением|осуществлением|выполнением)',
            
            # Официальные введения
            r'компания\s+[«"][^«"]+[»"]',
            r'ооо\s+[«"][^«"]+[»"]',
            r'зао\s+[«"][^«"]+[»"]',
            r'ао\s+[«"][^«"]+[»"]',
            
            # Корпоративные
            r'корпоративная почта',
            r'служебная записка',
            r'внутренний регламент',
            r'должностная инструкция',
            
            # Приветствия и прощания (официальные)
            r'с уважением,',
            r'искренне ваш',
            r'с наилучшими пожеланиями,',
            r'с почтением,',
        ]
        
        text_lower = text.lower()
        for pattern in official_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def check_casual(self, text):
        """Проверка на явно разговорный текст"""
        casual_patterns = [
            # Приветствия (неформальные)
            r'^\s*(?:привет|здравствуй|здаров|хай|йо|салют|прив|здрасьте)\b',
            r'^\s*йо[у]?,\s',
            r'^\s*хей,\s',
            r'^\s*добр(?:ый|ое)\s+(?:день|утро|вечер)\b',
            
            # Вопросы о делах
            r'\b(?:как дела|что нового|че как|чо нового|как жизнь|как сам|как ты)\b',
            r'\b(?:че там|что там|чо там)\b',
            r'\b(?:как оно|как успехи|как настроение)\b',
            
            # Сленг и интернет-лексика
            r'\b(?:лол|кек|ахах|хаха|ржу|ппц|офигеть|обалдеть|ништяк|круто|ого)\b',
            r'\b(?:жесть|жесть какая|кошмар|ужас|бред|дичь|лажа)\b',
            r'\b(?:чувак|братан|брат|братишка|друг|дружище|подруга)\b',
            r'\b(?:мужик|пацан|дед|предки|родители|мама|папа|батя)\b',
            
            # Неформальные обращения
            r'\b(?:дорогой|дорогая|милый|милая|любимый|любимая)\s*(?!коллега)',
            r'\b(?:слушай|послушай|смотри|глянь|слышь)\b',
            
            # Предложения встретиться/пообщаться
            r'\b(?:давай|встретимся|гоу|пошли|погнали|замутим)\b',
            r'\b(?:кафе|кофе|пиво|бар|кино|гулять|тусить|шашлыки)\b',
            r'\b(?:на\s+счет\s+встречи|может\s+встретимся|хочешь\s+встретиться)\b',
            
            # Эмоциональные выражения
            r'!{2,}|\?{2,}',  # Множественные знаки препинания
            r'\.{3,}',  # Многоточие
            r'\b(?:ух\s+ты|вау|ого|ничего\s+себе|вот\s+это\s+да)\b',
            
            # Бытовые темы
            r'\b(?:еда|кушать|пить|спать|дом|работа|учеба|отпуск|выходные)\b',
            r'\b(?:погода|дождь|снег|жара|холод|зима|лето)\b',
            r'\b(?:фильм|сериал|музыка|книга|игра|компьютер|телефон)\b',
            
            # Сокращения и упрощения
            r'\b(?:щас|ща|чо|чё|ничо|ничё|норм|ок|окей|огонь|огоньчик)\b',
            r'\b(?:спс|пасиб|спасибочки|пжл|плз|пжста)\b',
            r'\b(?:инет|комп|телик|телевизор|магаз|универ)\b',
            
            # Неформальные вопросы
            r'\b(?:чего|чего такого|а че|а что|ну и|ну че)\b',
            r'\b(?:правда что|неужели|да ладно|не может быть)\b',
            
            # Междометия и звуки
            r'\b(?:ага|угу|эге|мм|хм|ого|упс|ой|ай|эх)\b',
            r'\b(?:тьфу|брр|хм-м|а-а|э-э)\b',
            
            # Мемы и интернет-культура
            r'\b(?:красава|молодец|умница|красавчик)\b',
            r'\b(?:поздравляю|с\s+днем\s+рождения|с\s+праздником)\b',
            
            # Неформальные прощания
            r'\b(?:пока|до\s+свидания|до\s+встречи|чао|бай|всего\s+доброго)\b',
            r'\b(?:удачи|береги\s+себя|будь\s+здоров)\b',
            
            # Фразы согласия/несогласия
            r'\b(?:точно|конечно|естественно|разумеется|безусловно)\b',
            r'\b(?:нет\s+уж|ни\s+за\s+что|ни\s+в\s+коем\s+случае)\b',
            
            # Оценочные суждения (неформальные)
            r'\b(?:класс|супер|отлично|здорово|прекрасно|великолепно)\b',
            r'\b(?:ужасно|плохо|отвратительно|кошмарно|неприятно)\b',
            
            # Временные указания (неформальные)
            r'\b(?:сегодня|завтра|послезавтра|вчера|недавно|на днях)\b',
            r'\b(?:утром|днем|вечером|ночью|сейчас|потом|позже)\b',
        ]
        
        text_lower = text.lower()
        
        # Дополнительно: проверяем длину и структуру
        words = text.split()
        if len(words) < 4:  # Очень короткие сообщения чаще разговорные
            return True
        
        # Проверяем наличие вопросительных слов в начале
        if re.match(r'^\s*(?:а|но|и|или|может|возможно|наверное)', text_lower):
            return True
        
        for pattern in casual_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def class_text(self, text):
        """Классификация текста"""
        if not text or len(text.strip()) < 3:
            return {
                'style': 'НЕОПРЕДЕЛЕНО',
                'confidence': 0.0,
                'reason': 'текст слишком короткий',
                'text': text
            }
        
        # Проверка явных маркеров (если включено)
        if self.use_heuristics:
            if self.check_oficial(text):
                return {
                    'style': 'ОФИЦИАЛЬНЫЙ',
                    'confidence': 0.99,
                    'reason': 'явный официальный маркер',
                    'text': text
                }
            
            if self.check_casual(text):
                return {
                    'style': 'РАЗГОВОРНЫЙ',
                    'confidence': 0.99,
                    'reason': 'явный разговорный маркер',
                    'text': text
                }
        
        # Классификация моделью
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            probabilities = probabilities.cpu().numpy()
        
        # Интерпретация: LABEL_1 = официальный, LABEL_0 = разговорный
        if probabilities[1] > probabilities[0]:
            return {
                'style': 'ОФИЦИАЛЬНЫЙ',
                'confidence': float(probabilities[1]),
                'reason': 'модель',
                'prob_official': float(probabilities[1]),
                'prob_casual': float(probabilities[0]),
                'text': text
            }
        else:
            return {
                'style': 'РАЗГОВОРНЫЙ',
                'confidence': float(probabilities[0]),
                'reason': 'модель',
                'prob_official': float(probabilities[1]),
                'prob_casual': float(probabilities[0]),
                'text': text
            }
    
    def class_audio(self, audio_path):
            # 1. Конвертация аудио
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        converted = self.conv_audio(audio_path, wav_path)
        if not converted:
            return None
        
        try:
            # 2. Распознавание речи
            text = self.transcribe_audio(converted)
            if not text:
                return None
            
            # 3. Классификация текста
            result = self.class_text(text)
            
            # 4. Определение категории уверенности
            if result['confidence'] < 0.6:
                result['category'] = 'сомнительно'
            elif result['confidence'] < 0.8:
                result['category'] = 'вероятно'
            else:
                result['category'] = 'точно'
            
            return result
            
        finally:
            # Очистка временных файлов
            if os.path.exists(wav_path):
                os.unlink(wav_path)

# Инициализация классификатора
MODEL_PATH = r"C:\Users\user\Desktop\NLP\speech_style_classifier"
classifier = V_M_Class(MODEL_PATH)

# Обработчики Telegram бота
async def go(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🎙️ *Бот для классификации голосовых сообщений*

Отправьте мне голосовое сообщение, и я определю его стиль:

• 🔥 *РАЗГОВОРНЫЙ* — бытовая речь, неформальное общение
• 🏢 *ОФИЦИАЛЬНЫЙ* — деловая переписка, формальные обращения

_Примеры разговорных:_
"Привет, как дела?"
"Давай встретимся завтра"

_Примеры официальных:_
"Уважаемые коллеги, прошу предоставить отчет"
"Настоящим уведомляем о собрании"

Просто отправьте голосовое сообщение! 🎤
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📋 *Как использовать бота:*

1. Отправьте голосовое сообщение на русском языке
2. Бот распознает речь и определит стиль
3. Получите результат с уверенностью

⚙️ *Точность классификации:*
• 🔴 *сомнительно* — уверенность < 60%
• 🟡 *вероятно* — уверенность 60-80%
• 🟢 *точно* — уверенность > 80%

📊 *Статистика точности модели:* ~80%

Для начала отправьте голосовое сообщение!
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик голосовых сообщений"""
    user = update.effective_user
    
    # Отправляем статус обработки
    status_message = await update.message.reply_text(
        "🎤 Получил голосовое сообщение...\n"
        "Распознаю речь... ⏳"
    )
    
    try:
        # Скачиваем голосовое сообщение
        voice_file = await update.message.voice.get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp_file:
            audio_path = tmp_file.name
        
        await voice_file.download_to_drive(audio_path)
        
        # Обновляем статус
        await status_message.edit_text(
            "✅ Речь распознана!\n"
            "Анализирую стиль... 🔍"
        )
        
        # Классифицируем
        result = classifier.class_audio(audio_path)
        
        if not result:
            await status_message.edit_text(
                "❌ Не удалось обработать голосовое сообщение.\n"
                "Попробуйте записать сообщение четче."
            )
            return
        
        # Формируем ответ
        style_emoji = "🔥" if result['style'] == 'РАЗГОВОРНЫЙ' else "🏢"
        category_emoji = {
            'сомнительно': '🔴',
            'вероятно': '🟡', 
            'точно': '🟢'
        }.get(result['category'], '⚪')
        
        response = f"""
{style_emoji} *Стиль:* {result['style']}

{category_emoji} *Уверенность:* {result['category']} ({result['confidence']:.1%})

📝 *Распознанный текст:*
"{result['text']}"

🔍 *Основание:* {result['reason']}
        """
        
        if 'prob_official' in result:
            response += f"\n📊 *Вероятности:*\n"
            response += f"• Официальный: {result['prob_official']:.1%}\n"
            response += f"• Разговорный: {result['prob_casual']:.1%}"
        
        await status_message.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка обработки голосового: {e}")
        await status_message.edit_text(
            "❌ Произошла ошибка при обработке.\n"
            "Попробуйте еще раз или отправьте текст для проверки."
        )
    
    finally:
        # Очистка временных файлов
        if os.path.exists(audio_path):
            os.unlink(audio_path)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений (для тестирования)"""
    text = update.message.text
    
    if len(text) < 1000:  # Проверяем только короткие тексты
        result = classifier.class_text(text)
        
        style_emoji = "🔥" if result['style'] == 'РАЗГОВОРНЫЙ' else "🏢"
        
        response = f"""
{style_emoji} *Стиль:* {result['style']}
📊 *Уверенность:* {result['confidence']:.1%}
🔍 *Основание:* {result['reason']}
        """
        
        if 'prob_official' in result:
            response += f"\n📈 *Вероятности:*\n"
            response += f"• Официальный: {result['prob_official']:.1%}\n"
            response += f"• Разговорный: {result['prob_casual']:.1%}"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    else:
        await update.message.reply_text(
            "Текст слишком длинный для анализа. "
            "Отправьте голосовое сообщение или короткий текст."
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла ошибка. Попробуйте еще раз."
        )

def main():
    """Основная функция запуска бота"""
    # Токен бота 
    BOT_TOKEN = "7834928701:AAHTF7nnbQG54jA3SheQxaHIP8BsyCFkhlY"  # ЗАМЕНИТЬ НА СВОЙ ТОКЕН!
    
    if BOT_TOKEN == "ВАШ_ТОКЕН_БОТА":
        print("❌ ОШИБКА: Замените BOT_TOKEN на свой токен!")
        print("1. Создайте бота через @BotFather")
        print("2. Получите токен")
        print("3. Вставьте его в код вместо 'ВАШ_ТОКЕН_БОТА'")
        return
    
    # Создаем приложение
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", go))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    # Проверка зависимостей
    try:
        import telegram
        import whisper
        import ffmpeg
    except ImportError as e:
        print(f"❌ Не установлены зависимости: {e}")
        print("Установите: pip install python-telegram-bot openai-whisper ffmpeg-python")
        exit(1)
    
    # Проверка модели
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена по пути: {MODEL_PATH}")
        print("Убедитесь, что папка speech_style_classifier существует")
        exit(1)
    
    
    main()