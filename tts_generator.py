# tts_generator.py - для генерации синтетических голосовых сообщений
import torch
import torchaudio
import os
from pathlib import Path

class TTSGenerator:
    """Генератор синтетических голосовых сообщений"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sample_rate = 24000
    
    def load_model(self):
        """Загружает Silero TTS модель"""
        torch.hub.set_dir('./models')  # Папка для кэша
        
        self.model, self.symbols, self.sample_rate, _, self.apply_tts = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )
        self.model.to(self.device)
        print(f"✅ TTS модель загружена (sample rate: {self.sample_rate})")
    
    def generate_audio(self, text, output_path, speaker='xenia'):
        """Генерирует аудио из текста"""
        if self.model is None:
            self.load_model()
        
        try:
            audio = self.apply_tts(
                texts=[text],
                model=self.model,
                sample_rate=self.sample_rate,
                symbols=self.symbols,
                device=self.device
            )
            
            # Сохраняем аудио
            torchaudio.save(output_path, audio[0].unsqueeze(0), self.sample_rate)
            print(f"✅ Аудио сохранено: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return False
    
    def generate_dataset(self, output_dir="synthetic_audio"):
        """Генерирует набор синтетических аудио"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Примеры текстов для генерации
        official_texts = [
            "Уважаемые коллеги, прошу предоставить отчет о проделанной работе",
            "Настоящим уведомляем о проведении собрания акционеров",
            "В соответствии с пунктом договора, оплата должна быть произведена",
            "Протокол заседания правления компании от пятнадцатого марта",
            "Доводим до вашего сведения информацию об изменении регламента",
        ]
        
        casual_texts = [
            "Привет, как дела? Что нового?",
            "Давай встретимся завтра в кафе на Петровке",
            "Слушай, а ты видел вчерашний матч? Было просто нереально",
            "Йоу, че как сам? Что делаешь сегодня вечером?",
            "Лол, это видео было просто огонь, ржу не могу",
        ]
        
        generated_count = 0
        
        # Генерируем официальные
        for i, text in enumerate(official_texts):
            output_path = Path(output_dir) / f"official_{i+1}.wav"
            if self.generate_audio(text, output_path):
                generated_count += 1
        
        # Генерируем разговорные
        for i, text in enumerate(casual_texts):
            output_path = Path(output_dir) / f"casual_{i+1}.wav"
            if self.generate_audio(text, output_path):
                generated_count += 1
        
        print(f"\n✅ Сгенерировано {generated_count} аудиофайлов в {output_dir}/")
        return generated_count

if __name__ == "__main__":
    generator = TTSGenerator()
    generator.generate_dataset()