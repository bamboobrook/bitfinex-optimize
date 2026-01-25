# telegram_bot.py

import time
import requests
import threading
from loguru import logger

class TelegramBot:
    """Telegram消息发送器"""
    
    def __init__(self, api_token: str, chat_id: str, test_mod: bool):
        self.api_token = api_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{api_token}"
        self.test_mod = test_mod
    
    def send_message_sync(self, message: str, time_delta: float = 1.0) -> bool:
        """
        同步发送消息到 Telegram，带有重试机制
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }
        
        max_retries = 5
        retry_count = 0
        
        if self.test_mod == False:
            while retry_count < max_retries:
                try:
                    # 添加延迟
                    time.sleep(time_delta)
                    
                    response = requests.post(url, json=payload, timeout=10)
                    if response.status_code == 200:
                        logger.info("Telegram消息发送成功！")
                        return True
                    else:
                        logger.warning(f"发送失败 (尝试 {retry_count + 1}/{max_retries}): {response.status_code}, {response.text}")
                        
                except Exception as e:
                    logger.warning(f"发生错误 (尝试 {retry_count + 1}/{max_retries}): {e}")
                
                # 增加重试计数
                retry_count += 1
                
                # 如果还有重试次数，等待一段时间再重试（指数退避）
                if retry_count < max_retries:
                    backoff_time = time_delta + (2 ** retry_count)
                    logger.info(f"等待 {backoff_time} 秒后重试...")
                    time.sleep(backoff_time)
            
            # 所有重试都失败
            logger.error(f"发送Telegram消息失败，已尝试 {max_retries} 次")
            return False
        else:
            logger.warning('Test Mod...')
    
    def send_message_async(self, message: str):
        """
        异步发送消息到 Telegram（不等待结果）
        """
        if self.test_mod == False:
            thread = threading.Thread(target=self.send_message_sync, args=(message,))
            thread.daemon = True
            thread.start()
        else:
            logger.warning('Test Mod...')
            