from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# 自动回复逻辑
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=update.effective_chat.id, text="欢迎来到Bitfinex自动放贷机器人！")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 获取用户发送的消息
    user_text = update.message.text
    # 设置自动回复内容
    reply_text = "绑定成功"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)

if __name__ == '__main__':
    # 填入你的 Token
    application = ApplicationBuilder().token('8247146127:AAGQQuXgE2Xm8Dz6RXpo3ffvuAzaSg78Z-U').build()

    # 处理器
    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    # 运行机器人
    application.run_polling()
