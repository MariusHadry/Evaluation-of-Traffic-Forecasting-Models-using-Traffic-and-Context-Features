package TelegramApi;

import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.request.SendMessage;

import java.util.Calendar;
import java.util.Date;
import java.util.concurrent.TimeUnit;

public class Telegram {

    // TODO: Add your own bot token and private chat id for notifications.
    private static final String BOT_TOKEN = "";
    private static final int PRIVATE_CHAT_ID = 0;
    private static Date lastProtectedMessageDate;

    /**
     * Instantly send a message to the private chat id.
     *
     * @param message The message that should be sent.
     */
    public static void sendMessage(String message) {
        Thread t = new Thread(() -> {
            TelegramBot bot = new TelegramBot(BOT_TOKEN);
            bot.execute(new SendMessage(PRIVATE_CHAT_ID, message));
        });

        t.start();
    }

    /**
     * This method can only send messages once a day. It should be used for error messages that could be spammed if the
     * error keeps occurring.
     *
     * @param message The message that should be sent.
     */
    public static void sendMessageSpamProtected(String message) {
        if (lastProtectedMessageDate == null) {
            lastProtectedMessageDate = Calendar.getInstance().getTime();
            sendMessage(message);
        }
        else {
            long diffInMillis = Math.abs(lastProtectedMessageDate.getTime() - Calendar.getInstance().getTime().getTime());
            long diff = TimeUnit.DAYS.convert(diffInMillis, TimeUnit.MILLISECONDS);
            if (diff >= 1){
                lastProtectedMessageDate = Calendar.getInstance().getTime();
                sendMessage(message);
            }
        }
    }
}
