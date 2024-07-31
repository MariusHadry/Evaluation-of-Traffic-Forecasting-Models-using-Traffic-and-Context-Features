import DataSaving.DataZipper;
import DataSaving.FileSaver;
import DwdWeatherApi.DwdApiRequest;
import HereApi.HereApiRequest;
import TelegramApi.Telegram;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.util.Calendar;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

public class Main {

    private static final Logger logger = LogManager.getLogger("Main");

    public static void main(String[] args) {
        logger.info("Started data scraping..");

        Date startingTimeFiveMinutes = determineStartingTimeFiveMinute();
        logger.info("determined starting point for scraping: " + startingTimeFiveMinutes);

        Timer timerHereScraping = new Timer("timerHereScraping");
        timerHereScraping.schedule(new TimerTask() {
            @Override
            public void run() {
                HereApiRequest request = new HereApiRequest();
                    request.retrieveData();
            }
        }, startingTimeFiveMinutes, 300 * 1000); // wait until startingTime before doing the action and do it every 300s



        Date startingTimeDaily = determineStartingTimeDaily();
        logger.info("Determined starting time for health checks: " + startingTimeDaily);

        Date startingTimeNightly = determineStartingTimeNightly();
        logger.info("Determined starting time for data zipping: " + startingTimeNightly);

        Timer dailyTasks = new Timer("timerTelegramUpdates");
        dailyTasks.schedule(new TimerTask() {
            @Override
            public void run() {
                Telegram.sendMessage("Data scraping is still up and running");
            }
        }, startingTimeDaily, 24 * 60 * 60 * 1000);

        dailyTasks.schedule(new TimerTask() {
            @Override
            public void run() {
                // weather gemeindeWarnings / community warnings
                DataZipper.zipAndDeleteYesterdaysData(FileSaver.getDwdDataBasePath() + File.separator + DwdApiRequest.GEMEINDE_WARNINGS_NAME);
                // weather nowCast warnings
                DataZipper.zipAndDeleteYesterdaysData(FileSaver.getDwdDataBasePath() + File.separator + DwdApiRequest.NOWCAST_WARNINGS_NAME);
                // traffic flow
                DataZipper.zipAndDeleteYesterdaysData(FileSaver.getHereDataBasePath() + File.separator + "flow");
                // traffic incidents
                DataZipper.zipAndDeleteYesterdaysData(FileSaver.getHereDataBasePath() + File.separator + "incidents");
            }
        }, startingTimeNightly, 24 * 60 * 60 * 1000);

        Date startingTimeHourly = determineStartingTimeHourly();
        logger.info("Determined starting time for DWD API scraping: " + startingTimeHourly);

        Timer timerDwdScarping = new Timer("timerDwdScraping");
        timerDwdScarping.schedule(new TimerTask() {
            @Override
            public void run() {
                DwdApiRequest dwdApiRequest = new DwdApiRequest();
                dwdApiRequest.retrieveData();
            }
        }, startingTimeHourly, 60 * 60 * 1000);
    }


    /**
     * Determines the next starting point for data scraping so that the newly collected data is aligned with the old data.
     * This Function is for five minute intervals.
     *
     * @return Date object that contains the specified time
     */
    public static Date determineStartingTimeFiveMinute() {
        Calendar currentTime = Calendar.getInstance();
        currentTime.set(Calendar.SECOND, 0);
        int toAdd = 5 - (currentTime.get(Calendar.MINUTE) % 5);
        currentTime.add(Calendar.MINUTE, toAdd);
        return currentTime.getTime();
    }

    /**
     * Determines the next starting point for data scraping so that the newly collected data is aligned with the old data.
     * This Function is for hourly intervals.
     *
     * @return Date object that contains the specified time
     */
    public static Date determineStartingTimeHourly() {
        Calendar currentTime = Calendar.getInstance();
        currentTime.set(Calendar.SECOND, 0);
        currentTime.set(Calendar.MINUTE, 0);
        currentTime.add(Calendar.HOUR_OF_DAY, 1);
        return currentTime.getTime();
    }

    /**
     * Determines the next point in time that is at 8 o'clock in the morning.
     *
     * @return Date object that contains the specified time
     */
    public static Date determineStartingTimeDaily() {
        Calendar currentTime = Calendar.getInstance();
        currentTime.set(Calendar.SECOND, 0);
        currentTime.set(Calendar.MINUTE, 0);
        currentTime.set(Calendar.HOUR_OF_DAY, 8);

        if(currentTime.get(Calendar.HOUR_OF_DAY) >= 8) {
            currentTime.add(Calendar.DAY_OF_MONTH, 1);
        }

        return currentTime.getTime();
    }

    /**
     * Determines the next point in time that is at 1 o'clock in the morning.
     *
     * @return Date object that contains the specified time
     */
    public static Date determineStartingTimeNightly() {
        Calendar currentTime = Calendar.getInstance();
        currentTime.set(Calendar.SECOND, 0);
        currentTime.set(Calendar.MINUTE, 0);
        currentTime.set(Calendar.HOUR_OF_DAY, 1);

        if(currentTime.get(Calendar.HOUR_OF_DAY) >= 1) {
            currentTime.add(Calendar.DAY_OF_MONTH, 1);
        }

        return currentTime.getTime();
    }
}
