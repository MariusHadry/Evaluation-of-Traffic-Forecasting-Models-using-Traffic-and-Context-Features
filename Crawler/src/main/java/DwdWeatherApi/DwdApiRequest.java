package DwdWeatherApi;

import DataSaving.FileSaver;
import TelegramApi.Telegram;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

public class DwdApiRequest {

    private static final Logger logger = LogManager.getLogger("DwdApiRequest");

    private static final String GEMEINDE_WARNINGS = "https://s3.eu-central-1.amazonaws.com/app-prod-static.warnwetter.de/v16/gemeinde_warnings_v2_en.json";
    private static final String NOWCAST_WARNINGS = "https://s3.eu-central-1.amazonaws.com/app-prod-static.warnwetter.de/v16/warnings_nowcast.json";

    public static final String GEMEINDE_WARNINGS_NAME = "gemeindeWarnings";
    public static final String NOWCAST_WARNINGS_NAME = "nowCastWarnings";

    /**
     * Method for retrieving information from the DWD API.
     *
     */
    public void retrieveData() {
        // Get current timestamp
        Date currentTime = Calendar.getInstance().getTime();
        SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
        String dayDateString = dateFormatter.format(currentTime);

        SimpleDateFormat timeFormatter = new SimpleDateFormat("HH-mm-ss");
        String timeDateString = timeFormatter.format(currentTime);

        FileSaver fs = new FileSaver();
        fs.saveDwdData(getStringFromAPI(GEMEINDE_WARNINGS), dayDateString, timeDateString, GEMEINDE_WARNINGS_NAME);
        fs.saveDwdData(getStringFromAPI(NOWCAST_WARNINGS), dayDateString, timeDateString, NOWCAST_WARNINGS_NAME);

        logger.info("finished retrieving DWD warnings data");
    }

    private String getStringFromAPI(String url) {
        try {
            URL apiEndpoint = new URL(url);
            HttpURLConnection conn = (HttpURLConnection) apiEndpoint.openConnection();
            conn.setRequestProperty("Accept-Encoding", "gzip");
            InputStream inStream = new GZIPInputStream(conn.getInputStream());
            ReadableByteChannel rbc = Channels.newChannel(inStream);

            // convert rbc to String and return it
            Scanner sc = new Scanner(rbc);
            StringBuilder sb = new StringBuilder();
            while (sc.hasNext()) {
                sb.append(sc.next());
            }

            return sb.toString();
        } catch (IOException e) {
            logger.error("Could not retrieve data from DWD API: " + e, e);
            Telegram.sendMessageSpamProtected("Could not retrieve data from DWD API: " + e);
            e.printStackTrace();
        }

        return "Error while retrieving data";
    }

}
