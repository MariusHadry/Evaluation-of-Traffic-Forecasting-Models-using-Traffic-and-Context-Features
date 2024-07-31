package DataSaving;

import HereApi.BoundingBox;
import TelegramApi.Telegram;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FileSaver {

    private static final Logger logger = LogManager.getLogger("FileSaver");

    public static String getHereDataBasePath(){
        return "retrievedData" + File.separator + "traffic";
    }

    public void saveHereData(String answer, String dayDate, String timeDate, BoundingBox bbox, boolean isIncident){
        String incident = isIncident ? "incidents" : "flow";
        String path = getHereDataBasePath() + File.separator + incident + File.separator + dayDate + File.separator;
        String fileName = timeDate + "_" + bbox.toString() + ".txt";

        writeToFile(answer, path, fileName);
    }

    public static String getDwdDataBasePath(){
        return "retrievedData" + File.separator + "weather";
    }

    public void saveDwdData(String weatherData, String dayDate, String timeDate, String directory){
        String path = getDwdDataBasePath() + File.separator + directory + File.separator + dayDate + File.separator;
        String fileName = timeDate + ".json";

        writeToFile(weatherData, path, fileName);
    }

    private void writeToFile(String answer, String path, String fileName) {
        File directory = new File(path);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        File file = new File(path + fileName);
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(file.getAbsoluteFile());
            bw = new BufferedWriter(fw);
            bw.write(answer);
            bw.close();
            fw.close();
        } catch (IOException e) {
            logger.error("Could not write to file: " + e, e);
            Telegram.sendMessage("Could not write to file: " + e);
            e.printStackTrace();
        }

        if(bw != null) {
            try {
                bw.close();
            } catch (IOException e) {
                logger.error("Could not close BufferedWriter: " + e, e);
                Telegram.sendMessage("Could not close BufferedWriter: " + e);
                e.printStackTrace();
            }
        }
        if(fw != null) {
            try {
                fw.close();
            } catch (IOException e) {
                logger.error("Could not close FileWriter: " + e, e);
                Telegram.sendMessage("Could not close FileWriter: " + e);
                e.printStackTrace();
            }
        }
    }

}
