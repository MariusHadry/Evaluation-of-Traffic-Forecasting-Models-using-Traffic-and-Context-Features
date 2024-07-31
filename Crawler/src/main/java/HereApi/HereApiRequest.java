package HereApi;

import DataSaving.FileSaver;
import Exceptions.InvalidBboxException;
import Exceptions.InvalidWGS84CoordinateException;
import TelegramApi.Telegram;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.regex.Pattern;



public class HereApiRequest {

    private static final Logger logger = LogManager.getLogger("HereApiRequest");

    public static String dayDateString = "";
    public static String timeDateString = "";
    private FileSaver fileSaver = new FileSaver();


    // TODO: your API key
    private String hereApikey = "";


    public HereApiRequest() {
    }

    /**
     * Sets URL with given bbox.
     * @throws MalformedURLException URL is in the wrong format or an unknown transmission protocol is specified.
     */
    private URL setUrl(String bbox, boolean incidentData) throws MalformedURLException {
        String baseUrl = "https://data.traffic.hereapi.com";
        String incidents = "/v7/incidents";
        String flow = "/v7/flow";
        String apiKey = "?apiKey=" + hereApikey;
        String referencing = "&locationReferencing=olr";
        if (incidentData)
            return new URL(baseUrl + incidents + apiKey + referencing + bbox);
        return new URL(baseUrl + flow + apiKey + referencing + bbox);
    }

    /**
     * Sends request to HERE API.
     * API returns xml, xml is converted to String.
     * @param bboxString Coordinates for bbox given as String to use in Api Request URL.
     * @return HERE Api answer as String
     * @throws IOException Signals a general input / output error
     */
    private String sendRequest(String bboxString, boolean incidentData) throws IOException {

        URL request = setUrl(bboxString, incidentData);
        HttpURLConnection con = (HttpURLConnection) request.openConnection();
        con.setRequestMethod("GET");
        con.setRequestProperty("Accept", "application/xml");
        int responseCode = con.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(con.getInputStream()));
            StringBuilder response = new StringBuilder();
            String readLine;
            while ((readLine = in .readLine()) != null) {
                response.append(readLine);
            } in .close();
            return response.toString();
        } else {
            logger.error("GET Request failed. URL: {}", request);
        }
        return null;
    }

    /**
     * Generates current timestamp
     *
     * @return timestamp
     */
    @NotNull
    @Contract(" -> new")
    private Timestamp getTimeStamp() {
        return new Timestamp(System.currentTimeMillis());
    }

    /**
     * Checks whether the specified bounding box is valid and the coordinates correspond to the wgs84 format.
     *
     * @param coordinatesArray Contains the WGS84 coordinates of the upper left and lower right corner of the bounding box
     * @throws InvalidBboxException            Invalid bounding box
     * @throws InvalidWGS84CoordinateException Coordinates out of WGS85 bounds
     */
    private void checkWGS84validity(double[] coordinatesArray) throws InvalidBboxException, InvalidWGS84CoordinateException {
        if (coordinatesArray.length != 4)
            throw new InvalidBboxException();

        for (int i = 0; i < 4; i++) {
            if (i == 0 || i == 2) {
                boolean validLat = (-180 <= coordinatesArray[i]) && (coordinatesArray[i] <= 180);
                if (!validLat)
                    throw new InvalidWGS84CoordinateException();
            }
            if (i == 1 || i == 3) {
                boolean validLon = (-90 <= coordinatesArray[i]) && (coordinatesArray[i] <= 90);
                if (!validLon)
                    throw new InvalidWGS84CoordinateException();
            }
        }
    }

    /**
     * Method to set bounding box size in terminal window.
     * Bounding box needs to be given in WGS84.
     * Example: 51.057,13.744;51.053,13.751
     */
    private BoundingBox setBoundingBox() throws InvalidBboxException, InvalidWGS84CoordinateException {
        //BBox Format: Bottom Left Lat,Bottom Left Lon;Upper Right Lat,Upper Right Lon
        //BBox to request data for Unterfranken
        String bboxString = "49.47898,8.974646;50.56779,10.88051";

        //get coordinates as double values
        Pattern pattern = Pattern.compile("[,;]");

        double[] coordinates = pattern.splitAsStream(bboxString)
                .mapToDouble(Double::parseDouble)
                .toArray();

        checkWGS84validity(coordinates);

        return new BoundingBox(coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
    }

    /**
     * Checks the bounding box. For the Api Request, the request bounding box is limited to a
     * maximum of 1 degrees.
     * If the specified bounding box is too large, it is broken down into sufficiently small boxes.
     * For each bounding box an API request is made and the answers of the incidents and flow API are saved as files.
     *
     * @param bbox Bounding box
     */
    private void getData(@NotNull BoundingBox bbox) {
        // Recursive bounding box query
        if ((bbox.width > 1) || (bbox.height > 1)) {
            // Box upper left
            getData(new BoundingBox(bbox.getBottomLeftLat(), bbox.getBottomLeftLon() + (bbox.getWidth() / 2),
                    bbox.getUpperRightLat() - (bbox.getHeight() / 2), bbox.getUpperRightLon()));
            // Box upper right
            getData(new BoundingBox(bbox.getBottomLeftLat() + (bbox.getHeight() / 2), bbox.getBottomLeftLon() + (bbox.getWidth() / 2),
                    bbox.getUpperRightLat(), bbox.getUpperRightLon()));
            // Box lower left
            getData(new BoundingBox(bbox.getBottomLeftLat(), bbox.getBottomLeftLon(),
                    bbox.getUpperRightLat() - (bbox.getHeight() / 2), bbox.getUpperRightLon() - (bbox.getWidth() / 2)));
            // Box lower right
            getData(new BoundingBox(bbox.getBottomLeftLat() + (bbox.getHeight() / 2), bbox.getBottomLeftLon(),
                    bbox.getUpperRightLat(), bbox.getUpperRightLon() - (bbox.getWidth() / 2)));
        } else {
            String incidentData = "";
            try {
                incidentData = sendRequest(bbox.getBboxRequestString(), true);
            } catch (IOException e) {
                logger.error("Exception: " + e, e);
                Telegram.sendMessageSpamProtected("An error occurred while trying to get data from the Here Incidents API: " + e);
                e.printStackTrace();
            }

            String flowData = "";
            try {
                flowData = sendRequest(bbox.getBboxRequestString(), false);
            } catch (IOException e) {
                logger.error("Exception: " + e, e);
                Telegram.sendMessageSpamProtected("An error occurred while trying to get data from the Here Flow API: " + e);
                e.printStackTrace();
            }

            if (incidentData != null) {
                fileSaver.saveHereData(incidentData, dayDateString, timeDateString, bbox, true);
            } else {
                Telegram.sendMessageSpamProtected("Incident data was null");
                logger.warn("Incident data was null");
            }
            if (flowData != null) {
                fileSaver.saveHereData(flowData, dayDateString, timeDateString, bbox, false);
            } else {
                Telegram.sendMessageSpamProtected("Flow data was null");
                logger.warn("Flow data was null");
            }
        }
    }

    /**
     * Method for retrieving information from the HereAPI.
     *
     * @throws InvalidBboxException            Invalid bounding box
     * @throws InvalidWGS84CoordinateException Coordinates out of WGS85 bounds
     */
    public void retrieveData() {

        // Get current timestamp
        Timestamp currentTimestamp = getTimeStamp();
        SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
        HereApiRequest.dayDateString = dateFormatter.format(currentTimestamp);

        SimpleDateFormat timeFormatter = new SimpleDateFormat("HH-mm-ss");
        HereApiRequest.timeDateString = timeFormatter.format(currentTimestamp);

        // Get recursive bounding boxes if bbox is bigger than 2 degrees
        try {
            getData(setBoundingBox());
        } catch (InvalidBboxException | InvalidWGS84CoordinateException e) {
            Telegram.sendMessageSpamProtected("Error with bounding box in HERE API request" + e);
            logger.error("Error with bounding box in HERE API request" + e, e);
        }


        logger.info("finished retrieving flow and incident data");
    }

}
