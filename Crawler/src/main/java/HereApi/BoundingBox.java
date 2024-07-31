package HereApi;

import java.math.RoundingMode;
import java.text.DecimalFormat;

/**
 *
 * This class describes the bounding box.
 *
 */

public class BoundingBox {

    double height;
    double width;
    /**
     * The upper right corner latitude of the bounding box
     */
    private double upperRightLat;
    /**
     * The upper right corner longitude of the bounding box
     */
    private double upperRightLon;
    /**
     * The bottom left latitude corner of the bounding box
     */
    private double bottomLeftLat;
    /**
     * The bottom left latitude corner of the bounding box
     */
    private double bottomLeftLon;

    public BoundingBox(double bottomLeftLat, double bottomLeftLon, double upperRightLat, double upperRightLon) {
        this.bottomLeftLat = bottomLeftLat;
        this.bottomLeftLon = bottomLeftLon;
        this.upperRightLat = upperRightLat;
        this.upperRightLon = upperRightLon;
        this.height = upperRightLat - bottomLeftLat;
        this.width = upperRightLon - bottomLeftLon;
    }

    public double getUpperRightLat() {
        return upperRightLat;
    }

    public double getUpperRightLon() {
        return upperRightLon;
    }

    public double getBottomLeftLat() {
        return bottomLeftLat;
    }

    public double getBottomLeftLon() {
        return bottomLeftLon;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    /**
     * Builds String from bounding box information to use in Here Api request
     *
     * @return Bounding Box String to use in Here Api request
     */
    public String getBboxRequestString() {
        return "&in=bbox:" + bottomLeftLon + "," + bottomLeftLat + "," + upperRightLon + "," + upperRightLat;
    }

    @Override
    public String toString() {
        DecimalFormat df = new DecimalFormat("##.#####");
        df.setRoundingMode(RoundingMode.HALF_UP);
        return df.format(bottomLeftLat) + "," + df.format(bottomLeftLon) + ";" +
                df.format(upperRightLat) + "," + df.format(upperRightLon);
    }
}
