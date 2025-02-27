package GeometryFunctions;

public class GeometryFunctions {

    /**
     * Converts meter to degree.
     * @param lat latitude
     * @param dist distance in meter
     * @return returns distance in degree
     */
    public static double distToDeg(double lat, double dist) {

        return dist / (111.32 * 1000 * Math.cos(lat * (Math.PI / 180)));
    }

    /**
     * Converts degree to meter.
     * @param distDeg distance in degree
     * @return distance in meter
     */
    public static int distToMeter(double distDeg) {

        return (int) Math.round(distDeg * (Math.PI/180) * 6378137);
    }

}
