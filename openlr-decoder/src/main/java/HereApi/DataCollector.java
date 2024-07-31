package HereApi;

import Decoder.HereDecoder;
import Loader.RoutableOSMMapLoader;
import openlr.location.Location;
import openlr.map.Line;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class DataCollector {

    // Contains affected lines
    private List<AffectedLine> listAffectedLines;
    private RoutableOSMMapLoader osmMapLoader;

    public DataCollector() {

        // List to contains all affected lines
        this.listAffectedLines = new ArrayList<>();
    }

    public List<AffectedLine> getListAffectedLines() {
        return listAffectedLines;
    }

    public void clearAffectedLines() {
        this.listAffectedLines = new ArrayList<>();
    }

    public void collectInformationBasic(@NotNull List<String> olrCodeLists) throws Exception {

        // Initialize Decoder for HERE OpenLR Codes.
        HereDecoder decoderHere = new HereDecoder();

        // load map only once!
        if (osmMapLoader == null) {
            this.osmMapLoader = new RoutableOSMMapLoader();
            this.osmMapLoader.close();
        }


        for (String olrcode : olrCodeLists) {
            // Reads out TPEG-OLR Locations
            Location location = decoderHere.decodeHere(olrcode, osmMapLoader);

            int posOff;
            int negOff;

            // If location is invalid positive and negative offsets get the value -100
            if (location != null) {
                // Gets positive and negative offset
                posOff = location.getPositiveOffset();
                negOff = location.getNegativeOffset();

                // Extract affected lines from location and add to list
                getAffectedLines(location, olrcode, posOff, negOff);
            }
        }
    }

    /**
     * Extracts affected lines from decoded location. Adds incident id, line and positive / negative offset to List.
     *
     * @param location   Location decoded by the OpenLR code
     * @param posOff     From location extracted positive offset, defines the distance between the start of the
     *                   location reference path and the start of the location
     * @param negOff     From location extracted negative offset, defines the distance between the end of the
     *                   location and the end of the location reference path
     */
    private void getAffectedLines(Location location, String olrCode, int posOff, int negOff) {
        // decode location, extract list of affected lines
        List<Line> listLines = location.getLocationLines();
        AffectedLine affectedLine;

        if (listLines != null && !listLines.isEmpty()) {
            for (int i = 0; i < listLines.size(); i++) {
                affectedLine = new AffectedLine(listLines.get(i).getID(), olrCode);
                affectedLine.setIndexInPath(i);

                // if/else-if is not possible, because an olr-code could be a reference to a single edge, hence being
                // first and last edge at the same time!
                if (i == 0) {
                    // first edge
                    affectedLine.setPosOff(posOff);
                    affectedLine.setFirst(true);
                }
                if (i == listLines.size() - 1) {
                    // last edge
                    affectedLine.setNegOff(negOff);
                    affectedLine.setLast(true);
                }

                this.listAffectedLines.add(affectedLine);
            }
        }
    }
}
