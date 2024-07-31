package HereApi;

/**
 * Contains information needed to describe an affected line. Information needed are
 * the lineId of the affected line the negative and positive offsets for the incident.
 *
 */

public class AffectedLine {

    private long lineId;
    // positive offset
    private int posOff;
    // negative offset
    private int negOff;
    // coressponding OLR-Code
    private String olrCode;

    private boolean isFirst = false;
    private boolean isLast = false;
    private int indexInPath;

    /**
     * Constructor for lines affected by incidents.
     * @param lineId Id of the affected line
     */
    public AffectedLine(long lineId, String olrCode) {
        this.lineId = lineId;
        this.olrCode = olrCode;
    }

    public long getLineId() {
        return lineId;
    }

    public int getPosOff() {
        return posOff;
    }

    public void setPosOff(int posOff) {
        this.posOff = posOff;
    }

    public int getNegOff() {
        return negOff;
    }

    public void setNegOff(int negOff) {
        this.negOff = negOff;
    }

    public String getOlrCode() {
        return olrCode;
    }

    public boolean isFirst() {
        return isFirst;
    }

    public void setFirst(boolean first) {
        isFirst = first;
    }

    public boolean isLast() {
        return isLast;
    }

    public void setLast(boolean last) {
        isLast = last;
    }

    public int getIndexInPath() {
        return indexInPath;
    }

    public void setIndexInPath(int indexInPath) {
        this.indexInPath = indexInPath;
    }
}
