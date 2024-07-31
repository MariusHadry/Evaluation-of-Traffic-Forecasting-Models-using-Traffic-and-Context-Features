import DataBase.DatasourceConfig;
import HereApi.AffectedLine;
import HereApi.DataCollector;
import org.jooq.DSLContext;
import org.jooq.SQLDialect;
import org.jooq.impl.DSL;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import static org.jooq.sources.tables.OlrEidMapping.OLR_EID_MAPPING;
import static org.jooq.sources.tables.OlrCodes.OLR_CODES;


public class DatabaseOlrToEdges {

    static Connection con;
    static DSLContext ctx;

    public static void main(String[] args) throws SQLException {
        // Load OLR Codes from Table
        con = DatasourceConfig.getConnection();
        ctx = DSL.using(con, SQLDialect.POSTGRES);
        List<String> allOLRCodesFromDB = ctx.select(OLR_CODES.OLR_CODE).distinctOn(OLR_CODES.OLR_CODE)
                .from(OLR_CODES).fetch().getValues(OLR_CODES.OLR_CODE);


        int batch_size = 50;
        int overall_counter = 0;
        ArrayList<String> batchOLRCodes = new ArrayList<>();

        DataCollector collector = new DataCollector();
        for (String olrc : allOLRCodesFromDB) {
            batchOLRCodes.add(olrc);
            overall_counter++;

            if (overall_counter % batch_size == 0 || overall_counter == allOLRCodesFromDB.size()) {
                try {
                    collector.collectInformationBasic(batchOLRCodes);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                // save mappings to database
                insertAffectedLines(collector.getListAffectedLines());

                // write progress to console
                System.out.print(" - finished: " + overall_counter + "/" + allOLRCodesFromDB.size() + "\n");

                // restore starting state for next batch
                batchOLRCodes = new ArrayList<>();
                collector.clearAffectedLines();
            }
        }
    }

    private static void insertAffectedLines(List<AffectedLine> affectedLines){
        for (AffectedLine affectedLine : affectedLines) {
            ctx.insertInto(OLR_EID_MAPPING, OLR_EID_MAPPING.OLR_CODE, OLR_EID_MAPPING.EDGE_ID,
                            OLR_EID_MAPPING.STARTING_EDGE, OLR_EID_MAPPING.ENDING_EDGE, OLR_EID_MAPPING.INDEX_IN_PATH)
                    .values(affectedLine.getOlrCode(), affectedLine.getLineId(), affectedLine.isFirst(),
                            affectedLine.isLast(), affectedLine.getIndexInPath()).onConflictDoNothing()
                    .execute();
        }
    }
}