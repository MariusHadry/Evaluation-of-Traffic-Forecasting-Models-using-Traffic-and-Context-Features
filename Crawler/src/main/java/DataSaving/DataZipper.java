package DataSaving;

import TelegramApi.Telegram;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class DataZipper {
    /**
     * A constants for buffer size used to read/write data
     */
    private static final int BUFFER_SIZE = 4096;

    private static final Logger logger = LogManager.getLogger("DataZipper");

    /**
     * Compresses a list of files to a destination zip file
     *
     * @param listFiles   A collection of files and directories
     * @param destZipFile The path of the destination zip file
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void zip(List<File> listFiles, String destZipFile) throws FileNotFoundException,
            IOException {
        ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(destZipFile));
        for (File file : listFiles) {
            if (file.isDirectory()) {
                zipDirectory(file, file.getName(), zos);
            } else {
                zipFile(file, zos);
            }
        }
        zos.flush();
        zos.close();
    }

    /**
     * Compresses files represented in an array of paths
     *
     * @param files       a String array containing file paths
     * @param destZipFile The path of the destination zip file
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void zip(String[] files, String destZipFile) throws FileNotFoundException, IOException {
        List<File> listFiles = new ArrayList<File>();
        for (int i = 0; i < files.length; i++) {
            listFiles.add(new File(files[i]));
        }
        zip(listFiles, destZipFile);
    }

    /**
     * Compresses file or directory represented in a string containing the path
     *
     * @param file        a String containing a file or directory path
     * @param destZipFile The path of the destination zip file
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void zip(String file, String destZipFile) throws FileNotFoundException, IOException {
        List<File> listFiles = new ArrayList<File>();
        listFiles.add(new File(file));
        zip(listFiles, destZipFile);
    }

    /**
     * Adds a directory to the current zip output stream
     *
     * @param folder       the directory to be  added
     * @param parentFolder the path of parent directory
     * @param zos          the current zip output stream
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static void zipDirectory(File folder, String parentFolder,
                              ZipOutputStream zos) throws FileNotFoundException, IOException {
        for (File file : folder.listFiles()) {
            if (file.isDirectory()) {
                zipDirectory(file, parentFolder + "/" + file.getName(), zos);
                continue;
            }
            zos.putNextEntry(new ZipEntry(parentFolder + "/" + file.getName()));
            BufferedInputStream bis = new BufferedInputStream(
                    new FileInputStream(file));
            long bytesRead = 0;
            byte[] bytesIn = new byte[BUFFER_SIZE];
            int read = 0;
            while ((read = bis.read(bytesIn)) != -1) {
                zos.write(bytesIn, 0, read);
                bytesRead += read;
            }
            zos.closeEntry();
            bis.close();
        }
    }

    /**
     * Adds a file to the current zip output stream
     *
     * @param file the file to be added
     * @param zos  the current zip output stream
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static void zipFile(File file, ZipOutputStream zos)
            throws FileNotFoundException, IOException {
        zos.putNextEntry(new ZipEntry(file.getName()));
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(
                file));
        long bytesRead = 0;
        byte[] bytesIn = new byte[BUFFER_SIZE];
        int read = 0;
        while ((read = bis.read(bytesIn)) != -1) {
            zos.write(bytesIn, 0, read);
            bytesRead += read;
        }
        zos.closeEntry();
        bis.close();
    }

    /**
     * Deletes the given directory and all the files it contains.
     *
     * @param directoryToBeDeleted path to the directory as String
     * @return true if successful
     */
    public static boolean deleteDirectory(String directoryToBeDeleted) {
        return deleteDirectory(new File(directoryToBeDeleted));
    }

    /**
     * Deletes the given directory and all the files it contains.
     *
     * @param directoryToBeDeleted path to the directory as File
     * @return true if successful
     */
    public static boolean deleteDirectory(File directoryToBeDeleted) {
        File[] allContents = directoryToBeDeleted.listFiles();
        if (allContents != null) {
            for (File file : allContents) {
                deleteDirectory(file);
            }
        }
        return directoryToBeDeleted.delete();
    }

    public static void zipAndDeleteYesterdaysData(String baseDirectory) {
        String toZip = baseDirectory + File.separator + getYesterdayDateString();
        String resultingZip = toZip + ".zip";
        boolean failed = false;
        try {
            zip(toZip, resultingZip);
            failed = !deleteDirectory(toZip);
        } catch (IOException e) {
            failed = true;
            logger.error("Zipping and deleting yesterdays data failed:" + e, e);
            Telegram.sendMessageSpamProtected("Zipping and deleting yesterdays data failed:" + e);
            e.printStackTrace();
        }

        if (!failed) {
            logger.info("zipping and deleting data was successful");
        }
        else {
            logger.info("deletion/zipping was not successful");
        }
    }

    private static String getYesterdayDateString() {
        Calendar currentTime = Calendar.getInstance();
        currentTime.add(Calendar.DAY_OF_MONTH, -1);
        Date d = currentTime.getTime();
        SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
        return dateFormatter.format(d);
    }
}
