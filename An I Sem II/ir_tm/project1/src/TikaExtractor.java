import java.util.HashMap;
import java.io.File;
import java.io.FileInputStream;

import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.sax.BodyContentHandler;

public class TikaExtractor {
  protected HashMap<String, String> getDocs(String folderPath) throws Exception {
    HashMap<String, String> docs = new HashMap<String, String>();

    File folder = new File(folderPath);
    if (folder.listFiles() == null) {
      throw new Exception("The path does not exist or the folder is empty!");
    }

    File[] listOfFiles = folder.listFiles();
    for (File file : listOfFiles) {
      String fileName = folderPath + "/" + file.getName();

      String ext = fileName.substring(fileName.lastIndexOf(".") + 1);
      if (!"txt".equals(ext) && !"pdf".equals(ext) && !"doc".equals(ext) && !"docx".equals(ext)) {
        throw new Exception("The folder does not contain only txt, pdf and doc(x) files!");
      }

      File currFile = new File(fileName);

      Parser parser = new AutoDetectParser();
      BodyContentHandler handler = new BodyContentHandler();
      Metadata metadata = new Metadata();
      FileInputStream inputstream = new FileInputStream(currFile);
      ParseContext context = new ParseContext();

      parser.parse(inputstream, handler, metadata, context);

      docs.put(file.getName(), handler.toString());
    }

    return docs;
  }

/*
 *   public static void main(final String[] args) throws Exception {
 *      String folderPath = "";
 *      try {
 *        folderPath = args[0];
 *      } catch (Exception e) {
 *        throw new Exception("Please specify a folder path!");
 *      }
 *
 *      File folder = new File(folderPath);
 *      if (folder.listFiles() == null) {
 *        throw new Exception("The path does not exist or the folder is empty!");
 *      }
 *
 *      File[] listOfFiles = folder.listFiles();
 *      for (File file : listOfFiles) {
 *        System.out.println(file.getName());
 *
 *        String fileName = folderPath + "/" + file.getName();
 *
 *        String ext = fileName.substring(fileName.lastIndexOf(".") + 1);
 *        System.out.println(ext);
 *        if (!"txt".equals(ext) && !"pdf".equals(ext) && !"doc".equals(ext) && !"docx".equals(ext)) {
 *          throw new Exception("The folder does not contain only txt, pdf and doc(x) files!");
 *        }
 *
 *        File currFile = new File(fileName);
 *
 *        Parser parser = new AutoDetectParser();
 *        BodyContentHandler handler = new BodyContentHandler();
 *        Metadata metadata = new Metadata();
 *        FileInputStream inputstream = new FileInputStream(currFile);
 *        ParseContext context = new ParseContext();
 *
 *        parser.parse(inputstream, handler, metadata, context);
 *        System.out.println("File content : " + handler.toString());
 *      }
 *   }
 */
}
