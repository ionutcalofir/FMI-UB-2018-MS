import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import org.apache.lucene.store.Directory;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

public class Main {
  public static void main(String[] args) throws Exception {
    // Check if the folder path was specified
    String folderPath = "";
    try {
      folderPath = args[0];
    } catch (Exception e) {
      throw new Exception("Please specify a folder path!");
    }

    // Create new TikaExtraction object to get the text from the specified files
    TikaExtractor tikaExtractor = new TikaExtractor();
    HashMap<String, String> docs = tikaExtractor.getDocs(folderPath);

    // Create a custom analyzer
    MyAnalyzer analyzer = new MyAnalyzer();

    // DEBUG
    // -------------------------------------------------------------------------
    // Iterate through each file
    System.out.println("\nProcessed documents:\n");
    for (String i : docs.keySet()) {

      List<String> result = new ArrayList<String>();
      String text = docs.get(i);
      TokenStream tokenStream = analyzer.tokenStream("text", text);
      CharTermAttribute attr = tokenStream.addAttribute(CharTermAttribute.class);
      tokenStream.reset();
      while(tokenStream.incrementToken()) {
        result.add(attr.toString());
      }
      tokenStream.close();
      System.out.println("Document name:\n" + i);
      System.out.println("Text:\n" + result);
      System.out.println();
    }
    // -------------------------------------------------------------------------

    // Create Indexer
    Indexer indexer = new Indexer();
    Directory index = indexer.buildIndex(docs, analyzer);

    // Create the query
    String rawQuery;
    try {
      rawQuery = args[1];
    } catch (Exception e) {
      throw new Exception("Please specify a query!");
    }
    MyQuery myQuery = new MyQuery();
    Query query = myQuery.buildQuery("text", analyzer, rawQuery);

    // DEBUG
    // -------------------------------------------------------------------------
    System.out.println("The processed query is:\n" + query + "\n");
    // -------------------------------------------------------------------------

    // Create the searcher
    int nDocs = 10;
    Searcher searcher = new Searcher();
    searcher.buildSearch(nDocs, index, query);
    searcher.showResults();
  }
}
