import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.search.Query;
import org.apache.lucene.document.Document;

public class Searcher {
  public IndexReader reader;
  public IndexSearcher searcher;
  public TopDocs docs;
  public ScoreDoc[] bestDocs;

  public void buildSearch(int nDocs, Directory index, Query query) throws Exception {
    reader = DirectoryReader.open(index);
    searcher = new IndexSearcher(reader);
    docs = searcher.search(query, nDocs);
    bestDocs = docs.scoreDocs;
  }

  public void showResults() throws Exception {
    System.out.println("Found the following documents:");
    for (int i = 0; i < bestDocs.length; i++) {
      int docID = bestDocs[i].doc;
      Document doc = searcher.doc(docID);
      System.out.println("Document name: " + doc.get("docName"));
    }

    reader.close();
  }
}
