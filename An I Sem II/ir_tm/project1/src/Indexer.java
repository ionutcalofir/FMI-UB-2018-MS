import java.util.HashMap;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;

public class Indexer {
  public Directory buildIndex(HashMap<String, String> docs, MyAnalyzer analyzer) throws Exception {
    Directory index = new RAMDirectory();

    IndexWriterConfig config = new IndexWriterConfig(analyzer);
    IndexWriter w = new IndexWriter(index, config);
    for (String i : docs.keySet()) {
      addDoc(w, docs.get(i), i);
    }
    w.close();

    return index;
  }

  private static void addDoc(IndexWriter w, String text, String docName) throws Exception {
    Document doc = new Document();
    doc.add(new TextField("text", text, Field.Store.YES));

    doc.add(new StringField("docName", docName, Field.Store.YES));

    w.addDocument(doc);
  }
}
