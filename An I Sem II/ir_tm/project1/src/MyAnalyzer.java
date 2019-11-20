import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.tartarus.snowball.ext.RomanianStemmer;

public class MyAnalyzer extends Analyzer {
  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    ArrayList<String> words = new ArrayList<String>();
    try {
      BufferedReader reader = new BufferedReader(new FileReader("../utils/stopwords.txt"));
      String line = reader.readLine();
      while (line != null) {
        words.add(line);
        line = reader.readLine();
      }
      reader.close();
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    CharArraySet stopwords = new CharArraySet(words, false);

    StandardTokenizer source = new StandardTokenizer(); // tokenizer

    TokenStream filter = new LowerCaseFilter(source); // convert all letters to lowercase
    filter = new StopFilter(filter, stopwords); // remove stopwords
    filter = new SnowballFilter(filter, new RomanianStemmer()); // apply stemmer
    filter = new ASCIIFoldingFilter(filter); // remove diacritics

    return new TokenStreamComponents(source, filter);
  }
}
