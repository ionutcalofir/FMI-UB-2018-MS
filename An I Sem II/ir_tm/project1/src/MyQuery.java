import org.apache.lucene.search.Query;
import org.apache.lucene.queryparser.classic.QueryParser;

public class MyQuery {
  public Query buildQuery(String field, MyAnalyzer analyzer, String rawQuery) throws Exception {
    Query query = new QueryParser(field, analyzer).parse(rawQuery);

    return query;
  }
}
