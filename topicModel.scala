import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.{Tokenizer, CountVectorizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession

object TopicModeling {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: inputPath outputPath")
    }

    val sc = new SparkContext(new SparkConf().setAppName("Spark TopicModeling"))
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val corpus = sc.textFile(args(0)).map(_.toLowerCase())

    val corpus_body = corpus.map(_.split("\\n")).map(_.mkString("\n"))

    val corpus_df = corpus_body.zipWithIndex.map(_.swap).flatMapValues(_.split("\\n")).toDF("id", "corpus")

    val corpus_DF = corpus_df.withColumn("id", monotonically_increasing_id())

    val tokenizer = new RegexTokenizer().setPattern("[\\W_]+").setMinTokenLength(4) // Filter away tokens with length < 4
      .setInputCol("corpus")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(corpus_DF)

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val removeEmpty = udf((array: Seq[String]) => !array.isEmpty)

    val tokenized_DF = tokenized_df.filter(removeEmpty(col("tokens")))

    val filtered_df = remover.transform(tokenized_DF)

    val cv = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinTF(10)
      .setMinDF(10)
      .setBinary(true)

    val cvFitted = cv.fit(filtered_df)

    val prepped = cvFitted.transform(filtered_df)

    val lda = new LDA().setK(5).setMaxIter(60)

    val model = lda.fit(prepped)

    val vocabList = cvFitted.vocabulary

    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

    val topics = model.describeTopics(maxTermsPerTopic = 6)
      .withColumn("terms", termsIdx2Str(col("termIndices")))

    topics.select("terms").rdd.saveAsTextFile(args(1))


  }
}
