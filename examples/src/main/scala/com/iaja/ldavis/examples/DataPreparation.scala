package com.iaja.ldavis.examples


import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.{Pipeline, WordLengthFilter}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.broadcast
import org.apache.spark.ml.linalg.{DenseVector, Vector => MLVector}
import org.slf4j.LoggerFactory
import org.apache.spark.sql.functions.udf

import scala.collection.mutable

case class NewsData(content: String)

trait Logger {
  val log = LoggerFactory.getLogger(this.getClass.getName)
}

/**
  * Created by mageswarand on 9/6/17.
  */
object DataPreparation extends Logger {

  val getSize: mutable.WrappedArray[String] => Int =  _.size

  val sizeUdf = udf(getSize)

  def main(args: Array[String]): Unit = {

    val vocabSize = 50000

    val spark = SparkSession
      .builder()
      .appName("DataPreparation").master("local[4]")
      .config("spark.sql.parquet.enableVectorizedReader", "false")
      .getOrCreate()

    val sc = spark.sparkContext
    import spark.implicits._

    sc.setLogLevel("ERROR")

    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    val df = spark
      .sparkContext
      .wholeTextFiles("resources/dataset/20news-bydate/20news-bydate-train/*")
      .map(x => NewsData(x._2))
      .toDS()

    df.show()

    val tokenizer = new Tokenizer()
      .setInputCol("content")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words_filtered")

    //Custom filter Transformer
    val filterOnlyText = new WordLengthFilter()
      .setInputCol("words_filtered")
      .setOutputCol("filteredWords")
      .setWordLength(3)

    val cv = new CountVectorizer()
      .setInputCol("filteredWords")
      .setOutputCol("features")
      .setVocabSize(vocabSize)

    val lda = new LDA()
      .setOptimizer("online")
      .setK(10)
      .setMaxIter(3)

    val pipeline = new Pipeline().setStages(
      Array(tokenizer, remover, filterOnlyText, cv, lda))

    log.info("Starting pipeline fit...")
    val pipeLineModel = pipeline.fit(df)

    val transformedDf = pipeLineModel.transform(df)

    //TODO can we move below code to scalLDAvis???
    val ldaModel = pipeLineModel.stages(4).asInstanceOf[LDAModel]
    val cvModel = pipeLineModel.stages(3).asInstanceOf[CountVectorizerModel]

    val vocab = pipeLineModel.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val vocabDf = sc.parallelize(vocab.zipWithIndex).toDF("term", "termIndex")

    //For scalaLDAVis
    cvModel.save("/tmp/scalaLDAvis/model/cv-model")
    ldaModel.save("/tmp/scalaLDAvis/model/spark-lda")
    transformedDf.write.json("/tmp/scalaLDAvis/model/transformedDF")

    transformedDf.
      withColumn("doc_size", sizeUdf(transformedDf("filteredWords"))).
      select("doc_size", "topicDistribution").
      filter($"doc_size" > 0).show()

    try {
      val (phi, thetaWithSize, termFreq, vocabRdd) =
        generateLDAVisData(spark, vocabSize, ldaModel, transformedDf, vocab)

      //For python LDAvis
      phi.saveAsTextFile("/tmp/scalaLDAvis/phi/")
      thetaWithSize.saveAsTextFile("/tmp/scalaLDAvis/theta")
      termFreq.saveAsTextFile("/tmp/scalaLDAvis/termFreq")
      vocabRdd.saveAsTextFile("/tmp/scalaLDAvis/vocab")
    } catch {
      case (e: Exception) => log.error("Exception occured while writing the LDAViz data", e)
    }

    spark.close()
    System.exit(0)
  }

  def generateLDAVisData(spark: SparkSession, vocabSize: Int,
                         ldaModel: LDAModel, transformedDf: DataFrame,
                         vocab: Array[String]):
  (RDD[String], RDD[String], RDD[Int], RDD[String]) = {
    import spark.implicits._
    val sc = spark.sparkContext

    transformedDf.cache()

    val topicsMatrix = ldaModel.topicsMatrix
    //  assert(vocabSize == topicsMatrix.numRows)

    val phiMatrix = topicsMatrix.transpose.rowIter.map(_.toDense.toArray).toArray
    val phiMatRdd = sc.parallelize(phiMatrix.map(_.mkString(","))).coalesce(1)

    val thetaMatWithDocSizesRdd = transformedDf.
      withColumn("doc_size", sizeUdf(transformedDf("filteredWords"))).
      select("doc_size", "topicDistribution").
      filter($"doc_size" > 0).
      rdd.
      map(r => r.getInt(0).toDouble :: r.get(1).asInstanceOf[MLVector].toArray.toList).
      map(_.mkString(",")).
      coalesce(100)

    val docTopicDist = transformedDf.select("topicDistribution")
      .rdd.map(r => r.get(0).asInstanceOf[MLVector].toArray.length)

    val freqArray = transformedDf.select("features").
      rdd.map(x => x.get(0).asInstanceOf[MLVector].toDense).
      reduce((a, b) =>
        new DenseVector(a.toArray.zip(b.toArray).map(x => x._1 + x._2))
      ).toArray.map(_.toInt)

    val termFreqRdd = sc.parallelize(freqArray).coalesce(1)
    val vocabRdd = sc.parallelize(vocab).coalesce(1)


    thetaMatWithDocSizesRdd.take(10).foreach(println)
    (phiMatRdd, thetaMatWithDocSizesRdd, termFreqRdd, vocabRdd)
  }
}
